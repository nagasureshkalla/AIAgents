"""Orchestrator: Manager -> Planner -> concurrent DAG execution -> Validation -> Aggregation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.registry import AgentRegistry
from app.core.config import Settings
from app.core.device import DeviceInfo
from app.core.events import EventBus, EventType
from app.core.exceptions import TaskExecutionError
from app.core.logging import bind_context, log_event, reset_context
from app.orchestration.result_aggregator import ResultAggregator
from app.orchestration.task_queue import TaskStore
from app.schemas.agent import AgentResult, AgentStatus, AgentTask
from app.schemas.common import InputType, TaskStatus
from app.schemas.message import AgentMessage, MessageType
from app.schemas.task import Plan, PlanStep, TaskRecord
from app.utils.text import truncate

logger = logging.getLogger(__name__)

# Recovery policy per error type (see `_next_action`).
_RETRYABLE = {"timeout", "agent_error", "model_load", "oom", "cuda", "resources"}
_NOT_RECOVERABLE = {"invalid_input", "missing_model"}


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        agents: AgentRegistry,
        aggregator: ResultAggregator,
        events: EventBus,
        store: TaskStore,
        device_info: DeviceInfo,
    ) -> None:
        self.settings = settings
        self.agents = agents
        self.aggregator = aggregator
        self.events = events
        self.store = store
        self.device_info = device_info
        self._agent_slots = asyncio.Semaphore(settings.runtime.max_concurrent_agents)
        self._active_agents = 0

    @property
    def active_agents(self) -> int:
        return self._active_agents

    # ------------------------------------------------------------------ task lifecycle
    def _emit(self, task_id: str, event: EventType, agent: str | None = None, **data: Any) -> None:
        self.events.emit(task_id, event, agent, **data)
        level = logging.WARNING if event in (EventType.AGENT_FAILED, EventType.TASK_FAILED) else logging.INFO
        log_event(
            logger,
            event.value,
            level=level,
            task_id=task_id,
            agent=agent,
            duration=data.get("execution_time") or data.get("duration"),
            status=data.get("status"),
            step=data.get("step_id"),
        )

    async def run_task(self, record: TaskRecord) -> TaskRecord:
        task_id = record.task_id
        ctx_tokens = bind_context(task_id=task_id)
        record.status = TaskStatus.RUNNING
        record.started_at = time.time()
        self.store.save(record)
        self._emit(task_id, EventType.TASK_STARTED, priority=record.request.priority.value)
        timeout = record.request.options.timeout_seconds or self.settings.runtime.task_timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                record.result = await self._run(record)
            record.status = TaskStatus.COMPLETED
            record.completed_at = time.time()
            self._emit(task_id, EventType.TASK_COMPLETED, result=record.result, duration=record.duration)
        except asyncio.CancelledError:
            record.status = TaskStatus.CANCELLED
            record.error = "Task cancelled"
            record.completed_at = time.time()
            self._emit(task_id, EventType.TASK_CANCELLED)
            raise
        except TimeoutError:
            self._fail(record, f"Task timed out after {timeout:.0f}s")
        except TaskExecutionError as exc:
            self._fail(record, exc.message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s crashed", task_id)
            self._fail(record, f"{exc.__class__.__name__}: {exc}")
        finally:
            record.completed_at = record.completed_at or time.time()
            self.store.save(record)
            self.store.mark_done(task_id)
            reset_context(ctx_tokens)
        return record

    def _fail(self, record: TaskRecord, message: str) -> None:
        record.status = TaskStatus.FAILED
        record.error = message
        record.completed_at = time.time()
        self._emit(record.task_id, EventType.TASK_FAILED, error=message, duration=record.duration)

    def _base_task(self, record: TaskRecord, agent: str, step_id: str, instruction: str) -> AgentTask:
        req = record.request
        input_type = req.input_type or (InputType.IMAGE if req.image_path else InputType.TEXT)
        return AgentTask(
            task_id=record.task_id,
            step_id=step_id,
            agent=agent,
            instruction=instruction,
            prompt=req.prompt,
            input_type=input_type,
            image_path=req.image_path,
            priority=req.priority,
            max_tokens=req.options.max_tokens,
            temperature=req.options.temperature,
        )

    async def _run(self, record: TaskRecord) -> dict[str, Any]:
        opts = record.request.options

        # 1. Manager: understand the request and pick agents.
        manager_task = self._base_task(record, "manager", "manager", "Analyze the request and select agents")
        manager_task.params = {"forced_agents": opts.agents, "skip_validation": opts.skip_validation}
        decision = await self._run_control(record, self.agents.get("manager"), manager_task)
        record.decision = decision.result

        # 2. Planner: build the dependency graph from the manager's decision.
        planner_task = self._base_task(record, "planner", "planner", "Create an execution plan")
        planner_task.params = {"skip_validation": opts.skip_validation}
        planner_task.messages = [
            AgentMessage(
                task_id=record.task_id,
                sender="manager",
                receiver="planner",
                message_type=MessageType.RESULT,
                payload=decision.result,
            )
        ]
        plan_result = await self._run_control(record, self.agents.get("planner"), planner_task)
        plan = Plan.model_validate(plan_result.result["plan"])
        record.plan = plan
        self.store.save(record)
        self._emit(
            record.task_id,
            EventType.PLAN_CREATED,
            intent=plan.intent,
            steps=[s.model_dump() for s in plan.steps],
            parallel_groups=plan.parallel_groups,
        )

        # 3. Execute steps concurrently wherever dependencies allow.
        results = await self.execute_plan(record, plan)

        # 4. Aggregate.
        self._emit(record.task_id, EventType.AGGREGATION_STARTED, agent="aggregator")
        agg_task = self._base_task(record, "aggregator", "aggregate", "Synthesize the final answer")
        return await self.aggregator.aggregate(agg_task, plan, results)

    async def _run_control(self, record: TaskRecord, agent: BaseAgent, task: AgentTask) -> AgentResult:
        self._emit(record.task_id, EventType.AGENT_STARTED, agent.name, step_id=task.step_id, task=task.instruction)
        result = await agent.execute(task)
        record.agent_results[task.step_id] = result
        if not result.ok:
            self._emit(record.task_id, EventType.AGENT_FAILED, agent.name, step_id=task.step_id, error=result.error)
            raise TaskExecutionError(f"{agent.name} agent failed: {result.error}")
        self._emit(
            record.task_id,
            EventType.AGENT_COMPLETED,
            agent.name,
            step_id=task.step_id,
            status=result.status.value,
            execution_time=result.execution_time,
            summary=truncate(result.text, 300),
        )
        return result

    # ------------------------------------------------------------------ DAG execution
    async def execute_plan(self, record: TaskRecord, plan: Plan) -> dict[str, AgentResult]:
        """Run every step as soon as its dependencies finish; independent steps run concurrently."""
        results: dict[str, AgentResult] = {}
        pending: dict[str, PlanStep] = {s.id: s for s in plan.steps}
        running: dict[asyncio.Task[AgentResult], PlanStep] = {}
        try:
            while pending or running:
                ready = [s for s in pending.values() if all(d in results for d in s.depends_on)]
                for step in ready:
                    pending.pop(step.id)
                    job = asyncio.create_task(self._run_step(record, plan, step, results), name=f"{record.task_id}:{step.id}")
                    running[job] = step
                if not running:
                    raise TaskExecutionError(f"Plan has unsatisfiable dependencies: {sorted(pending)}")

                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for job in done:
                    step = running.pop(job)
                    result = job.result()
                    results[step.id] = result
                    record.agent_results[step.id] = result
                    if not result.ok and not step.optional:
                        raise TaskExecutionError(
                            f"Required step '{step.id}' ({step.agent}) failed: {result.error}",
                            details={"step_id": step.id, "error_type": result.error_type},
                        )
                self.store.save(record)
        finally:
            for job in running:
                job.cancel()
            for job in running:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await job
        return results

    def _step_task(self, record: TaskRecord, plan: Plan, step: PlanStep, results: dict[str, AgentResult]) -> AgentTask:
        task = self._base_task(record, step.agent, step.id, step.task)
        failed = [
            {"step_id": sid, "agent": r.agent, "error": r.error}
            for sid, r in results.items()
            if not r.ok and sid in self._ancestors(plan, step)
        ]
        task.params = {**step.params, "failed_steps": failed}
        task.timeout = step.timeout
        # Messages follow `depends_on` order: the validation agent treats the first one as the candidate answer.
        task.messages = [
            AgentMessage(
                task_id=record.task_id,
                sender=results[dep].agent,
                receiver=step.agent,
                message_type=MessageType.RESULT,
                payload={"step_id": dep, **results[dep].result},
            )
            for dep in step.depends_on
            if results[dep].ok
        ]
        return task

    @staticmethod
    def _ancestors(plan: Plan, step: PlanStep) -> set[str]:
        seen: set[str] = set()
        stack = list(step.depends_on)
        while stack:
            sid = stack.pop()
            if sid in seen:
                continue
            seen.add(sid)
            stack.extend(plan.step(sid).depends_on)
        return seen

    async def _run_step(
        self, record: TaskRecord, plan: Plan, step: PlanStep, results: dict[str, AgentResult]
    ) -> AgentResult:
        agent = self.agents.get(step.agent)
        task = self._step_task(record, plan, step, results)
        attempt = 0
        while True:
            attempt += 1
            async with self._agent_slots:
                self._active_agents += 1
                try:
                    self._emit(
                        record.task_id,
                        EventType.AGENT_STARTED,
                        agent.name,
                        step_id=step.id,
                        task=step.task,
                        attempt=attempt,
                        device=task.device_override,
                    )
                    result = await agent.execute(task)
                finally:
                    self._active_agents -= 1
            result.attempts = attempt

            if result.ok:
                self._emit(
                    record.task_id,
                    EventType.AGENT_COMPLETED,
                    agent.name,
                    step_id=step.id,
                    status=result.status.value,
                    execution_time=result.execution_time,
                    summary=truncate(result.text, 300),
                )
                return result

            self._emit(
                record.task_id,
                EventType.AGENT_FAILED,
                agent.name,
                step_id=step.id,
                status=result.status.value,
                error=result.error,
                error_type=result.error_type,
                attempt=attempt,
            )
            action = self._next_action(result, step, attempt, task.device_override)
            if action == "retry":
                self._emit(record.task_id, EventType.AGENT_RETRY, agent.name, step_id=step.id, action="retry")
                await asyncio.sleep(min(2.0, 0.25 * attempt))
                continue
            if action == "fallback_cpu":
                task.device_override = "cpu"
                self._emit(record.task_id, EventType.AGENT_RETRY, agent.name, step_id=step.id, action="fallback_cpu")
                continue
            if action == "skip":
                result.status = AgentStatus.SKIPPED
                self._emit(record.task_id, EventType.AGENT_SKIPPED, agent.name, step_id=step.id, error=result.error)
            return result  # "cancel": a failed required step; execute_plan stops the task

    def _next_action(self, result: AgentResult, step: PlanStep, attempt: int, device_override: str | None) -> str:
        """retry | fallback_cpu | skip | cancel."""
        rt = self.settings.runtime
        error_type = result.error_type or "agent_error"
        give_up = "skip" if step.optional else "cancel"
        if error_type in _NOT_RECOVERABLE:
            return give_up
        if (
            error_type in ("oom", "cuda")
            and rt.allow_cpu_fallback
            and device_override != "cpu"
            and self.device_info.type != "cpu"
        ):
            return "fallback_cpu"
        if error_type in _RETRYABLE and attempt <= rt.max_retries:
            return "retry"
        return give_up
