from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agents.base_agent import BaseAgent
from app.core.exceptions import TaskExecutionError
from app.orchestration.scheduler import PriorityPool
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.common import Priority, TaskStatus
from app.schemas.task import Plan, PlanStep, TaskOptions, TaskRecord, TaskRequest
from tests.fakes import CUDA_DEVICE, FakeFactory, build_container, make_image, make_settings


def events_of(container, task_id: str) -> list[dict]:
    return [e.to_message() for e in container.events.history(task_id)]


def index_of(events: list[dict], event: str, step_id: str) -> int:
    return next(i for i, e in enumerate(events) if e["event"] == event and e.get("step_id") == step_id)


# ---------------------------------------------------------------- end to end
async def test_text_task_runs_manager_planner_agents_aggregator(container):
    record = container.submit(TaskRequest(prompt="Summarize this document: local models are fast."))
    assert record.status == TaskStatus.QUEUED

    done = await container.store.wait(record.task_id, timeout=10)
    assert done.status == TaskStatus.COMPLETED, done.error
    assert done.plan.intent == "summarize"
    assert set(done.agent_results) >= {"manager", "planner", "summary", "analysis", "validate"}
    assert done.result["answer"]
    assert done.result["synthesized"] is True  # two content agents -> base model synthesis

    names = [e["event"] for e in events_of(container, record.task_id)]
    assert names[0] == "task_queued"
    assert names[-1] == "task_completed"
    assert "plan_created" in names


async def test_independent_steps_run_concurrently(tmp_path):
    c = build_container(make_settings(tmp_path), factory=FakeFactory(latency=0.2))
    await c.start()
    try:
        record = c.submit(TaskRequest(prompt="Summarize this: concurrency matters."))
        await c.store.wait(record.task_id, timeout=10)
        events = events_of(c, record.task_id)
        # Both first-wave steps start before either finishes.
        first_completion = min(
            index_of(events, "agent_completed", "summary"), index_of(events, "agent_completed", "analysis")
        )
        assert index_of(events, "agent_started", "summary") < first_completion
        assert index_of(events, "agent_started", "analysis") < first_completion
    finally:
        await c.stop()


async def test_dependencies_are_respected_for_image_task(container):
    img = make_image(container.settings.input_dir)
    record = container.submit(
        TaskRequest(prompt="Analyze this image and explain what is happening.", image_path=str(img))
    )
    done = await container.store.wait(record.task_id, timeout=10)
    assert done.status == TaskStatus.COMPLETED, done.error
    assert done.plan.parallel_groups == [["intent", "see"], ["reason"], ["validate"]]

    events = events_of(container, record.task_id)
    reason_start = index_of(events, "agent_started", "reason")
    assert index_of(events, "agent_completed", "see") < reason_start
    assert index_of(events, "agent_completed", "intent") < reason_start
    assert index_of(events, "agent_completed", "reason") < index_of(events, "agent_started", "validate")
    assert done.result["vision"]["objects"] == ["cat", "sofa", "lamp"]
    assert "vision" in done.result["agents_used"]


# ---------------------------------------------------------------- failure handling
class AlwaysFails(BaseAgent):
    name = "always_fails"
    capabilities = ["failing"]

    async def run(self, task: AgentTask) -> dict[str, Any]:
        raise RuntimeError("boom")


class FailsOnce(BaseAgent):
    name = "fails_once"
    capabilities = ["flaky"]

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.calls = 0

    async def run(self, task: AgentTask) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient failure")
        return {"text": "recovered"}


def make_record(prompt: str = "hello") -> TaskRecord:
    return TaskRecord(task_id="task_manual", request=TaskRequest(prompt=prompt))


async def test_optional_step_failure_is_skipped(container):
    container.agents.register(AlwaysFails)
    plan = Plan(
        task_id="task_manual",
        intent="custom",
        steps=[
            PlanStep(id="a", agent="always_fails", task="fail", optional=True),
            PlanStep(id="b", agent="text", task="answer", params={"mode": "generate"}),
            PlanStep(id="c", agent="validation", task="check", capability="validation", depends_on=["b", "a"]),
        ],
    )
    results = await container.orchestrator.execute_plan(make_record(), plan)
    assert results["a"].status == AgentStatus.SKIPPED
    assert results["a"].attempts == 2  # retried once (max_retries=1) before skipping
    assert results["b"].ok and results["c"].ok
    upstream_issue = [i for i in results["c"].result["issues"] if i["check"] == "upstream_failure"]
    assert upstream_issue and "boom" in upstream_issue[0]["message"]


async def test_required_step_failure_fails_task(container):
    container.agents.register(AlwaysFails)
    plan = Plan(task_id="task_manual", intent="custom", steps=[PlanStep(id="a", agent="always_fails", task="fail")])
    with pytest.raises(TaskExecutionError):
        await container.orchestrator.execute_plan(make_record(), plan)


async def test_transient_failure_is_retried(container):
    agent = container.agents.register(FailsOnce)
    plan = Plan(task_id="task_manual", intent="custom", steps=[PlanStep(id="a", agent="fails_once", task="x")])
    results = await container.orchestrator.execute_plan(make_record(), plan)
    assert results["a"].ok
    assert results["a"].attempts == 2
    assert agent.calls == 2


async def test_inference_oom_falls_back_to_cpu(tmp_path):
    factory = FakeFactory(oom_on_generate={"fast_text"})
    settings = make_settings(tmp_path, runtime={"device": "cuda"})
    c = build_container(settings, factory=factory, device=CUDA_DEVICE)
    plan = Plan(
        task_id="task_manual",
        intent="custom",
        steps=[PlanStep(id="a", agent="text", task="answer", params={"mode": "generate"})],
    )
    results = await c.orchestrator.execute_plan(make_record(), plan)
    assert results["a"].ok, results["a"].error
    assert results["a"].attempts == 2
    assert "fast_text@cpu" in c.models.loaded_models()


async def test_failed_task_reports_error(container):
    container.agents.register(AlwaysFails)
    record = container.submit(TaskRequest(prompt="x", options=TaskOptions(agents=["always_fails"])))
    done = await container.store.wait(record.task_id, timeout=10)
    assert done.status == TaskStatus.FAILED
    assert "boom" in done.error
    assert events_of(container, record.task_id)[-1]["event"] == "task_failed"


# ---------------------------------------------------------------- cancellation & priority
async def test_cancel_running_task(tmp_path):
    c = build_container(make_settings(tmp_path), factory=FakeFactory(latency=0.5))
    await c.start()
    try:
        record = c.submit(TaskRequest(prompt="Explain why slow things are slow"))
        for _ in range(100):
            if c.store.get(record.task_id).status == TaskStatus.RUNNING:
                break
            await asyncio.sleep(0.01)
        assert c.cancel(record.task_id) is True
        done = await c.store.wait(record.task_id, timeout=5)
        assert done.status == TaskStatus.CANCELLED
        assert events_of(c, record.task_id)[-1]["event"] == "task_cancelled"
    finally:
        await c.stop()


async def test_cancel_queued_task(tmp_path):
    settings = make_settings(tmp_path)
    c = build_container(settings)  # workers not started: tasks stay queued
    record = c.submit(TaskRequest(prompt="hello"))
    assert c.cancel(record.task_id) is True
    assert c.store.get(record.task_id).status == TaskStatus.CANCELLED
    assert c.cancel(record.task_id) is False


async def test_priority_pool_serves_highest_priority_first():
    pool = PriorityPool("gpu", capacity=1)
    await pool.acquire(Priority.NORMAL)
    order: list[str] = []

    async def waiter(name: str, prio: Priority) -> None:
        await pool.acquire(prio)
        order.append(name)
        pool.release()

    tasks = [asyncio.create_task(waiter("low", Priority.LOW))]
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(waiter("critical", Priority.CRITICAL)))
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(waiter("normal", Priority.NORMAL)))
    await asyncio.sleep(0.01)
    pool.release()
    await asyncio.gather(*tasks)
    assert order == ["critical", "normal", "low"]


async def test_cancelled_waiter_does_not_leak_slot():
    pool = PriorityPool("gpu", capacity=1)
    await pool.acquire()
    waiter = asyncio.create_task(pool.acquire(Priority.HIGH))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    pool.release()
    assert pool.in_use == 0
    await asyncio.wait_for(pool.acquire(), timeout=1)
