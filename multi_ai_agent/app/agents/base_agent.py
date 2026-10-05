"""BaseAgent: the contract every agent implements, plus shared inference plumbing."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from app.core.config import AgentConfig, Settings
from app.core.device import empty_cache, is_oom_error
from app.core.exceptions import GPUOutOfMemoryError, classify_error
from app.core.logging import bind_context, log_event, reset_context
from app.models.base import GenerationOutput, GenerationRequest
from app.schemas.agent import AgentInfo, AgentResult, AgentStatus, AgentTask
from app.utils.text import truncate

if TYPE_CHECKING:
    from PIL.Image import Image

    from app.agents.registry import AgentRegistry
    from app.models.model_manager import ModelManager
    from app.orchestration.scheduler import ResourceScheduler
    from app.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

# Models used during the current agent execution (per asyncio task, so shared agent instances are safe).
_models_used: ContextVar[list[str] | None] = ContextVar("models_used", default=None)


@dataclass
class AgentContext:
    """Dependencies injected into every agent."""

    settings: Settings
    scheduler: "ResourceScheduler"
    models: "ModelManager"
    monitor: "PerformanceMonitor"
    registry: "AgentRegistry | None" = None


class BaseAgent(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    role: ClassVar[str] = "worker"  # "control" agents (manager/planner) coordinate; workers do task work
    capabilities: ClassVar[list[str]] = []
    keywords: ClassVar[list[str]] = []  # prompt keywords that make the manager consider this agent
    default_model: ClassVar[str | None] = None

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ metadata
    @property
    def config(self) -> AgentConfig:
        return self.ctx.settings.agent_config(self.name)

    @property
    def model_key(self) -> str | None:
        return self.config.model or self.default_model

    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            description=self.description,
            role=self.role,
            capabilities=list(self.capabilities),
            keywords=list(self.keywords),
            model_key=self.model_key,
            enabled=self.config.enabled,
        )

    # ------------------------------------------------------------------ contract
    @abstractmethod
    async def run(self, task: AgentTask) -> dict[str, Any]:
        """Do the agent's work and return its result payload (wrapped into AgentResult by `execute`)."""
        raise NotImplementedError

    async def execute(self, task: AgentTask) -> AgentResult:
        """Run with timeout, timing, error classification and metrics. Never raises except on cancellation."""
        tokens = bind_context(task_id=task.task_id, agent=self.name)
        used_token = _models_used.set([])
        wall_start, start = time.time(), time.perf_counter()
        timeout = task.timeout or self.config.timeout_seconds or self.ctx.settings.runtime.agent_timeout_seconds
        status, payload, error, error_type = AgentStatus.COMPLETED, {}, None, None
        log_event(logger, "started", step=task.step_id, message=truncate(task.instruction, 120))
        try:
            async with asyncio.timeout(timeout):
                payload = await self.run(task)
            if not isinstance(payload, dict):
                payload = {"text": str(payload)}
        except TimeoutError:
            status, error, error_type = AgentStatus.FAILED, f"Agent '{self.name}' timed out after {timeout:.0f}s", "timeout"
        except Exception as exc:  # noqa: BLE001 - one failed agent must not crash the system
            status, error, error_type = AgentStatus.FAILED, str(exc) or exc.__class__.__name__, classify_error(exc)
            logger.debug("agent %s failed", self.name, exc_info=True)
        finally:
            elapsed = time.perf_counter() - start
            models = list(_models_used.get() or [])
            _models_used.reset(used_token)
            reset_context(tokens)

        self.ctx.monitor.record(
            kind="agent",
            task_id=task.task_id,
            agent=self.name,
            model=",".join(models) or None,
            start_time=wall_start,
            end_time=wall_start + elapsed,
            status="ok" if status == AgentStatus.COMPLETED else "error",
            error=error,
        )
        log_event(
            logger,
            "completed" if status == AgentStatus.COMPLETED else "failed",
            level=logging.INFO if status == AgentStatus.COMPLETED else logging.WARNING,
            task_id=task.task_id,
            agent=self.name,
            duration=elapsed,
            status=status.value,
            step=task.step_id,
            message=error or "",
        )
        return AgentResult(
            agent=self.name,
            status=status,
            result=payload,
            execution_time=round(elapsed, 4),
            error=error,
            error_type=error_type,
            step_id=task.step_id,
            models=models,
        )

    # ------------------------------------------------------------------ helpers
    async def generate(
        self,
        task: AgentTask,
        user: str,
        *,
        system: str | None = None,
        image: "Image | None" = None,
        model_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationOutput:
        """Run one inference through the resource scheduler."""
        key = model_key or self.model_key
        if key is None:
            raise RuntimeError(f"Agent '{self.name}' has no model configured")
        models, gen = self.ctx.models, self.ctx.settings.generation
        resolved = models.resolve(key)
        spec = models.registry.get(resolved)

        limit = task.max_tokens or spec.max_tokens or gen.max_tokens
        if max_tokens is not None:
            limit = min(limit, max_tokens)
        temp = task.temperature if task.temperature is not None else temperature
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        request = GenerationRequest(
            messages=messages,
            max_new_tokens=limit,
            temperature=gen.temperature if temp is None else temp,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            image=image,
        )

        wall_start = time.time()
        device = None
        try:
            async with self.ctx.scheduler.acquire(
                resolved,
                priority=task.priority,
                device=task.device_override,
                task_id=task.task_id,
                agent=self.name,
            ) as lease:
                device = lease.device
                output = await lease.run(lease.model.generate_sync, request)
        except Exception as exc:
            self.ctx.monitor.record(
                kind="inference",
                task_id=task.task_id,
                agent=self.name,
                model=resolved,
                device=device,
                start_time=wall_start,
                end_time=time.time(),
                status="error",
                error=str(exc),
            )
            if is_oom_error(exc) and not isinstance(exc, GPUOutOfMemoryError):
                empty_cache(device)
                raise GPUOutOfMemoryError(f"Out of memory running '{resolved}' on {device}: {exc}") from exc
            raise

        self.ctx.monitor.record(
            kind="inference",
            task_id=task.task_id,
            agent=self.name,
            model=output.model,
            device=output.device,
            start_time=wall_start,
            end_time=wall_start + output.latency,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
        )
        used = _models_used.get()
        if used is not None and output.model not in used:
            used.append(output.model)
        return output

    @staticmethod
    def render_upstream(task: AgentTask, limit: int = 6000) -> str:
        """Format results received from upstream agents as prompt context."""
        blocks: list[str] = []
        for msg in task.upstream():
            payload = msg.payload
            text = payload.get("text")
            body = text if isinstance(text, str) and text.strip() else json.dumps(payload, default=str)[:1500]
            extras = []
            if payload.get("objects"):
                extras.append("Objects: " + ", ".join(map(str, payload["objects"])))
            header = f"### From {msg.sender} agent ({payload.get('step_id', '')})"
            blocks.append("\n".join([header, body.strip(), *extras]))
        return truncate("\n\n".join(blocks), limit)
