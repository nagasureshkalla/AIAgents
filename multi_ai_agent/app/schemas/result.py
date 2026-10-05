"""Response schemas for tasks, models and health."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent import AgentResult
from app.schemas.common import TaskStatus
from app.schemas.task import Plan, TaskRecord


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    answer: str | None = None
    result: dict[str, Any] | None = None
    plan: Plan | None = None
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    error: str | None = None
    duration: float | None = None

    @classmethod
    def from_record(cls, record: TaskRecord) -> "TaskResult":
        answer = record.result.get("answer") if record.result else None
        return cls(
            task_id=record.task_id,
            status=record.status,
            answer=answer,
            result=record.result,
            plan=record.plan,
            agent_results=record.agent_results,
            error=record.error,
            duration=record.duration,
        )


class ChatResponse(BaseModel):
    task_id: str
    status: TaskStatus
    answer: str | None
    error: str | None = None
    duration: float | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    type: str
    path: str
    hf_id: str | None = None
    installed: bool
    state: str
    device: str | None = None
    priority: str = "normal"
    source: str = "config"
    disk_size_gb: float | None = None
    memory_gb: float | None = None
    last_used: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu: str | None = None
    vram_total: str | None = None
    vram_used: str | None = None
    ram_total: str | None = None
    ram_used: str | None = None
    cpu_percent: float | None = None
    gpu_utilization: float | None = None
    agents_active: int
    tasks_running: int
    tasks_queued: int
    models_loaded: list[str]
    offline_mode: bool
    uptime_seconds: float
