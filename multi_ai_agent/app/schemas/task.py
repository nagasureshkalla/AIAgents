"""Task, plan and request schemas."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.agent import AgentResult
from app.schemas.common import InputType, Priority, TaskStatus


class TaskOptions(BaseModel):
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0, le=2)
    agents: list[str] | None = Field(
        default=None, description="Force a specific set of agents (run in parallel) instead of automatic routing."
    )
    skip_validation: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra per-request parameters for worker agents (e.g. research depth)."
    )


class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user's instruction or question.")
    input_type: InputType | None = Field(default=None, description="text | image | multimodal (inferred when omitted).")
    image_path: str | None = Field(
        default=None, description="Path to an image inside data/input (e.g. ./data/input/image.png)."
    )
    priority: Priority = Priority.NORMAL
    options: TaskOptions = Field(default_factory=TaskOptions)

    @field_validator("priority", mode="before")
    @classmethod
    def _parse_priority(cls, v: Any) -> Priority:
        return Priority.parse(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"prompt": "Summarize this document: ...", "input_type": "text", "priority": "NORMAL"},
                {
                    "prompt": "Analyze this image and explain what is happening.",
                    "input_type": "image",
                    "image_path": "./data/input/image.png",
                    "priority": "HIGH",
                },
            ]
        }
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    image_path: str | None = None
    priority: Priority = Priority.HIGH
    options: TaskOptions = Field(default_factory=TaskOptions)
    wait_timeout: float = Field(default=300, gt=0, le=3600)

    @field_validator("priority", mode="before")
    @classmethod
    def _parse_priority(cls, v: Any) -> Priority:
        return Priority.parse(v)


class AnalyzeRequest(BaseModel):
    instruction: str = Field(default="Analyze the following input and explain the key points.")
    text: str | None = None
    image_path: str | None = None
    priority: Priority = Priority.NORMAL
    wait_timeout: float = Field(default=300, gt=0, le=3600)

    @field_validator("priority", mode="before")
    @classmethod
    def _parse_priority(cls, v: Any) -> Priority:
        return Priority.parse(v)


class PlanStep(BaseModel):
    id: str
    agent: str
    task: str
    capability: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    optional: bool = False
    timeout: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    task_id: str
    intent: str
    steps: list[PlanStep]
    parallel_groups: list[list[str]] = Field(default_factory=list)

    def step(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def compute_parallel_groups(self) -> list[list[str]]:
        """Topological levels: steps in the same group have no dependency on each other."""
        remaining = {s.id: set(s.depends_on) for s in self.steps}
        done: set[str] = set()
        groups: list[list[str]] = []
        while remaining:
            level = sorted(sid for sid, deps in remaining.items() if deps <= done)
            if not level:
                raise ValueError(f"Plan has cyclic or unknown dependencies: {sorted(remaining)}")
            groups.append(level)
            done.update(level)
            for sid in level:
                remaining.pop(sid)
        return groups


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.QUEUED
    request: TaskRequest
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    decision: dict[str, Any] | None = None
    plan: Plan | None = None
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        return round((self.completed_at or time.time()) - self.started_at, 3)


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: TaskStatus
