"""Agent-facing schemas: what an agent receives (AgentTask) and returns (AgentResult)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import InputType, Priority
from app.schemas.message import AgentMessage, MessageType


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    agent: str
    status: AgentStatus
    result: dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    error: str | None = None
    error_type: str | None = None
    step_id: str | None = None
    attempts: int = 1
    models: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.COMPLETED

    @property
    def text(self) -> str:
        value = self.result.get("text")
        return value if isinstance(value, str) else ""


class AgentInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str
    role: str
    capabilities: list[str]
    keywords: list[str] = Field(default_factory=list)
    model_key: str | None = None
    enabled: bool = True


class AgentTask(BaseModel):
    """A single unit of work handed to an agent."""

    task_id: str
    step_id: str
    agent: str
    instruction: str
    prompt: str
    input_type: InputType = InputType.TEXT
    image_path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    messages: list[AgentMessage] = Field(default_factory=list)
    priority: Priority = Priority.NORMAL
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    device_override: str | None = None

    @property
    def mode(self) -> str | None:
        mode = self.params.get("mode")
        return str(mode) if mode else None

    def upstream(self) -> list[AgentMessage]:
        return [m for m in self.messages if m.message_type == MessageType.RESULT]

    def upstream_by_agent(self, agent: str) -> list[dict[str, Any]]:
        return [m.payload for m in self.upstream() if m.sender == agent]

    def upstream_text(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for m in self.upstream():
            text = m.payload.get("text")
            if isinstance(text, str) and text.strip():
                out.append((m.sender, text.strip()))
        return out
