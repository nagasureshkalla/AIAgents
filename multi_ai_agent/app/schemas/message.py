"""Structured agent-to-agent messages. Agents never touch each other's state directly."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(StrEnum):
    REQUEST = "request"
    RESULT = "result"
    ERROR = "error"
    INFO = "info"


class AgentMessage(BaseModel):
    task_id: str
    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
