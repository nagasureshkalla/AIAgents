"""Enums shared across schema modules."""

from __future__ import annotations

from enum import StrEnum


class Priority(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"

    @property
    def rank(self) -> int:
        """Lower rank is served first."""
        return _PRIORITY_RANK[self]

    @classmethod
    def parse(cls, value: "str | Priority | None") -> "Priority":
        if value is None:
            return cls.NORMAL
        if isinstance(value, Priority):
            return value
        return cls(str(value).strip().upper())


_PRIORITY_RANK = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.NORMAL: 2, Priority.LOW: 3}


class InputType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
