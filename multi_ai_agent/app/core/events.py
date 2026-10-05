"""In-process per-task event bus used for WebSocket streaming and status replay."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    TASK_QUEUED = "task_queued"
    TASK_STARTED = "task_started"
    PLAN_CREATED = "plan_created"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_RETRY = "agent_retry"
    AGENT_SKIPPED = "agent_skipped"
    AGGREGATION_STARTED = "aggregation_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


TERMINAL_EVENTS = frozenset({EventType.TASK_COMPLETED, EventType.TASK_FAILED, EventType.TASK_CANCELLED})


class TaskEvent(BaseModel):
    task_id: str
    event: EventType
    agent: str | None = None
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.event in TERMINAL_EVENTS

    def to_message(self) -> dict[str, Any]:
        """Flat wire format, e.g. {"task_id": ..., "event": "agent_started", "agent": "vision"}."""
        msg: dict[str, Any] = {"task_id": self.task_id, "event": self.event.value}
        if self.agent is not None:
            msg["agent"] = self.agent
        msg["timestamp"] = self.timestamp
        msg.update(self.data)
        return msg


class EventBus:
    def __init__(self, history_per_task: int = 300, max_tasks: int = 1000, queue_size: int = 1000) -> None:
        self._history: OrderedDict[str, deque[TaskEvent]] = OrderedDict()
        self._subscribers: dict[str, set[asyncio.Queue[TaskEvent]]] = {}
        self._history_per_task = history_per_task
        self._max_tasks = max_tasks
        self._queue_size = queue_size

    def publish(self, event: TaskEvent) -> None:
        history = self._history.get(event.task_id)
        if history is None:
            history = deque(maxlen=self._history_per_task)
            self._history[event.task_id] = history
            while len(self._history) > self._max_tasks:
                self._history.popitem(last=False)
        history.append(event)
        for queue in list(self._subscribers.get(event.task_id, ())):
            if queue.full():  # slow consumer: drop oldest rather than block the orchestrator
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    def emit(self, task_id: str, event: EventType, agent: str | None = None, **data: Any) -> TaskEvent:
        evt = TaskEvent(task_id=task_id, event=event, agent=agent, data=data)
        self.publish(evt)
        return evt

    def subscribe(self, task_id: str, replay: bool = True) -> asyncio.Queue[TaskEvent]:
        queue: asyncio.Queue[TaskEvent] = asyncio.Queue(maxsize=self._queue_size)
        if replay:
            for evt in self._history.get(task_id, ()):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(evt)
        self._subscribers.setdefault(task_id, set()).add(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue[TaskEvent]) -> None:
        subs = self._subscribers.get(task_id)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            self._subscribers.pop(task_id, None)

    def history(self, task_id: str) -> list[TaskEvent]:
        return list(self._history.get(task_id, ()))

    def subscriber_count(self, task_id: str | None = None) -> int:
        if task_id is not None:
            return len(self._subscribers.get(task_id, ()))
        return sum(len(s) for s in self._subscribers.values())
