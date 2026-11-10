"""In-process async event bus: agents publish, the WebSocket and CLI subscribe."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Events listed in the spec, plus a few extra the UI uses.
EVENT_TYPES = (
    "task_started", "task_completed", "task_failed", "task_stopped", "task_paused", "task_resumed",
    "state_changed", "plan_created", "subtask_started", "subtask_completed",
    "application_detected", "application_opened", "agent_created", "agent_message",
    "screen_updated", "observation", "action_started", "action_completed",
    "verification_started", "verification_completed", "recovery",
    "confirmation_required", "user_input_required", "user_response", "log",
)


class EventBus:
    def __init__(self, history: int = 500) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self.history: deque[dict[str, Any]] = deque(maxlen=history)

    def publish(self, event: str, **data: Any) -> dict[str, Any]:
        msg = {"event": event, "ts": time.time(), **data}
        if event != "screen_updated":  # frames are too big and too frequent to replay
            self.history.append(msg)
        for q in list(self._subscribers):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()  # slow consumer: drop its oldest event instead of blocking agents
            q.put_nowait(msg)
        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception:  # noqa: BLE001 - a broken listener must not break the agent
                logger.exception("event callback failed")
        return msg

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def on(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def recent(self, task_id: str | None = None, limit: int = 300) -> list[dict[str, Any]]:
        items = [e for e in self.history if task_id is None or e.get("task_id") == task_id]
        return items[-limit:]
