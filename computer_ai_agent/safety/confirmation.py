"""Human-in-the-loop: approvals, multiple-choice questions, and pause / resume / stop for a running task."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.events import EventBus


class TaskStopped(Exception):
    """Raised inside agent loops when the user presses Stop."""


@dataclass
class PendingRequest:
    id: str
    task_id: str
    kind: str                  # "confirm" | "choice" | "input"
    message: str
    choices: list[str] = field(default_factory=list)
    agent: str = ""
    category: str = ""
    created: float = field(default_factory=time.time)
    future: asyncio.Future = field(default=None, repr=False)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "task_id": self.task_id, "kind": self.kind, "message": self.message,
                "choices": self.choices, "agent": self.agent, "category": self.category, "created": self.created}


class TaskControl:
    """Checked between every observe/plan/act step."""

    def __init__(self, bus: EventBus, task_id: str) -> None:
        self.bus = bus
        self.task_id = task_id
        self._running = asyncio.Event()
        self._running.set()
        self.stopped = False

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def pause(self) -> None:
        if not self.stopped and self._running.is_set():
            self._running.clear()
            self.bus.publish("task_paused", task_id=self.task_id)

    def resume(self) -> None:
        if not self._running.is_set():
            self._running.set()
            self.bus.publish("task_resumed", task_id=self.task_id)

    def stop(self) -> None:
        self.stopped = True
        self._running.set()  # wake a paused loop so it can see the stop

    async def checkpoint(self) -> None:
        if self.stopped:
            raise TaskStopped()
        await self._running.wait()
        if self.stopped:
            raise TaskStopped()


class ConfirmationBroker:
    """Agents await `confirm()` / `choose()`; the web UI or CLI resolves them via `respond()`."""

    def __init__(self, bus: EventBus, timeout: float = 300.0) -> None:
        self.bus = bus
        self.timeout = timeout
        self.pending: dict[str, PendingRequest] = {}

    async def _ask(self, req: PendingRequest, screenshot_jpeg: bytes | None, event: str) -> Any:
        loop = asyncio.get_running_loop()
        req.future = loop.create_future()
        self.pending[req.id] = req
        payload = req.to_dict()
        if screenshot_jpeg:
            payload["screenshot"] = base64.b64encode(screenshot_jpeg).decode()
        self.bus.publish(event, task_id=req.task_id, request=payload)
        try:
            return await asyncio.wait_for(req.future, timeout=self.timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending.pop(req.id, None)

    async def confirm(self, task_id: str, message: str, agent: str = "", category: str = "",
                      screenshot_jpeg: bytes | None = None) -> bool:
        req = PendingRequest(uuid.uuid4().hex[:10], task_id, "confirm", message, ["Approve", "Deny"], agent, category)
        answer = await self._ask(req, screenshot_jpeg, "confirmation_required")
        approved = answer is True or (isinstance(answer, str) and answer.lower() in ("approve", "yes", "y", "true"))
        self.bus.publish("user_response", task_id=task_id, request_id=req.id, approved=approved,
                         answer="approve" if approved else ("timeout" if answer is None else "deny"))
        return approved

    async def choose(self, task_id: str, question: str, choices: list[str], agent: str = "",
                     screenshot_jpeg: bytes | None = None) -> str | None:
        kind = "choice" if choices else "input"
        req = PendingRequest(uuid.uuid4().hex[:10], task_id, kind, question, choices, agent)
        answer = await self._ask(req, screenshot_jpeg, "user_input_required")
        self.bus.publish("user_response", task_id=task_id, request_id=req.id, answer=answer)
        return None if answer is None else str(answer)

    def respond(self, request_id: str, answer: Any) -> bool:
        req = self.pending.get(request_id)
        if req is None or req.future is None or req.future.done():
            return False
        req.future.get_loop().call_soon_threadsafe(req.future.set_result, answer)
        return True

    def cancel_task(self, task_id: str) -> None:
        for req in list(self.pending.values()):
            if req.task_id == task_id and req.future is not None and not req.future.done():
                req.future.get_loop().call_soon_threadsafe(req.future.set_result, None)

    def list(self, task_id: str | None = None) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.pending.values() if task_id is None or r.task_id == task_id]
