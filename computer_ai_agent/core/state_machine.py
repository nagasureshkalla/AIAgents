"""Explicit agent state machine. Every transition is validated, timestamped and published."""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.events import EventBus


class AgentState(str, Enum):
    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    APPLICATION_DETECTION = "APPLICATION_DETECTION"
    APPLICATION_LAUNCH = "APPLICATION_LAUNCH"
    AGENT_CREATION = "AGENT_CREATION"
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


S = AgentState
_ACTIVE = {S.OBSERVING, S.PLANNING, S.EXECUTING, S.VERIFYING, S.RECOVERING}
# Any working state may also pause for the user, fail, or (for the master) move on to the next subtask.
TRANSITIONS: dict[AgentState, set[AgentState]] = {
    S.IDLE: {S.UNDERSTANDING},
    S.UNDERSTANDING: {S.APPLICATION_DETECTION, S.PLANNING, S.WAITING_FOR_USER, S.FAILED, S.COMPLETED},
    S.APPLICATION_DETECTION: {S.APPLICATION_LAUNCH, S.AGENT_CREATION, S.WAITING_FOR_USER, S.FAILED},
    S.APPLICATION_LAUNCH: {S.AGENT_CREATION, S.APPLICATION_DETECTION, S.RECOVERING, S.WAITING_FOR_USER, S.FAILED},
    S.AGENT_CREATION: {S.OBSERVING, S.FAILED},
    S.OBSERVING: {S.PLANNING, S.VERIFYING, S.RECOVERING, S.WAITING_FOR_USER, S.FAILED, S.COMPLETED},
    S.PLANNING: {S.EXECUTING, S.OBSERVING, S.VERIFYING, S.WAITING_FOR_USER, S.RECOVERING, S.COMPLETED, S.FAILED},
    S.EXECUTING: {S.OBSERVING, S.VERIFYING, S.RECOVERING, S.WAITING_FOR_USER, S.FAILED},
    S.VERIFYING: {S.OBSERVING, S.PLANNING, S.EXECUTING, S.RECOVERING, S.COMPLETED, S.FAILED, S.WAITING_FOR_USER},
    S.RECOVERING: {S.OBSERVING, S.PLANNING, S.EXECUTING, S.APPLICATION_DETECTION, S.WAITING_FOR_USER, S.FAILED},
    S.WAITING_FOR_USER: _ACTIVE | {S.APPLICATION_DETECTION, S.APPLICATION_LAUNCH, S.UNDERSTANDING, S.FAILED, S.COMPLETED},
    S.COMPLETED: {S.IDLE, S.UNDERSTANDING, S.APPLICATION_DETECTION},
    S.FAILED: {S.IDLE, S.UNDERSTANDING, S.APPLICATION_DETECTION},
}


class InvalidTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, owner: str, bus: EventBus | None = None, task_id: str | None = None) -> None:
        self.owner = owner
        self.bus = bus
        self.task_id = task_id
        self.state = AgentState.IDLE
        self.history: list[tuple[float, str]] = [(time.time(), self.state.value)]

    def to(self, new: AgentState, detail: str = "") -> None:
        if new == self.state:
            return
        if new not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.owner}: {self.state.value} -> {new.value} is not allowed")
        old, self.state = self.state, new
        self.history.append((time.time(), new.value))
        if self.bus:
            self.bus.publish(
                "state_changed", task_id=self.task_id, agent=self.owner, old=old.value, state=new.value, detail=detail
            )

    def force(self, new: AgentState, detail: str = "") -> None:
        """Used only for hard stops (user pressed Stop); bypasses the transition table."""
        old, self.state = self.state, new
        self.history.append((time.time(), new.value))
        if self.bus:
            self.bus.publish(
                "state_changed", task_id=self.task_id, agent=self.owner, old=old.value, state=new.value, detail=detail
            )

    def reset(self) -> None:
        self.state = AgentState.IDLE
