"""Research pipeline stages. Each stage calls the local LLM through an `LLM` callable."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

Role = Literal["fast", "reasoning"]


class LLM(Protocol):
    def __call__(
        self, role: Role, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0
    ) -> Awaitable[str]: ...


Notify = Callable[..., None]
