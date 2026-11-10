"""Search provider contract. A search provider retrieves public web results; it is never an AI provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.research.schemas import SearchResult


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Return up to `max_results` results for `query`. Raise SearchError on provider failure."""
        raise NotImplementedError

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
