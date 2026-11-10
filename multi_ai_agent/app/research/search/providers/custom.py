"""Template for adding your own search backend (e.g. Bing, Brave, an internal index).

Implement `search` and select it by registering it in SearchManager.from_settings. Search API keys are
fine here: they are only ever used to retrieve public search results, never for LLM inference.
"""

from __future__ import annotations

from app.research.schemas import SearchResult
from app.research.search.base import SearchProvider


class StaticSearchProvider(SearchProvider):
    """Returns fixed results; useful for tests, demos and offline evaluation."""

    name = "static"

    def __init__(self, results: dict[str, list[SearchResult]] | list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        self.queries.append(query)
        if isinstance(self.results, dict):
            found = self.results.get(query, self.results.get("*", []))
        else:
            found = self.results
        return [r.model_copy(update={"query": query}) for r in found[:max_results]]
