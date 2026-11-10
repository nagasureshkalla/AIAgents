"""DuckDuckGo (and other engines aggregated by the `ddgs` package). No API key required."""

from __future__ import annotations

import asyncio
import logging

from app.research.errors import SearchError
from app.research.schemas import SearchResult
from app.research.search.base import SearchProvider

logger = logging.getLogger(__name__)


class DDGSProvider(SearchProvider):
    name = "duckduckgo"

    def __init__(self, timeout: float = 15, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        self.timeout = timeout
        self.region = region
        self.safesearch = safesearch

    def _search_sync(self, query: str, max_results: int) -> list[dict]:
        from ddgs import DDGS

        return DDGS(timeout=int(self.timeout)).text(
            query, region=self.region, safesearch=self.safesearch, max_results=max_results
        )

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            raw = await asyncio.to_thread(self._search_sync, query, max_results)
        except Exception as exc:  # noqa: BLE001 - ddgs raises its own exception types (rate limit, timeout)
            message = str(exc) or exc.__class__.__name__
            if "no results" in message.lower():
                return []
            raise SearchError(f"DuckDuckGo search failed for {query!r}: {message}") from exc
        results: list[SearchResult] = []
        for i, item in enumerate(raw or []):
            url = item.get("href") or item.get("url")
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or "",
                    snippet=item.get("body") or item.get("snippet") or "",
                    provider=self.name,
                    query=query,
                    rank=i + 1,
                )
            )
        return results
