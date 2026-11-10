"""Self-hosted SearXNG metasearch (JSON API). Set research.searxng_url, e.g. http://127.0.0.1:8888."""

from __future__ import annotations

import httpx

from app.research.errors import SearchError
from app.research.schemas import SearchResult
from app.research.search.base import SearchProvider


class SearXNGProvider(SearchProvider):
    name = "searxng"

    def __init__(self, base_url: str, timeout: float = 15, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            resp = await self._client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"SearXNG search failed for {query!r}: {exc}") from exc
        return [
            SearchResult(
                url=item["url"],
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                provider=self.name,
                query=query,
                rank=i + 1,
            )
            for i, item in enumerate(data.get("results", [])[:max_results])
            if item.get("url")
        ]

    async def aclose(self) -> None:
        await self._client.aclose()
