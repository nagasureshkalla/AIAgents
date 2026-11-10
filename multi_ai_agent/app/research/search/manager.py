"""Runs queries against the configured provider: concurrency, retry, URL de-duplication, offline gate."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import ResearchConfig
from app.research.errors import SearchError, WebDisabledError
from app.research.retrieval.deduplicator import canonical_url
from app.research.schemas import SearchResult
from app.research.search.base import SearchProvider

logger = logging.getLogger(__name__)


class SearchManager:
    def __init__(self, provider: SearchProvider, cfg: ResearchConfig, retries: int = 1) -> None:
        self.provider = provider
        self.cfg = cfg
        self.retries = retries

    @classmethod
    def from_settings(cls, cfg: ResearchConfig) -> "SearchManager":
        if cfg.search_provider == "searxng":
            if not cfg.searxng_url:
                raise SearchError("search_provider is 'searxng' but research.searxng_url is not set")
            from app.research.search.providers.searxng import SearXNGProvider

            return cls(SearXNGProvider(cfg.searxng_url, timeout=cfg.request_timeout), cfg)
        from app.research.search.providers.ddgs_provider import DDGSProvider

        return cls(DDGSProvider(timeout=cfg.request_timeout), cfg)

    async def _one(self, query: str, max_results: int) -> list[SearchResult]:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return await self.provider.search(query, max_results)
            except SearchError as exc:
                last = exc
                logger.info("search attempt %d for %r failed: %s", attempt + 1, query, exc.message)
                await asyncio.sleep(0.5 * (attempt + 1))
        assert last is not None
        raise last

    async def search_many(self, queries: list[str], max_results: int | None = None) -> tuple[list[SearchResult], list[str]]:
        """Search all queries concurrently. Returns (unique results in rank order, error messages).

        Raises SearchError only when every query failed.
        """
        if not self.cfg.web_enabled:
            raise WebDisabledError("Web access is disabled (LOCAL AI + OFFLINE mode)")
        limit = max_results or self.cfg.max_search_results
        outcomes = await asyncio.gather(*(self._one(q, limit) for q in queries), return_exceptions=True)
        errors = [str(o) for o in outcomes if isinstance(o, BaseException)]
        if queries and len(errors) == len(queries):
            raise SearchError("All web searches failed: " + "; ".join(errors[:3]))

        # Interleave results across queries so each query's best hits come first.
        lists = [o for o in outcomes if isinstance(o, list)]
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for rank in range(max((len(lst) for lst in lists), default=0)):
            for lst in lists:
                if rank < len(lst):
                    key = canonical_url(lst[rank].url)
                    if key not in seen:
                        seen.add(key)
                        merged.append(lst[rank])
        return merged, errors

    async def aclose(self) -> None:
        await self.provider.aclose()
