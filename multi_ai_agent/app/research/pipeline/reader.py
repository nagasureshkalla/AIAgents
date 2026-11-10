"""Source selection and reading: search results -> fetched documents -> relevant chunks."""

from __future__ import annotations

import asyncio
import logging
import re

from app.research.browser.browser import Browser
from app.research.memory import ResearchMemory
from app.research.pipeline import Notify
from app.research.retrieval.deduplicator import dedupe_chunks
from app.research.retrieval.ranker import bm25_scores, retrieve_relevant_chunks, tokenize
from app.research.schemas import Document, FetchStatus, SearchResult, domain_of

logger = logging.getLogger(__name__)

# Pages that rarely contain readable article text for a static fetcher.
LOW_VALUE_DOMAINS = (
    "youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com",
    "x.com", "twitter.com", "linkedin.com", "vimeo.com",
)
MAX_PER_DOMAIN = 2


def select_sources(results: list[SearchResult], memory: ResearchMemory, limit: int) -> list[SearchResult]:
    """Rank unseen results by snippet relevance and search rank, keeping domains diverse."""
    focus = memory.plan.focus if memory.plan else memory.question
    candidates = [
        r for r in results
        if not memory.was_attempted(r.url)
        and r.url.startswith(("http://", "https://"))
        and not any(r.domain == d or r.domain.endswith("." + d) for d in LOW_VALUE_DOMAINS)
    ]
    if not candidates:
        return []
    lexical = bm25_scores(focus, [f"{r.title} {r.snippet}" for r in candidates])
    top = max(lexical) or 1.0
    terms = set(tokenize(memory.question))

    def score(r: SearchResult, lex: float) -> float:
        # Snippet relevance dominates; the engine's rank breaks ties; a domain named after the subject
        # (python.org for a Python question) is usually the primary source.
        labels = set(re.split(r"[.-]", r.domain))
        official = 0.3 if terms & labels else 0.0
        return 0.65 * lex / top + 0.35 / (r.rank or 10) + official

    scored = sorted(zip(candidates, lexical, strict=True), key=lambda pair: -score(*pair))
    per_domain: dict[str, int] = {}
    for doc in memory.attempted.values():
        per_domain[doc.domain] = per_domain.get(doc.domain, 0) + 1
    chosen: list[SearchResult] = []
    for result, _ in scored:
        if per_domain.get(result.domain, 0) >= MAX_PER_DOMAIN:
            continue
        per_domain[result.domain] = per_domain.get(result.domain, 0) + 1
        chosen.append(result)
        if len(chosen) >= limit:
            break
    return chosen


async def read_sources(
    browser: Browser,
    selected: list[SearchResult],
    memory: ResearchMemory,
    *,
    concurrency: int,
    chunks_per_source: int,
    notify: Notify,
    offset: int = 0,
    total: int | None = None,
) -> list[int]:
    """Open pages concurrently, register readable ones as citable sources, keep their relevant chunks.

    Returns the citation ids of the sources read in this batch.
    """
    focus = memory.plan.focus if memory.plan else memory.question
    gate = asyncio.Semaphore(concurrency)
    total = total or offset + len(selected)
    counter = {"n": offset}

    async def one(result: SearchResult) -> Document:
        async with gate:
            counter["n"] += 1
            n = counter["n"]
            notify("status", f"Reading source {n}/{total}: {result.domain or domain_of(result.url)}",
                   stage="read", url=result.url, index=n, total=total)
            try:
                doc = await browser.open(result.url)
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the research
                doc = Document(
                    url=result.url, domain=result.domain, status=FetchStatus.ERROR, error=f"{exc.__class__.__name__}: {exc}"
                )
            if not doc.title:
                doc.title = result.title
            return doc

    docs = await asyncio.gather(*(one(r) for r in selected))
    new_ids: list[int] = []
    for doc in docs:
        memory.mark_attempted(doc)
        if not doc.ok:
            memory.errors.append(f"{doc.url}: {doc.error or doc.status}")
            notify("source", doc.title or doc.url, url=doc.url, status=str(doc.status), error=doc.error, domain=doc.domain)
            continue
        sid = memory.citations.register(doc)
        if sid in memory.chunks:  # same page reached through a different URL
            continue
        chunks = retrieve_relevant_chunks(doc, focus, max_chunks=chunks_per_source)
        memory.chunks[sid] = chunks
        new_ids.append(sid)
        notify(
            "source",
            doc.title,
            url=memory.citations.get(sid).url,
            source_id=sid,
            status="ok",
            domain=doc.domain,
            published_date=doc.published_date,
            renderer=doc.renderer,
        )

    # Remove passages duplicated across sources (syndicated copies), keeping the earlier source's copy.
    all_chunks = [(sid, c) for sid in sorted(memory.chunks) for c in memory.chunks[sid]]
    keep = {id(c) for c in dedupe_chunks([c for _, c in all_chunks])}
    for sid in new_ids:
        memory.chunks[sid] = [c for c in memory.chunks[sid] if id(c) in keep]
    return new_ids
