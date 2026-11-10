"""Evidence extraction: relevant chunks of one source -> claims with exact supporting quotes."""

from __future__ import annotations

import asyncio
import logging
import re

from app.research.memory import ResearchMemory
from app.research.pipeline import LLM
from app.research.schemas import Evidence
from app.utils.text import extract_json, truncate

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = (
    "You extract evidence from one web source for a research question. Use ONLY the source text. "
    "List up to {n} factual claims from the text that help answer the question (numbers, dates, names, "
    "specifications, definitions, statements). For each claim copy a short supporting quote verbatim "
    "from the source (max 30 words). Never add outside knowledge. Reply with JSON only:\n"
    '{{"claims": [{{"claim": "<self-contained factual statement>", "quote": "<exact words from the source>"}}]}}\n'
    'If the source has nothing relevant, reply {{"claims": []}}.'
)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def quote_support(quote: str, source_text: str) -> float:
    """1.0 if the quote appears verbatim (ignoring punctuation/case), else the fraction of its words present."""
    q, src = _norm(quote), _norm(source_text)
    if not q:
        return 0.0
    if q in src:
        return 1.0
    words = q.split()
    src_words = set(src.split())
    return sum(1 for w in words if w in src_words) / len(words)


async def extract_from_source(llm: LLM, memory: ResearchMemory, sid: int, max_claims: int = 6) -> list[Evidence]:
    chunks = memory.chunks.get(sid) or []
    if not chunks:
        return []
    source = memory.citations.get(sid)
    text = "\n\n".join(c.text for c in chunks)
    plan = memory.plan
    question = plan.question if plan else memory.question
    focus = "\n".join(f"- {s}" for s in (plan.sub_questions if plan else []))
    user = (
        f"Research question: {question}\n" + (f"Aspects to cover:\n{focus}\n" if focus else "")
        + f"\nSource: {source.title} ({source.domain})"
        + (f", published {source.published_date[:10]}" if source.published_date else "")
        + f"\n\nSource text:\n{truncate(text, 6000)}"
    )
    raw = await llm("reasoning", EXTRACT_SYSTEM.format(n=max_claims), user, max_tokens=700)
    data = extract_json(raw)
    items = data.get("claims") if isinstance(data, dict) else data if isinstance(data, list) else []
    out: list[Evidence] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if len(claim) < 8:
            continue
        support = quote_support(quote, text) if quote else quote_support(claim, text) * 0.8
        if support < 0.6:  # not traceable to the page: drop rather than risk a fabricated citation
            logger.debug("dropping untraceable claim from source %s: %s", sid, claim)
            continue
        out.append(Evidence(claim=truncate(claim, 400), source_id=sid, quote=truncate(quote, 300), confidence=round(min(1.0, 0.5 + support / 2), 2)))
        if len(out) >= max_claims:
            break
    return out


async def extract_evidence(llm: LLM, memory: ResearchMemory, source_ids: list[int]) -> int:
    """Extract evidence from newly read sources (inference is serialized by the GPU scheduler)."""
    results = await asyncio.gather(*(extract_from_source(llm, memory, sid) for sid in source_ids), return_exceptions=True)
    added = 0
    for sid, res in zip(source_ids, results, strict=True):
        if isinstance(res, BaseException):
            memory.errors.append(f"evidence extraction failed for source {sid}: {res}")
            continue
        memory.evidence.extend(res)
        added += len(res)
    return added
