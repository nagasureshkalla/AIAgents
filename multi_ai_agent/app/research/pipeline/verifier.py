"""Cross-source verification: SUPPORTED / CONFLICTING / INSUFFICIENT per key claim (reasoning model)."""

from __future__ import annotations

import logging

from app.research.memory import ResearchMemory
from app.research.pipeline import LLM
from app.research.schemas import ClaimStatus, VerifiedClaim
from app.utils.text import extract_json, truncate

logger = logging.getLogger(__name__)

VERIFY_SYSTEM = (
    "You are a fact-checking module. You receive a research question and evidence statements, each tagged "
    "with the number of the web source it came from, e.g. [2]. Identify the key claims needed to answer the "
    "question (at most {n}) and compare the sources for each:\n"
    "- SUPPORTED: at least one source states it and no source contradicts it.\n"
    "- CONFLICTING: sources disagree (different numbers, dates or facts). Describe the disagreement in note.\n"
    "- INSUFFICIENT: the evidence does not establish it.\n"
    "Use only the evidence given; never resolve a conflict by guessing. Values that differ only because the "
    "sources are from different dates (e.g. an older 'latest version') are not a conflict: the most recently "
    "published official source supersedes older ones; mention the dates in note. Reply with JSON only:\n"
    '{{"claims": [{{"claim": "<claim>", "status": "SUPPORTED|CONFLICTING|INSUFFICIENT", '
    '"sources": [<source numbers>], "note": "<short note, required for CONFLICTING>"}}]}}'
)


def format_evidence(memory: ResearchMemory, limit_chars: int) -> str:
    lines: list[str] = []
    for sid in memory.evidence_sources:
        src = memory.citations.get(sid)
        date = f", {src.published_date[:10]}" if src.published_date else ""
        lines.append(f"Source [{sid}] {src.title} ({src.domain}{date}):")
        for ev in memory.evidence_for(sid):
            quote = f' (quote: "{ev.quote}")' if ev.quote else ""
            lines.append(f"  - [{sid}] {ev.claim}{quote}")
    return truncate("\n".join(lines), limit_chars)


async def verify_claims(llm: LLM, memory: ResearchMemory, evidence_chars: int, max_claims: int = 8) -> list[VerifiedClaim]:
    if not memory.evidence:
        return []
    plan = memory.plan
    question = plan.question if plan else memory.question
    user = f"Question: {question}\n\nEvidence:\n{format_evidence(memory, evidence_chars)}"
    raw = await llm("reasoning", VERIFY_SYSTEM.format(n=max_claims), user, max_tokens=900)
    data = extract_json(raw)
    items = data.get("claims") if isinstance(data, dict) else data if isinstance(data, list) else []
    valid_ids = set(memory.evidence_sources)
    out: list[VerifiedClaim] = []
    for item in items or []:
        if not isinstance(item, dict) or not str(item.get("claim") or "").strip():
            continue
        try:
            status = ClaimStatus(str(item.get("status", "")).upper().strip())
        except ValueError:
            status = ClaimStatus.INSUFFICIENT
        sources = []
        for s in item.get("sources") or []:
            try:
                sid = int(str(s).strip("[] "))
            except ValueError:
                continue
            if sid in valid_ids and sid not in sources:
                sources.append(sid)
        if status != ClaimStatus.INSUFFICIENT and not sources:
            status = ClaimStatus.INSUFFICIENT  # a verdict without real sources cannot count as evidence
        if status == ClaimStatus.CONFLICTING and len(sources) < 2:
            status = ClaimStatus.SUPPORTED if sources else ClaimStatus.INSUFFICIENT
        out.append(VerifiedClaim(claim=truncate(str(item["claim"]).strip(), 400), status=status, sources=sources, note=truncate(str(item.get("note") or ""), 300)))
        if len(out) >= max_claims:
            break
    if not out:  # model output unusable: every extracted claim counts as single-source support
        out = [VerifiedClaim(claim=e.claim, status=ClaimStatus.SUPPORTED, sources=[e.source_id]) for e in memory.evidence[:max_claims]]
    return out
