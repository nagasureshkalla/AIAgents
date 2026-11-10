"""Final answer: the reasoning model writes Answer + Key Findings from verified evidence; disagreements,
confidence and the Sources list are assembled in code so citations always point at visited pages."""

from __future__ import annotations

import re

from app.research.memory import ResearchMemory
from app.research.pipeline import LLM
from app.research.pipeline.verifier import format_evidence
from app.research.schemas import ClaimStatus, SourceRef

SYNTH_SYSTEM = (
    "You write the final answer of a web research assistant. Today's date is {today}. Use ONLY the evidence "
    "provided; each statement is tagged with its source number like [2]. Write in Markdown with exactly these "
    "two sections:\n"
    "## Answer\n<direct answer to the question>\n\n## Key Findings\n- <finding> [n]\n\n"
    "Rules:\n"
    "- Every factual sentence and every finding ends with one or more citations such as [1] or [2][3], using "
    "only the source numbers given.\n"
    "- If sources disagree, say so explicitly and cite each side; never invent a resolution.\n"
    "- For questions about the latest or current state, rely on the most recently published official sources, "
    "mention their dates, and note when older sources report outdated values.\n"
    "- If the evidence does not answer part of the question, say what is unknown.\n"
    "- {length}\n"
    "- Do not write URLs, a sources list, a confidence section, or your reasoning process."
)

_SIMPLE = "This is a simple question: keep the Answer to 1-3 sentences and give at most 3 findings."
_COMPLEX = "This is a research question: give a thorough but tight Answer (one or two paragraphs) and 3-7 findings."
_CUT = re.compile(r"(?im)^\s*#{1,6}\s*(sources|references|citations|confidence|bibliography)\b.*\Z", re.S)


def assess_confidence(memory: ResearchMemory, cited: list[int]) -> tuple[str, str]:
    verified = memory.verified
    supported = [c for c in verified if c.status == ClaimStatus.SUPPORTED]
    corroborated = [c for c in supported if len(c.sources) >= 2]
    conflicts = [c for c in verified if c.status == ClaimStatus.CONFLICTING]
    domains = {memory.citations.get(s).domain for s in cited if memory.citations.has(s)}
    if not cited or not supported:
        return "Low", "the sources found do not directly establish the answer"
    share = len(supported) / max(1, len(verified))
    if corroborated and len(domains) >= 3 and not conflicts and share >= 0.6:
        return "High", f"key claims are corroborated across {len(domains)} independent sources with no conflicts"
    if len(domains) >= 2 or corroborated:
        why = f"{len(supported)} supported claim(s) from {len(domains)} source domain(s)"
        if conflicts:
            why += f"; {len(conflicts)} point(s) where sources disagree"
        return "Medium", why
    return "Low", "the answer rests on a single source"


def _source_line(src: SourceRef) -> str:
    meta = ", ".join(x for x in (src.domain, src.published_date[:10] if src.published_date else None) if x)
    title = src.title.replace("[", "(").replace("]", ")")
    return f"[{src.id}] [{title}]({src.url}) — {meta}"


def is_simple(memory: ResearchMemory) -> bool:
    plan = memory.plan
    return bool(plan and plan.answer_type in ("fact", "news") and len(plan.sub_questions) <= 2)


async def synthesize(llm: LLM, memory: ResearchMemory, evidence_chars: int, today: str) -> dict:
    plan = memory.plan
    question = plan.question if plan else memory.question
    verdicts = "\n".join(
        f"- {c.status.value}: {c.claim} " + "".join(f"[{s}]" for s in c.sources) + (f" — {c.note}" if c.note else "")
        for c in memory.verified
    )
    user = (
        f"Question: {question}\n\nEvidence:\n{format_evidence(memory, evidence_chars)}"
        + (f"\n\nVerification of key claims:\n{verdicts}" if verdicts else "")
    )
    system = SYNTH_SYSTEM.format(today=today, length=_SIMPLE if is_simple(memory) else _COMPLEX)
    raw = await llm("reasoning", system, user, max_tokens=1100, temperature=0.1)
    body = _CUT.sub("", raw).strip()
    if not re.search(r"(?im)^#{1,6}\s*answer", body):
        body = "## Answer\n\n" + body
    body, cited, invalid = memory.citations.sanitize(body)
    return assemble(memory, body, cited, invalid)


def assemble(memory: ResearchMemory, body: str, cited: list[int], invalid: list[int]) -> dict:
    level, reason = assess_confidence(memory, cited)
    parts = [body]
    conflicts = [c for c in memory.verified if c.status == ClaimStatus.CONFLICTING]
    if conflicts:
        lines = ["## Disagreements Between Sources", ""]
        for c in conflicts:
            refs = "".join(f"[{s}]" for s in c.sources)
            lines.append(f"- {c.claim} {refs}" + (f" — {c.note}" if c.note else ""))
        parts.append("\n".join(lines))
    parts.append(f"## Confidence\n\n**{level}** — {reason}.")
    listed = sorted(cited) or memory.evidence_sources
    sources = [memory.citations.get(s) for s in listed]
    if sources:
        parts.append("## Sources\n\n" + "\n".join(_source_line(s) for s in sources))
    return {"text": "\n\n".join(parts).strip(), "confidence": level, "confidence_reason": reason, "cited": cited, "invalid_citations": invalid}
