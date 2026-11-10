"""Intent understanding, research plan and search-query generation (fast local model)."""

from __future__ import annotations

import datetime as dt
import re

from app.research.memory import ResearchMemory, ResearchPlan
from app.research.retrieval.ranker import tokenize
from app.research.pipeline import LLM
from app.utils.text import extract_json, truncate

PLAN_SYSTEM = (
    "You are the planning module of a web research assistant. Today's date is {today}. "
    "Read the user's question and plan how to research it on the web. Reply with JSON only, no prose:\n"
    '{{"intent": "<one sentence: what the user needs>", '
    '"answer_type": "fact|list|comparison|explanation|howto|news", '
    '"time_sensitive": true|false, '
    '"sub_questions": ["<2-4 specific questions a complete answer must cover>"], '
    '"queries": ["<{n} short, diverse web search queries>"]}}\n'
    "Queries are keyword-style like a person would type into a search engine. For time-sensitive questions "
    "include the current year ({year}). Do not answer the question."
)

GAP_SYSTEM = (
    "You check whether collected research evidence is enough to answer a question accurately. Today's date "
    "is {today}. Reply with JSON only:\n"
    '{{"enough": true|false, "missing": ["<information still needed>"], "queries": ["<up to {n} new web search '
    'queries that would find the missing information>"]}}\n'
    "Say enough=true only if the evidence directly answers every part of the question from at least two "
    "independent sources (or one authoritative primary source for a simple fact). New queries must differ "
    "from the ones already run."
)


_RECENT = re.compile(r"\b(latest|newest|current|recent|today|now|this (?:week|month|year)|20\d\d)\b", re.I)


def _today() -> dt.date:
    return dt.date.today()


def _clean_queries(raw: object, limit: int, exclude: list[str]) -> list[str]:
    seen = {q.lower().strip() for q in exclude}
    out: list[str] = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, str):
            continue
        q = re.sub(r"\s+", " ", q).strip().strip('"')
        if 2 < len(q) <= 200 and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) >= limit:
            break
    return out


async def make_plan(llm: LLM, question: str, n_queries: int) -> ResearchPlan:
    today = _today()
    system = PLAN_SYSTEM.format(today=today.isoformat(), year=today.year, n=n_queries)
    raw = await llm("fast", system, f"Question: {truncate(question, 2000)}", max_tokens=320, temperature=0.0)
    data = extract_json(raw)
    data = data if isinstance(data, dict) else {}
    time_sensitive = bool(data.get("time_sensitive", False)) or bool(_RECENT.search(question))
    queries = _clean_queries(data.get("queries"), n_queries, [])
    fallback = re.sub(r"\s+", " ", question).strip()[:200]
    if fallback.lower() not in {q.lower() for q in queries}:
        queries = [fallback, *queries][:n_queries]  # the user's own wording is always a good first query
    if len(queries) < min(2, n_queries):  # small models often return no usable queries: derive some
        keywords = " ".join(tokenize(question)) or fallback
        extra = [f"{keywords} {today.year}" if time_sensitive else keywords, keywords]
        queries += _clean_queries(extra, n_queries - len(queries), queries)
    subs = [s.strip() for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()][:4]
    return ResearchPlan(
        question=question,
        intent=str(data.get("intent") or "").strip()[:300],
        answer_type=str(data.get("answer_type") or "explanation"),
        time_sensitive=time_sensitive,
        sub_questions=subs or [question],
        queries=queries,
    )


async def assess_gaps(llm: LLM, memory: ResearchMemory, n_queries: int) -> tuple[bool, list[str], list[str]]:
    """Ask the fast model whether the evidence suffices. Returns (enough, missing, new_queries)."""
    plan = memory.plan
    assert plan is not None
    lines = []
    for ev in memory.evidence[:40]:
        lines.append(f"- [{ev.source_id}] {truncate(ev.claim, 220)}")
    evidence = "\n".join(lines) or "(no evidence yet)"
    user = (
        f"Question: {plan.question}\nMust cover:\n" + "\n".join(f"- {s}" for s in plan.sub_questions)
        + f"\n\nQueries already run:\n" + "\n".join(f"- {q}" for q in memory.queries_run)
        + f"\n\nEvidence so far ({len(memory.evidence_sources)} sources):\n{evidence}"
    )
    today = _today()
    raw = await llm("fast", GAP_SYSTEM.format(today=today.isoformat(), n=n_queries), truncate(user, 7000), max_tokens=240)
    data = extract_json(raw)
    data = data if isinstance(data, dict) else {}
    enough = bool(data.get("enough", False))
    missing = [m for m in data.get("missing", []) if isinstance(m, str)][:4]
    queries = _clean_queries(data.get("queries"), n_queries, memory.queries_run)
    if not enough and not queries:  # model gave no new direction: search the gaps / sub-questions directly
        candidates = missing or [s for s in plan.sub_questions if s != plan.question]
        queries = _clean_queries(candidates, n_queries, memory.queries_run)
    return enough, missing, queries
