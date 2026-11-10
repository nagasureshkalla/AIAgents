"""Lexical relevance ranking (BM25 + query-term coverage). Runs on the CPU and needs no extra model."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.research.retrieval.chunker import chunk_document
from app.research.schemas import Chunk, Document

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be but by can did do does for from has have how i in is it its of on or than that the "
    "their there these this to was were what when where which who why will with you your about into over most "
    "latest current new best top vs versus please tell me explain find search web online".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


def bm25_scores(query: str, texts: Sequence[str], k1: float = 1.4, b: float = 0.75) -> list[float]:
    q_terms = set(tokenize(query))
    docs = [tokenize(t) for t in texts]
    if not q_terms or not docs:
        return [0.0] * len(texts)
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(term for d in docs for term in set(d) if term in q_terms)
    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for term in q_terms:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avgdl))
        coverage = sum(1 for term in q_terms if tf[term]) / len(q_terms)
        scores.append(score * (0.5 + coverage))
    return scores


def rank_chunks(chunks: list[Chunk], query: str) -> list[Chunk]:
    scores = bm25_scores(query, [f"{c.heading} {c.text}" for c in chunks])
    ranked = [c.model_copy(update={"score": round(s, 4)}) for c, s in zip(chunks, scores, strict=True)]
    return sorted(ranked, key=lambda c: (-c.score, c.index))


def retrieve_relevant_chunks(document: Document, query: str, max_chunks: int = 10) -> list[Chunk]:
    """Chunk a document and return its most query-relevant chunks (in reading order)."""
    chunks = chunk_document(document)
    if not chunks:
        return []
    ranked = [c for c in rank_chunks(chunks, query) if c.score > 0][:max_chunks]
    if not ranked:  # nothing matched lexically: keep the opening of the page
        ranked = chunks[: min(2, max_chunks)]
    return sorted(ranked, key=lambda c: c.index)
