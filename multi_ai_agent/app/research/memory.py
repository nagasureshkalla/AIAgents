"""Working memory of one research run."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.research.citations import CitationManager
from app.research.retrieval.deduplicator import canonical_url
from app.research.schemas import Chunk, Document, Evidence, SearchResult, VerifiedClaim


@dataclass
class ResearchPlan:
    question: str
    intent: str = ""
    answer_type: str = "explanation"
    time_sensitive: bool = False
    sub_questions: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    @property
    def focus(self) -> str:
        """Text used for lexical relevance ranking."""
        return " ".join([self.question, *self.sub_questions])


@dataclass
class ResearchMemory:
    question: str
    plan: ResearchPlan | None = None
    citations: CitationManager = field(default_factory=CitationManager)
    queries_run: list[str] = field(default_factory=list)
    results_seen: dict[str, SearchResult] = field(default_factory=dict)
    attempted: dict[str, Document] = field(default_factory=dict)  # canonical url -> fetch outcome
    chunks: dict[int, list[Chunk]] = field(default_factory=dict)  # source id -> relevant chunks
    evidence: list[Evidence] = field(default_factory=list)
    verified: list[VerifiedClaim] = field(default_factory=list)
    iterations: int = 0
    gaps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def was_attempted(self, url: str) -> bool:
        return canonical_url(url) in self.attempted

    def mark_attempted(self, doc: Document) -> None:
        self.attempted[canonical_url(doc.url)] = doc

    @property
    def readable_docs(self) -> list[Document]:
        return [d for d in self.attempted.values() if d.ok]

    @property
    def failed_docs(self) -> list[Document]:
        return [d for d in self.attempted.values() if not d.ok]

    def evidence_for(self, sid: int) -> list[Evidence]:
        return [e for e in self.evidence if e.source_id == sid]

    @property
    def evidence_sources(self) -> list[int]:
        return sorted({e.source_id for e in self.evidence})

    @property
    def evidence_domains(self) -> set[str]:
        return {self.citations.get(s).domain for s in self.evidence_sources}
