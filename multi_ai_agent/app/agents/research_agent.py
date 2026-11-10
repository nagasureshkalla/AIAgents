"""Research agent: autonomous web research with local-GPU inference and verified citations.

PLAN -> SEARCH -> READ -> ANALYZE -> (enough? no -> search again) -> VERIFY -> ANSWER

The local model drives every decision. The internet is only used to run searches and open public pages;
user questions are never sent to any AI service.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.config import ResearchConfig
from app.core.events import EventType
from app.research.browser.browser import Browser
from app.research.config import DepthProfile, depth_profile
from app.research.errors import GPURequiredError, SearchError
from app.research.memory import ResearchMemory
from app.research.pipeline import LLM, Role
from app.research.pipeline.extraction import extract_evidence
from app.research.pipeline.planner import assess_gaps, make_plan
from app.research.pipeline.reader import read_sources, select_sources
from app.research.pipeline.synthesizer import synthesize
from app.research.pipeline.verifier import verify_claims
from app.research.search.manager import SearchManager
from app.schemas.agent import AgentTask

logger = logging.getLogger(__name__)

NO_GPU_MESSAGE = "NVIDIA CUDA GPU was not detected. Local AI inference cannot start."
OFFLINE_NOTICE = (
    "> **Web research is disabled (LOCAL AI + OFFLINE mode).** This answer comes only from the local model's "
    "built-in knowledge: it has no citations and may be out of date."
)
OFFLINE_SYSTEM = (
    "You are a helpful assistant running fully offline. Answer from your own knowledge, concisely and "
    "accurately. If the question depends on recent events or you are unsure, say so plainly. Do not invent "
    "sources, URLs or citations."
)


class ResearchAgent(BaseAgent):
    name = "research"
    description = (
        "Autonomous web research: plans searches, reads public web pages, cross-checks sources and answers "
        "with verified citations. All AI inference runs on the local NVIDIA GPU."
    )
    capabilities = ["web_research", "fact_checking", "citation"]
    default_model = "research"

    # Swappable for tests or custom backends.
    search_factory: Callable[[ResearchConfig], SearchManager] = staticmethod(SearchManager.from_settings)
    browser_factory: Callable[[ResearchConfig], Browser] = staticmethod(lambda cfg: Browser(cfg))

    @property
    def rcfg(self) -> ResearchConfig:
        return self.ctx.settings.research

    @property
    def model_key(self) -> str | None:
        return self.config.model or self.rcfg.reasoning_model or self.default_model

    def required_models(self) -> list[str]:
        on_gpu = self.ctx.models.device_info.type == "cuda"
        if not on_gpu and self.rcfg.require_cuda and not self.rcfg.allow_cpu:
            return []  # the agent refuses to run without CUDA: don't load models for nothing
        keys = [self.model_key, self.rcfg.fast_model]
        return [k for k in dict.fromkeys(keys) if k]

    # ------------------------------------------------------------------ local LLM access
    def _check_gpu(self, task: AgentTask) -> None:
        on_gpu = self.ctx.models.device_info.type == "cuda" and task.device_override != "cpu"
        if not on_gpu and self.rcfg.require_cuda and not self.rcfg.allow_cpu:
            raise GPURequiredError(NO_GPU_MESSAGE)

    def _llm(self, task: AgentTask) -> LLM:
        keys: dict[Role, str] = {
            "fast": self.rcfg.fast_model,
            "reasoning": self.model_key or self.rcfg.reasoning_model,
        }

        async def call(role: Role, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
            out = await self.generate(
                task, user, system=system, model_key=keys[role], max_tokens=max_tokens, temperature=temperature
            )
            return out.text

        return call

    def _notifier(self, task: AgentTask) -> Callable[..., None]:
        def notify(kind: str, message: str, **data: Any) -> None:
            if kind == "source":
                self.emit(task, EventType.RESEARCH_SOURCE, title=message, **data)
            else:
                self.progress(task, message, **data)

        return notify

    # ------------------------------------------------------------------ main loop
    async def run(self, task: AgentTask) -> dict[str, Any]:
        self._check_gpu(task)
        depth = depth_profile(task.params.get("depth"), self.rcfg)
        llm = self._llm(task)
        notify = self._notifier(task)
        question = task.prompt.strip()

        if not self.rcfg.web_enabled:
            return await self._offline_answer(llm, question, notify, depth)

        memory = ResearchMemory(question=question)
        notify("status", "Understanding the question and planning research", stage="plan")
        memory.plan = await make_plan(llm, question, depth.queries_per_iteration)
        notify(
            "status",
            f"Research plan: {len(memory.plan.sub_questions)} aspect(s), {len(memory.plan.queries)} search queries",
            stage="plan",
            queries=memory.plan.queries,
            sub_questions=memory.plan.sub_questions,
        )

        search = self.search_factory(self.rcfg)
        browser = self.browser_factory(self.rcfg)
        try:
            await self._research_loop(llm, memory, depth, search, browser, notify)
        finally:
            await search.aclose()
            await browser.aclose()

        if not memory.evidence:
            notify("status", "No usable evidence found", stage="answer")
            return self._no_evidence(memory, depth)

        n_sources = len(memory.evidence_sources)
        notify("status", f"Comparing {n_sources} source{'s' if n_sources != 1 else ''}", stage="compare", sources=n_sources)
        notify("status", "Verifying claims across sources", stage="verify")
        memory.verified = await verify_claims(llm, memory, depth.evidence_chars)
        notify("status", "Generating answer", stage="answer")
        final = await synthesize(llm, memory, depth.evidence_chars, dt.date.today().isoformat())
        return self._payload(memory, depth, final)

    async def _research_loop(
        self,
        llm: LLM,
        memory: ResearchMemory,
        depth: DepthProfile,
        search: SearchManager,
        browser: Browser,
        notify: Callable[..., None],
    ) -> None:
        assert memory.plan is not None
        queries = list(memory.plan.queries)
        for iteration in range(1, depth.iterations + 1):
            memory.iterations = iteration
            notify("status", f"Searching the web ({len(queries)} quer{'ies' if len(queries) != 1 else 'y'})",
                   stage="search", iteration=iteration, queries=queries)
            try:
                results, errors = await search.search_many(queries)
                memory.errors.extend(errors)
            except SearchError as exc:
                memory.errors.append(exc.message)
                notify("status", f"Search failed: {exc.message}", stage="search", error=True)
                results = []
            memory.queries_run.extend(queries)

            budget = depth.sources - len(memory.attempted)
            selected = select_sources(results, memory, min(depth.sources_per_iteration, budget))
            if selected:
                already = len(memory.attempted)
                new_ids = await read_sources(
                    browser,
                    selected,
                    memory,
                    concurrency=self.rcfg.max_concurrent_fetches,
                    chunks_per_source=depth.chunks_per_source,
                    notify=notify,
                    offset=already,
                    total=already + len(selected),
                )
                if new_ids:
                    notify("status", f"Extracting evidence from {len(new_ids)} source(s)", stage="extract")
                    await extract_evidence(llm, memory, new_ids)
            elif not results:
                notify("status", "No search results", stage="search")

            if iteration >= depth.iterations or len(memory.attempted) >= depth.sources:
                return
            notify("status", "Checking whether more information is needed", stage="analyze")
            enough, missing, queries = await assess_gaps(llm, memory, depth.queries_per_iteration)
            memory.gaps = missing
            if enough and memory.evidence:
                notify("status", "Enough reliable information collected", stage="analyze")
                return
            if not queries:
                return
            notify("status", "Missing: " + "; ".join(missing[:3]) if missing else "Searching again for more evidence",
                   stage="analyze", missing=missing)

    # ------------------------------------------------------------------ outputs
    def _payload(self, memory: ResearchMemory, depth: DepthProfile, final: dict[str, Any]) -> dict[str, Any]:
        plan = memory.plan
        return {
            "mode": "web",
            "text": final["text"],
            "confidence": final["confidence"],
            "confidence_reason": final["confidence_reason"],
            "cited": final["cited"],
            "invalid_citations": final["invalid_citations"],
            "sources": [s.model_dump() for s in memory.citations.sources],
            "claims": [c.model_dump() for c in memory.verified],
            "evidence": [e.model_dump() for e in memory.evidence],
            "failed_sources": [
                {"url": d.url, "status": str(d.status), "error": d.error} for d in memory.failed_docs
            ],
            "queries": memory.queries_run,
            "iterations": memory.iterations,
            "depth": depth.name,
            "gaps": memory.gaps,
            "plan": {"intent": plan.intent, "answer_type": plan.answer_type, "sub_questions": plan.sub_questions}
            if plan
            else None,
        }

    def _no_evidence(self, memory: ResearchMemory, depth: DepthProfile) -> dict[str, Any]:
        lines = [
            "## Answer",
            "",
            "I could not find enough reliable information on the web to answer this question, so I am not "
            "giving an answer rather than guessing.",
            "",
            "## Key Findings",
            "",
            f"- Searched for: {', '.join(repr(q) for q in memory.queries_run) or 'nothing (search unavailable)'}",
        ]
        if memory.readable_docs:
            lines.append(f"- {len(memory.readable_docs)} page(s) were read but none contained relevant evidence.")
        problems = memory.errors[:5]
        if problems:
            lines.append("- Problems: " + "; ".join(problems))
        lines += ["", "## Confidence", "", "**Low** — no evidence was found."]
        final = {"text": "\n".join(lines), "confidence": "Low", "confidence_reason": "no evidence was found", "cited": [], "invalid_citations": []}
        return self._payload(memory, depth, final)

    async def _offline_answer(self, llm: LLM, question: str, notify: Callable[..., None], depth: DepthProfile) -> dict[str, Any]:
        from app.research.citations import CitationManager

        notify("status", "Web disabled: answering from the local model only", stage="answer", offline=True)
        raw = await llm("reasoning", OFFLINE_SYSTEM, question, max_tokens=900, temperature=0.2)
        body, _, _ = CitationManager().sanitize(raw)  # no visited sources -> strips any [n] and URLs
        text = f"{OFFLINE_NOTICE}\n\n## Answer\n\n{body}\n\n## Confidence\n\n**Low** — not verified against any source."
        return {
            "mode": "offline",
            "text": text,
            "confidence": "Low",
            "confidence_reason": "web research disabled",
            "cited": [],
            "invalid_citations": [],
            "sources": [],
            "claims": [],
            "evidence": [],
            "failed_sources": [],
            "queries": [],
            "iterations": 0,
            "depth": depth.name,
            "gaps": [],
            "plan": None,
        }
