"""Manager agent: understands the request and decides which specialist agents are needed."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import AgentNotFoundError, InvalidRequestError
from app.schemas.agent import AgentTask
from app.utils.text import extract_json

logger = logging.getLogger(__name__)

# Ordered: first match wins.
IMAGE_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("screenshot_debug", re.compile(r"\b(screenshot|wrong|error|bug|broken|issue|crash|warning|ui|layout)\b", re.I)),
    (
        "image_analysis",
        re.compile(r"\b(analy[sz]e|describe|explain|caption|what(?:'s| is) happening|scene|overview)\b", re.I),
    ),
    ("visual_qa", re.compile(r"\?|\b(what|which|how many|where|who|is there|are there|does|do you see|count)\b", re.I)),
]
TEXT_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("summarize", re.compile(r"\b(summari[sz]e|summary|tl;?dr|condense|key takeaways)\b", re.I)),
    ("extract", re.compile(r"\b(extract|pull out|list all|find all|parse)\b", re.I)),
    ("classify", re.compile(r"\b(classify|categori[sz]e|sentiment|label|which category)\b", re.I)),
    ("format", re.compile(r"\b(format|convert (?:it |this )?to|rewrite as|as json|as a table|markdown table)\b", re.I)),
    (
        "explain",
        re.compile(r"\b(explain|why|analy[sz]e|reason|compare|evaluate|pros and cons|how does|trade-?offs?)\b", re.I),
    ),
]
TEXT_INTENTS = ["summarize", "extract", "classify", "format", "explain", "generate"]

INTENT_CAPABILITIES: dict[str, list[str]] = {
    "image_analysis": ["image_analysis", "request_interpretation", "reasoning", "validation"],
    "visual_qa": ["visual_question_answering", "image_description", "reasoning", "validation"],
    "screenshot_debug": ["screenshot_analysis", "image_description", "reasoning", "validation"],
    "summarize": ["summarization", "reasoning", "validation"],
    "extract": ["extraction", "validation"],
    "classify": ["classification", "reasoning", "validation"],
    "format": ["formatting", "validation"],
    "explain": ["reasoning", "text_generation", "validation"],
    "generate": ["text_generation", "validation"],
}


class ManagerAgent(BaseAgent):
    name = "manager"
    role = "control"
    description = "Understands the user's request and decides which specialist agents are required."
    capabilities = ["request_analysis", "agent_selection", "routing"]
    default_model = "fast_text"

    @staticmethod
    def classify_intent(prompt: str, has_image: bool) -> str:
        rules = IMAGE_INTENT_RULES if has_image else TEXT_INTENT_RULES
        for intent, pattern in rules:
            if pattern.search(prompt):
                return intent
        return "image_analysis" if has_image else "generate"

    async def _llm_intent(self, task: AgentTask) -> str | None:
        system = (
            "You route requests to specialist agents. Classify the user's request into exactly one intent from: "
            + ", ".join(TEXT_INTENTS)
            + '. Reply with JSON only, e.g. {"intent": "summarize"}.'
        )
        try:
            out = await self.generate(task, task.prompt[:2000], system=system, max_tokens=24, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - routing falls back to rules
            logger.info("LLM routing unavailable (%s); using rules", exc)
            return None
        data = extract_json(out.text)
        intent = data.get("intent") if isinstance(data, dict) else None
        return intent if intent in TEXT_INTENTS else None

    def _keyword_agents(self, prompt: str, chosen: list[str]) -> list[str]:
        """Custom agents whose declared keywords appear in the prompt (enables drop-in agents)."""
        registry = self.ctx.registry
        assert registry is not None
        extra: list[str] = []
        for agent in registry.workers():
            if agent.name in chosen or not agent.keywords:
                continue
            if any(re.search(rf"\b{re.escape(k)}\b", prompt, re.I) for k in agent.keywords):
                extra.append(agent.name)
        return extra

    async def run(self, task: AgentTask) -> dict[str, Any]:
        registry = self.ctx.registry
        assert registry is not None
        has_image = task.image_path is not None
        forced: list[str] | None = task.params.get("forced_agents")

        intent = self.classify_intent(task.prompt, has_image)
        routing = "rules"
        if not has_image and not forced and self.ctx.settings.runtime.use_llm_routing:
            llm_intent = await self._llm_intent(task)
            if llm_intent:
                intent, routing = llm_intent, "llm"

        capabilities = list(INTENT_CAPABILITIES.get(intent, INTENT_CAPABILITIES["generate"]))
        if task.params.get("skip_validation"):
            capabilities = [c for c in capabilities if c != "validation"]

        agents: list[str] = []
        missing: list[str] = []
        for cap in capabilities:
            agent = registry.first_for(cap)
            if agent is None:
                missing.append(cap)
            elif agent.name not in agents:
                agents.append(agent.name)

        if forced:
            for name in forced:
                try:
                    agent = registry.get(name)
                except AgentNotFoundError as exc:
                    raise InvalidRequestError(str(exc)) from exc
                if agent.role != "worker":
                    raise InvalidRequestError(f"Agent '{name}' is a control agent and cannot be forced")
            agents, capabilities, missing, routing = list(forced), [], [], "forced"

        extra = [] if forced else self._keyword_agents(task.prompt, agents)
        input_type = "text"
        if has_image:
            input_type = "multimodal" if intent in ("visual_qa", "screenshot_debug") else "image"

        summary = f"intent={intent}; agents={', '.join(agents + extra)}"
        if missing:
            summary += f"; unavailable capabilities={', '.join(missing)}"
        return {
            "intent": intent,
            "input_type": input_type,
            "capabilities": capabilities,
            "agents": agents,
            "extra_agents": extra,
            "forced_agents": forced or [],
            "missing_capabilities": missing,
            "routing": routing,
            "text": summary,
        }
