"""Reasoning agent: step-by-step interpretation and inference over the request and upstream evidence."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentTask
from app.utils.text import bullet_lines, split_conclusion

_CONCLUDE = "Reason briefly step by step, then finish with a line starting with 'Conclusion:' that gives the answer."

SYSTEM_PROMPTS: dict[str, str] = {
    "interpret_visual": (
        "You are a reasoning specialist. You cannot see the image; rely only on the visual evidence provided by the "
        "vision specialists. Work out what is happening and what it means for the user's request. " + _CONCLUDE
    ),
    "diagnose": (
        "You are a debugging specialist. Using the screenshot analysis provided, identify what is wrong, the most "
        "likely causes, and concrete steps to fix it. " + _CONCLUDE
    ),
    "key_points": (
        "Identify the key claims in the text, how they relate, and any implications or gaps. "
        "Answer with 3-6 short bullet points."
    ),
    "rationale": "Consider the plausible categories for the input and explain briefly which fits best and why. " + _CONCLUDE,
    "explain": "You are a careful analytical thinker. Break the problem down and reason about it. " + _CONCLUDE,
}


class ReasoningAgent(BaseAgent):
    name = "reasoning"
    description = "Step-by-step reasoning, interpretation and inference over the request and other agents' findings."
    capabilities = ["reasoning", "analysis", "interpretation", "inference", "explanation"]
    default_model = "base"

    async def run(self, task: AgentTask) -> dict[str, Any]:
        mode = task.mode or "explain"
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["explain"])
        evidence = self.render_upstream(task)
        user = f"User request:\n{task.prompt}"
        if evidence:
            user += f"\n\nEvidence from specialist agents:\n{evidence}"

        out = await self.generate(task, user, system=system)
        if mode == "key_points":
            points = bullet_lines(out.text)
            return {"mode": mode, "key_points": points, "text": out.text, "confidence": out.confidence}

        reasoning, conclusion = split_conclusion(out.text)
        return {
            "mode": mode,
            "reasoning": reasoning,
            "conclusion": conclusion,
            "key_points": bullet_lines(reasoning, limit=8),
            "text": out.text,
            "confidence": out.confidence,
        }
