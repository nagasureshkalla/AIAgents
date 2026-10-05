"""Text agent: understanding, summarization, classification, extraction, generation and formatting."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentTask
from app.utils.text import bullet_lines, extract_json

SYSTEM_PROMPTS: dict[str, str] = {
    "summarize": (
        "You are a precise summarization assistant. Summarize the text faithfully and concisely, "
        "as a short paragraph followed by the key points as bullets. Do not add facts that are not in the text."
    ),
    "classify": (
        "You are a text classifier. Put the single best label on the first line, then a one-sentence justification. "
        "If the user lists allowed labels, use one of them."
    ),
    "extract": (
        "You extract structured information. Return only compact, valid JSON with the requested fields. "
        "Use null for anything that is not present in the text."
    ),
    "format": "Reformat the content exactly as requested. Preserve every piece of information. Output only the result.",
    "interpret_request": (
        "Restate in 2-4 short bullet points what the user is asking for and what a complete answer must include. "
        "Do not answer the request itself."
    ),
    "compose": (
        "Write the final answer for the user using the analysis provided. "
        "Be clear, well-structured and concise. Do not mention the analysis process or other agents."
    ),
    "generate": "You are a helpful, accurate assistant. Answer concisely. If you are unsure, say so.",
}


class TextAgent(BaseAgent):
    name = "text"
    description = "Text understanding, summarization, classification, extraction, generation and formatting."
    capabilities = [
        "text_generation",
        "summarization",
        "classification",
        "extraction",
        "formatting",
        "request_interpretation",
        "text_understanding",
    ]
    default_model = "fast_text"

    async def run(self, task: AgentTask) -> dict[str, Any]:
        mode = task.mode or "generate"
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["generate"])
        context = self.render_upstream(task)

        if mode == "interpret_request" and task.image_path:
            user = f"User request (an image is attached and analyzed by another agent):\n{task.prompt}"
        elif context:
            user = f"User request:\n{task.prompt}\n\nContext from other specialists:\n{context}"
        else:
            user = task.prompt

        max_tokens = 160 if mode in ("interpret_request", "classify") else None
        temperature = 0.0 if mode in ("classify", "extract", "format") else None
        out = await self.generate(task, user, system=system, max_tokens=max_tokens, temperature=temperature)

        result: dict[str, Any] = {"mode": mode, "text": out.text, "confidence": out.confidence}
        if mode == "extract":
            result["data"] = extract_json(out.text)
        elif mode == "classify":
            lines = [line.strip() for line in out.text.splitlines() if line.strip()]
            result["label"] = lines[0].strip("*: ") if lines else ""
            result["justification"] = " ".join(lines[1:])
        elif mode in ("interpret_request", "summarize"):
            result["points"] = bullet_lines(out.text)
        return result
