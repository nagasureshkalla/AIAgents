"""Multimodal agent: answers text questions grounded in an image (screenshots, charts, photos)."""

from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import InvalidImageError
from app.schemas.agent import AgentTask
from app.utils.image import load_image

SCREENSHOT_PROMPT = (
    "You are inspecting a software screenshot. Read any visible text, error messages, warnings or dialogs, "
    "and point out anything broken, misaligned, missing or unusual. Then answer the question.\n\nQuestion: {prompt}"
)


class MultimodalAgent(BaseAgent):
    name = "multimodal"
    description = "Joint text + image reasoning: visual question answering and screenshot analysis."
    capabilities = ["visual_question_answering", "screenshot_analysis", "multimodal_reasoning", "text_image_qa"]
    default_model = "multimodal"

    async def run(self, task: AgentTask) -> dict[str, Any]:
        if not task.image_path:
            raise InvalidImageError("The multimodal agent requires an image")
        image = await asyncio.to_thread(load_image, task.image_path, self.ctx.settings.security)
        mode = task.mode or "vqa"
        prompt = SCREENSHOT_PROMPT.format(prompt=task.prompt) if mode == "screenshot" else task.prompt
        out = await self.generate(task, prompt, image=image, temperature=0.0)
        return {"mode": mode, "answer": out.text, "text": out.text, "confidence": out.confidence}
