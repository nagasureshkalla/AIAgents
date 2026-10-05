"""Vision agent: image understanding, object identification, scene description and image-to-text."""

from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import InvalidImageError
from app.schemas.agent import AgentTask
from app.utils.image import load_image
from app.utils.text import split_list

DESCRIBE_PROMPT = "Describe this image in detail: the setting, the main subjects, and what is happening."
OBJECTS_PROMPT = (
    "List the distinct objects and people visible in this image as a comma-separated list of short nouns. "
    "Output only the list."
)


class VisionAgent(BaseAgent):
    name = "vision"
    description = "Image understanding: scene description, object identification and visual question answering."
    capabilities = [
        "image_analysis",
        "image_description",
        "object_detection",
        "scene_understanding",
        "image_to_text",
        "visual_question_answering",
    ]
    default_model = "vision"

    async def _load_image(self, task: AgentTask) -> Any:
        if not task.image_path:
            raise InvalidImageError("The vision agent requires an image")
        return await asyncio.to_thread(load_image, task.image_path, self.ctx.settings.security)

    async def run(self, task: AgentTask) -> dict[str, Any]:
        image = await self._load_image(task)
        mode = task.mode or "describe"

        if mode == "vqa":
            out = await self.generate(task, task.prompt, image=image, temperature=0.0)
            return {"mode": mode, "answer": out.text, "text": out.text, "confidence": out.confidence, "objects": []}

        # Both prompts go through the scheduler; on a single-GPU setup they run back to back on the same
        # warm model, on CPU or multi-slot setups they overlap.
        description, objects = await asyncio.gather(
            self.generate(task, DESCRIBE_PROMPT, image=image, temperature=0.0),
            self.generate(task, OBJECTS_PROMPT, image=image, max_tokens=96, temperature=0.0),
        )
        return {
            "mode": mode,
            "description": description.text,
            "objects": split_list(objects.text),
            "confidence": description.confidence if description.confidence is not None else 0.0,
            "text": description.text,
        }
