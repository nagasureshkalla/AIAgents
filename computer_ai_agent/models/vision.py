"""LocalVisionModel: screen understanding with a local vision-language model (same backends as the LLM)."""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from models.llm import LocalLLMProvider

logger = logging.getLogger(__name__)

LOCATE_SYSTEM = (
    "You locate user-interface elements in screenshots. The image is {w}x{h} pixels. "
    'Reply with JSON only: {{"found": true, "x": <int>, "y": <int>, "label": "<what you found>"}} giving the '
    'center of the element in image pixels, or {{"found": false, "reason": "..."}} if it is not visible.'
)
DESCRIBE_SYSTEM = (
    "You describe application screenshots for a computer-control agent. Be factual and brief. Reply with JSON only: "
    '{"application": "...", "state": "<one sentence>", "dialogs": ["..."], "notable": ["visible text or controls '
    'relevant to the task"]}'
)


class LocalVisionModel:
    def __init__(self, provider: LocalLLMProvider) -> None:
        self.provider = provider

    @property
    def model(self) -> str:
        return self.provider.model

    async def describe(self, image: Image.Image, task: str = "") -> dict[str, Any]:
        gen = await self.provider.ask(DESCRIBE_SYSTEM, f"Task context: {task or 'none'}", [image], max_tokens=300)
        data = gen.json()
        return data if isinstance(data, dict) else {"state": gen.text[:300]}

    async def locate(self, image: Image.Image, description: str) -> tuple[int, int] | None:
        """Pixel center of `description` in `image` coordinates (caller rescales to the screen), or None."""
        system = LOCATE_SYSTEM.format(w=image.width, h=image.height)
        gen = await self.provider.ask(system, f"Find: {description}", [image], max_tokens=120, temperature=0.0)
        data = gen.json()
        if not isinstance(data, dict) or not data.get("found"):
            return None
        try:
            x, y = int(data["x"]), int(data["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= x < image.width and 0 <= y < image.height):
            return None
        return x, y
