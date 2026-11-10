"""Builds the configured providers: planner LLM, an optional cheaper fast LLM, and the vision model."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings
from models.llm import Backend, LocalLLMProvider, OllamaBackend, TransformersBackend
from models.vision import LocalVisionModel

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings, backend: Backend | None = None,
                 vision_backend: Backend | None = None) -> None:
        self.settings = settings
        if backend is None:
            backend = self._make_backend(settings.model_path)
        if vision_backend is None:
            if settings.model_provider == "transformers" and settings.vision_model_path:
                vision_backend = self._make_backend(settings.vision_model_path)
            else:
                vision_backend = backend
        self.backend = backend
        self.llm = LocalLLMProvider(backend, settings.model_name if settings.model_provider == "ollama"
                                    else settings.model_path, settings.temperature)
        self.fast = (LocalLLMProvider(backend, settings.fast_model, settings.temperature)
                     if settings.model_provider == "ollama" and settings.fast_model != settings.model_name else self.llm)
        vmodel = settings.vision_model if settings.model_provider == "ollama" else (
            settings.vision_model_path or settings.model_path)
        self.vision = LocalVisionModel(
            self.llm if vision_backend is backend and vmodel in (self.llm.model,)
            else LocalLLMProvider(vision_backend, vmodel, settings.temperature)
        )

    def _make_backend(self, path: str) -> Backend:
        s = self.settings
        if s.model_provider == "ollama":
            return OllamaBackend(s.ollama_url, s.max_context, s.keep_alive, s.model_timeout, s.gpu_device)
        return TransformersBackend(path, s.gpu_device, s.gpu_memory_limit, s.max_context)

    async def health(self) -> dict[str, Any]:
        out = {"provider": self.settings.model_provider, "planner": await self.llm.health()}
        if self.vision.provider is not self.llm:
            out["vision"] = await self.vision.provider.health()
        if self.fast is not self.llm:
            out["fast"] = await self.fast.health()
        out["ok"] = all(v.get("ok") for k, v in out.items() if isinstance(v, dict))
        return out

    async def warmup(self) -> None:
        if isinstance(self.backend, OllamaBackend):
            try:
                await self.backend.warmup(self.llm.model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("model warmup failed: %s", exc)

    def stats(self) -> dict[str, Any]:
        return {
            "planner_model": self.llm.model,
            "vision_model": self.vision.model,
            "calls": self.llm.calls,
            "avg_ms": round(self.llm.total_ms / self.llm.calls) if self.llm.calls else 0,
            "last_ms": round(self.llm.last_ms),
        }
