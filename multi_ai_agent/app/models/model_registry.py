"""Model specifications from config.yaml plus filesystem discovery under models/."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
# Rough runtime overhead on top of raw weights (activations, KV cache, CUDA context share).
_MEMORY_OVERHEAD_FACTOR = 1.25
_MEMORY_OVERHEAD_GB = 0.3


class ModelType(StrEnum):
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"


@dataclass
class ModelSpec:
    name: str
    path: Path
    type: ModelType
    hf_id: str | None = None
    priority: str = "normal"
    max_tokens: int | None = None
    dtype: str = "auto"
    est_size_gb: float | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    source: str = "config"

    @property
    def installed(self) -> bool:
        return (self.path / "config.json").is_file() and any(
            p.suffix in _WEIGHT_SUFFIXES for p in self.path.rglob("*") if p.is_file()
        )

    def disk_size_gb(self) -> float:
        if not self.path.is_dir():
            return 0.0
        total = sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file() and p.suffix in _WEIGHT_SUFFIXES)
        return round(total / 1024**3, 3)

    def estimated_memory_gb(self) -> float:
        if self.est_size_gb is not None:
            return self.est_size_gb
        disk = self.disk_size_gb()
        if disk == 0:
            return 1.0
        return round(disk * _MEMORY_OVERHEAD_FACTOR + _MEMORY_OVERHEAD_GB, 2)

    @property
    def is_visual(self) -> bool:
        return self.type in (ModelType.VISION, ModelType.MULTIMODAL)


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = dict(specs or {})

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelRegistry":
        registry = cls()
        for name, cfg in settings.models.items():
            registry.register(
                ModelSpec(
                    name=name,
                    path=settings.resolve_path(cfg.path),
                    type=ModelType(cfg.type),
                    hf_id=cfg.hf_id,
                    priority=cfg.priority,
                    max_tokens=cfg.max_tokens,
                    dtype=cfg.dtype,
                    est_size_gb=cfg.est_size_gb,
                    trust_remote_code=cfg.trust_remote_code,
                    model_kwargs=dict(cfg.model_kwargs),
                    processor_kwargs=dict(cfg.processor_kwargs),
                )
            )
        if settings.discover_models:
            registry.discover(settings.models_dir)
        return registry

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.name] = spec

    def discover(self, models_dir: Path) -> list[ModelSpec]:
        """Register any HF model folder (one containing config.json) under models/{text,vision,multimodal}/."""
        found: list[ModelSpec] = []
        known_paths = {s.path.resolve() for s in self._specs.values()}
        for model_type in ModelType:
            type_dir = models_dir / model_type.value
            if not type_dir.is_dir():
                continue
            for candidate in sorted(type_dir.iterdir()):
                if not (candidate / "config.json").is_file() or candidate.resolve() in known_paths:
                    continue
                name = candidate.name if candidate.name not in self._specs else f"{model_type.value}/{candidate.name}"
                spec = ModelSpec(name=name, path=candidate.resolve(), type=model_type, source="discovered")
                self.register(spec)
                found.append(spec)
                logger.info("discovered local model %s (%s) at %s", name, model_type.value, candidate)
        return found

    def get(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ModelNotFoundError(
                f"Unknown model '{name}'. Known models: {', '.join(sorted(self._specs)) or 'none'}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def list(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def installed(self, model_type: ModelType | None = None) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.installed and (model_type is None or s.type == model_type)]
