"""Model specifications from config.yaml plus filesystem discovery under models/."""

from __future__ import annotations

import json
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
    engine: str = "auto"  # auto | transformers | llama_cpp | vllm | ollama
    ollama_model: str | None = None  # engine: ollama -> Ollama tag
    ollama_url: str = "http://127.0.0.1:11434"

    @property
    def is_ollama(self) -> bool:
        return self.engine == "ollama"

    @property
    def gguf_file(self) -> Path | None:
        if not self.path.is_dir():
            return None
        files = sorted(self.path.glob("*.gguf"))
        return files[0] if files else None

    @property
    def format(self) -> str:
        if self.is_ollama:
            return "ollama"
        if (self.path / "config.json").is_file():
            return "hf"
        return "gguf" if self.gguf_file is not None else "unknown"

    def resolved_engine(self, default: str = "transformers") -> str:
        if self.engine != "auto":
            return self.engine
        return "llama_cpp" if self.format == "gguf" else default

    @property
    def installed(self) -> bool:
        if self.is_ollama:
            from app.models.engines.ollama_engine import ollama_installed

            return bool(self.ollama_model) and ollama_installed(self.ollama_url, self.ollama_model)
        if self.format == "gguf":
            return True
        return (self.path / "config.json").is_file() and any(
            p.suffix in _WEIGHT_SUFFIXES for p in self.path.rglob("*") if p.is_file()
        )

    def metadata(self) -> dict[str, Any]:
        """Architecture, parameter count estimate, dtype and quantization read from config.json."""
        meta: dict[str, Any] = {"format": self.format, "architecture": None, "quantization": None, "dtype": None}
        if self.is_ollama:
            from app.models.engines.ollama_engine import _norm, ollama_tags

            details = (ollama_tags(self.ollama_url).get(_norm(self.ollama_model or "")) or {}).get("details", {})
            meta.update(architecture=details.get("family"), quantization=details.get("quantization_level"),
                        kind=f"Ollama {details.get('parameter_size', '')}".strip())
            return meta
        if self.format == "gguf":
            meta["architecture"] = "GGUF causal LM"
            name = self.gguf_file.name.lower() if self.gguf_file else ""
            for q in ("q2_k", "q3_k", "q4_k_m", "q4_0", "q5_k_m", "q6_k", "q8_0", "f16"):
                if q in name:
                    meta["quantization"] = q.upper()
            return meta
        cfg_path = self.path / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return meta
        archs = cfg.get("architectures") or []
        meta["architecture"] = archs[0] if archs else cfg.get("model_type")
        meta["model_type"] = cfg.get("model_type")
        meta["dtype"] = cfg.get("torch_dtype") or cfg.get("dtype")
        quant = cfg.get("quantization_config")
        if isinstance(quant, dict):
            meta["quantization"] = quant.get("quant_method") or ("4bit" if quant.get("load_in_4bit") else "8bit")
        meta["context_length"] = cfg.get("max_position_embeddings")
        meta["kind"] = "Causal LM" if any("CausalLM" in a for a in archs) else (
            "Vision-language" if any("Vision" in a or "Image" in a for a in archs) else "Transformer")
        return meta

    def disk_size_gb(self) -> float:
        if self.is_ollama:
            from app.models.engines.ollama_engine import ollama_size_gb

            return ollama_size_gb(self.ollama_url, self.ollama_model or "")
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
            is_ollama = cfg.engine == "ollama"
            registry.register(
                ModelSpec(
                    name=name,
                    path=Path(f"ollama:{cfg.ollama_model}") if is_ollama else settings.resolve_path(cfg.path),
                    type=ModelType(cfg.type),
                    hf_id=cfg.hf_id,
                    priority=cfg.priority,
                    max_tokens=cfg.max_tokens,
                    dtype=cfg.dtype,
                    est_size_gb=cfg.est_size_gb,
                    trust_remote_code=cfg.trust_remote_code,
                    model_kwargs=dict(cfg.model_kwargs),
                    processor_kwargs=dict(cfg.processor_kwargs),
                    engine=cfg.engine,
                    ollama_model=cfg.ollama_model,
                    ollama_url=settings.inference.ollama_url.rstrip("/"),
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
                is_model = (candidate / "config.json").is_file() or any(candidate.glob("*.gguf"))
                if not candidate.is_dir() or not is_model or candidate.resolve() in known_paths:
                    continue
                name = candidate.name if candidate.name not in self._specs else f"{model_type.value}/{candidate.name}"
                spec = ModelSpec(name=name, path=candidate.resolve(), type=model_type, source="discovered")
                self.register(spec)
                found.append(spec)
                logger.info("discovered local model %s (%s) at %s", name, model_type.value, candidate)
        return found

    def refresh(self, models_dir: Path) -> list[ModelSpec]:
        """Re-scan models_dir (e.g. after a download) and return newly discovered models."""
        return self.discover(models_dir)

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
