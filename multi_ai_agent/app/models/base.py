"""Common interface for locally loaded models."""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

    from app.core.config import Settings
    from app.models.model_registry import ModelSpec


@dataclass
class GenerationRequest:
    messages: list[dict[str, str]]  # [{"role": "system"|"user"|"assistant", "content": str}]
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    image: "Image | None" = None


@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    confidence: float | None = None
    model: str = ""
    device: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return round(self.completion_tokens / self.latency, 2) if self.latency > 0 else 0.0


class LocalModel(ABC):
    """A model loaded on a specific device. `generate_sync` runs in a worker thread."""

    def __init__(self, spec: "ModelSpec", device: str, settings: "Settings") -> None:
        self.spec = spec
        self.device = device
        self.settings = settings
        self.memory_gb: float = 0.0
        # HF models are not guaranteed thread-safe; serialize calls on the same instance.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _generate(self, request: GenerationRequest) -> GenerationOutput: ...

    def generate_sync(self, request: GenerationRequest) -> GenerationOutput:
        with self._lock:
            out = self._generate(request)
        out.model = self.name
        out.device = self.device
        return out

    @abstractmethod
    def unload(self) -> None: ...


def resolve_torch_dtype(dtype_name: str, device: str, bf16_supported: bool | None = None) -> Any:
    import torch

    if dtype_name != "auto":
        if dtype_name == "bfloat16" and device.startswith("cuda"):
            supported = torch.cuda.is_bf16_supported() if bf16_supported is None else bf16_supported
            if not supported:  # pre-Ampere NVIDIA GPUs: bf16 is emulated or unsupported
                return torch.float16
        return getattr(torch, dtype_name)
    if device.startswith("cuda"):
        if bf16_supported is None:
            bf16_supported = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16_supported else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def dtype_kwarg(torch_dtype: Any) -> dict[str, Any]:
    """transformers>=4.56 renamed `torch_dtype` to `dtype`."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    return {key: torch_dtype}


def sequence_confidence(model: Any, outputs: Any) -> float | None:
    """Mean per-token probability of the generated sequence, used as a rough confidence score."""
    try:
        import torch

        scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        logprobs = scores[0]
        logprobs = logprobs[torch.isfinite(logprobs)]
        if logprobs.numel() == 0:
            return None
        return round(float(math.exp(logprobs.float().mean().item())), 4)
    except Exception:  # noqa: BLE001 - confidence is optional metadata
        return None


def measure_weights_gb(model: Any) -> float:
    try:
        return round(model.get_memory_footprint() / 1024**3, 3)
    except Exception:  # noqa: BLE001
        return 0.0
