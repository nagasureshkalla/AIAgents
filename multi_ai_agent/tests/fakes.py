"""Test doubles: fake models and a fake GPU so tests need no weights, torch or CUDA."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.core.container import Container
from app.core.device import DeviceInfo
from app.models.base import GenerationOutput, GenerationRequest, LocalModel
from app.models.model_registry import ModelSpec

MODEL_LAYOUT = {
    "base": ("text", "base"),
    "fast_text": ("text", "fast_text"),
    "vision": ("vision", "fast_vision"),
    "multimodal": ("multimodal", "fast_multimodal"),
}

OOM_MESSAGE = "CUDA out of memory. Tried to allocate 2.00 GiB"

CPU_DEVICE = DeviceInfo(type="cpu", torch_device="cpu", name="CPU", ram_total_gb=32, ram_used_gb=4)
CUDA_DEVICE = DeviceInfo(
    type="cuda", torch_device="cuda:0", name="FakeGPU", vram_total_gb=24, vram_used_gb=0, gpu_count=1
)


class FakeGPU:
    def __init__(self, total_gb: float = 24.0) -> None:
        self.total_gb = total_gb
        self.allocated: dict[int, float] = {}

    def probe(self, device: str) -> tuple[float, float]:
        if device == "cpu":
            return 30.0, 32.0
        used = sum(self.allocated.values())
        return self.total_gb - used, self.total_gb


def research_respond(system: str, user: str) -> str | None:
    """Deterministic stand-ins for the research pipeline's prompts (system prompts are lower-cased)."""
    if "planning module of a web research assistant" in system:
        return json.dumps({
            "intent": "find the fact",
            "answer_type": "fact",
            "time_sensitive": True,
            "sub_questions": ["What is the fact?"],
            "queries": ["q1"],
        })
    if "whether collected research evidence is enough" in system:
        match = re.search(r"Evidence so far \((\d+) sources\)", user)
        n = int(match.group(1)) if match else 0
        if n < 2 and "follow up" not in user:
            return json.dumps({"enough": False, "missing": ["second source"], "queries": ["follow up"]})
        return json.dumps({"enough": True, "missing": [], "queries": []})
    if "you extract evidence from one web source" in system:
        text = user.split("Source text:", 1)[-1].strip()
        sentence = re.split(r"(?<=[.!?])\s", text.replace("\n", " "))[0].strip()
        return json.dumps({"claims": [
            {"claim": sentence, "quote": sentence},
            {"claim": "The moon is made of cheese.", "quote": "made of green cheese entirely"},  # untraceable
        ]})
    if "fact-checking module" in system:
        ids = sorted({int(x) for x in re.findall(r"Source \[(\d+)\]", user)})
        if "CONFLICT" in user and len(ids) >= 2:
            return json.dumps({"claims": [{"claim": "The value", "status": "CONFLICTING", "sources": ids[:2], "note": "sources report different values"}]})
        return json.dumps({"claims": [{"claim": "The fact", "status": "SUPPORTED", "sources": ids + [99]}]})
    if "you write the final answer of a web research assistant" in system:
        return (
            "## Answer\n\nThe fact is established [1]. See https://evil.example/fake for more [99].\n\n"
            "## Key Findings\n\n- Finding one [1][2]\n- Finding two [7]\n\n## Sources\n\n[1] Invented https://made.up"
        )
    if "running fully offline" in system:
        return "Offline answer from memory [3] (see https://made.up/page)."
    return None


def respond(request: GenerationRequest) -> str:
    system = " ".join(m["content"] for m in request.messages if m["role"] == "system").lower()
    user = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
    research = research_respond(system, user)
    if research is not None:
        return research
    if "strict reviewer" in system:
        return '{"valid": true, "issues": []}'
    if '"intent"' in system:
        return '{"intent": "summarize"}'
    if request.image is not None:
        if "comma-separated list" in user.lower():
            return "cat, sofa, lamp"
        return "A cat is sleeping on a sofa next to a lamp."
    if "conclusion:" in system:
        return "Step 1: the evidence shows a cat on a sofa near a lamp.\nConclusion: A cat is resting on a sofa."
    if "final response writer" in system:
        return "Final answer: a cat is resting on a sofa next to a lamp."
    return f"Answer: {user[:80]}"


class FakeModel(LocalModel):
    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        settings: Settings,
        gpu: FakeGPU,
        latency: float,
        oom_on_generate: bool,
    ) -> None:
        super().__init__(spec, device, settings)
        self.gpu = gpu
        self.latency = latency
        self.oom_on_generate = oom_on_generate

    def load(self) -> None:
        self.memory_gb = self.spec.estimated_memory_gb()
        if self.device != "cpu":
            self.gpu.allocated[id(self)] = self.memory_gb

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        if self.oom_on_generate and self.device != "cpu":
            raise RuntimeError(OOM_MESSAGE)
        time.sleep(self.latency)
        text = respond(request)
        return GenerationOutput(
            text=text,
            prompt_tokens=sum(len(m["content"].split()) for m in request.messages),
            completion_tokens=len(text.split()),
            latency=self.latency,
            confidence=0.9,
        )

    def unload(self) -> None:
        self.gpu.allocated.pop(id(self), None)


class FakeFactory:
    """Callable model factory recording loads; can inject load/inference OOM per model name."""

    def __init__(
        self,
        gpu: FakeGPU | None = None,
        latency: float = 0.01,
        oom_on_load: set[str] | None = None,
        oom_on_generate: set[str] | None = None,
        load_delay: float = 0.0,
    ) -> None:
        self.gpu = gpu or FakeGPU()
        self.latency = latency
        self.oom_on_load = oom_on_load or set()
        self.oom_on_generate = oom_on_generate or set()
        self.load_delay = load_delay
        self.calls: Counter[str] = Counter()
        self.devices: list[tuple[str, str]] = []

    def __call__(self, spec: ModelSpec, device: str, settings: Settings) -> LocalModel:
        self.calls[spec.name] += 1
        self.devices.append((spec.name, device))
        if self.load_delay:
            time.sleep(self.load_delay)
        if spec.name in self.oom_on_load and device != "cpu":
            raise RuntimeError(OOM_MESSAGE)
        model = FakeModel(spec, device, settings, self.gpu, self.latency, spec.name in self.oom_on_generate)
        model.load()
        return model


def install_fake_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"\0" * 1024)


def make_settings(
    tmp_path: Path,
    *,
    runtime: dict[str, Any] | None = None,
    security: dict[str, Any] | None = None,
    install: tuple[str, ...] = tuple(MODEL_LAYOUT),
    est_size_gb: float = 1.0,
) -> Settings:
    models_dir = tmp_path / "models"
    models: dict[str, Any] = {}
    for key, (model_type, folder) in MODEL_LAYOUT.items():
        path = models_dir / model_type / folder
        if key in install:
            install_fake_model(path)
        models[key] = {"path": str(path), "type": model_type, "est_size_gb": est_size_gb, "hf_id": f"fake/{key}"}
    data: dict[str, Any] = {
        "harness": {"root": str(tmp_path / "harness")},
        "models": models,
        "agents": {
            "manager": {"model": "fast_text"},
            "text": {"model": "fast_text"},
            "vision": {"model": "vision"},
            "multimodal": {"model": "multimodal"},
            "reasoning": {"model": "base"},
            "validation": {"model": "fast_text"},
            "aggregator": {"model": "base"},
        },
        "runtime": {
            "device": "cpu",
            "offline_mode": True,
            "preload": [],
            "max_retries": 1,
            "agent_timeout_seconds": 10,
            "task_timeout_seconds": 30,
            "cpu_concurrency": 4,
            "model_fallbacks": {"base": ["fast_text"], "multimodal": ["vision"]},
            **(runtime or {}),
        },
        "security": security or {},
        "paths": {
            "models_dir": str(models_dir),
            "input_dir": str(tmp_path / "data" / "input"),
            "output_dir": str(tmp_path / "data" / "output"),
            "cache_dir": str(tmp_path / "data" / "cache"),
            "tasks_dir": str(tmp_path / "data" / "tasks"),
        },
        "base_dir": str(tmp_path),
    }
    return Settings.model_validate(data)


def build_container(
    settings: Settings,
    *,
    factory: FakeFactory | None = None,
    device: DeviceInfo = CPU_DEVICE,
) -> Container:
    factory = factory or FakeFactory()
    return Container.build(settings, device_info=device, model_factory=factory, probe=factory.gpu.probe)


def make_image(directory: Path, name: str = "image.png", size: tuple[int, int] = (64, 48)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size, (200, 120, 40)).save(path, format="PNG")
    return path
