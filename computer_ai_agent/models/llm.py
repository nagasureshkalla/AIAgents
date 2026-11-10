"""LocalLLMProvider: one interface over local inference backends. No cloud APIs anywhere.

* OllamaBackend       - talks to the local Ollama server (http://127.0.0.1:11434), which runs on the NVIDIA GPU.
* TransformersBackend - loads a Hugging Face model directory (MODEL_PATH) onto cuda:GPU_DEVICE in-process.

Both accept images, so the same provider serves vision-language models (Qwen-VL, Qwen3.x, Gemma, Llama-Vision...).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image

from core.jsonutil import extract_json

logger = logging.getLogger(__name__)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class ModelError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str                      # system | user | assistant
    content: str
    images: list[Image.Image] = field(default_factory=list)


@dataclass
class Generation:
    text: str
    model: str
    duration_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0

    def json(self) -> Any:
        return extract_json(self.text)


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Backend(ABC):
    @abstractmethod
    async def chat(self, model: str, messages: list[ChatMessage], *, json_mode: bool, max_tokens: int,
                   temperature: float) -> Generation: ...

    @abstractmethod
    async def health(self, model: str) -> dict[str, Any]: ...


class OllamaBackend(Backend):
    def __init__(self, url: str, num_ctx: int, keep_alive: str, timeout: float, gpu_device: int = 0) -> None:
        host = urlparse(url).hostname or ""
        if host not in LOCAL_HOSTS:
            raise ModelError(f"OLLAMA_URL must point at this computer (got {host}); cloud inference is not allowed")
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.gpu_device = gpu_device
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    async def chat(self, model: str, messages: list[ChatMessage], *, json_mode: bool, max_tokens: int,
                   temperature: float) -> Generation:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content, **({"images": [_png_b64(i) for i in m.images]} if m.images else {})}
                for m in messages
            ],
            "stream": False,
            "think": False,  # reasoning traces cost seconds per step; the prompts ask for a short "reason" instead
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens,
                        "main_gpu": self.gpu_device},
        }
        if json_mode:
            payload["format"] = "json"
        t = time.perf_counter()
        try:
            r = await self.client.post(f"{self.url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise ModelError(f"Ollama is not reachable at {self.url}. Start it with: ollama serve") from exc
        if r.status_code == 400 and "think" in r.text:  # model without thinking support rejects the flag
            payload.pop("think")
            r = await self.client.post(f"{self.url}/api/chat", json=payload)
        if r.status_code != 200:
            raise ModelError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return Generation(
            text=data.get("message", {}).get("content", ""),
            model=model,
            duration_ms=(time.perf_counter() - t) * 1000,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )

    async def health(self, model: str) -> dict[str, Any]:
        try:
            tags = (await self.client.get(f"{self.url}/api/tags")).json().get("models", [])
            ps = (await self.client.get(f"{self.url}/api/ps")).json().get("models", [])
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Ollama not reachable at {self.url}: {exc}"}
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        present = model in names or f"{model}:latest" in names
        loaded = next((m for m in ps if m.get("name") in (model, f"{model}:latest")), None)
        return {
            "ok": present,
            "backend": "ollama",
            "model": model,
            "installed": present,
            "loaded": loaded is not None,
            "vram_gb": round((loaded or {}).get("size_vram", 0) / 1e9, 2),
            "error": None if present else f"model {model} not installed: run `ollama pull {model}`",
        }

    async def warmup(self, model: str) -> None:
        await self.client.post(f"{self.url}/api/generate", json={"model": model, "prompt": "", "keep_alive": self.keep_alive})


class TransformersBackend(Backend):
    """Loads MODEL_PATH with Hugging Face transformers on one GPU. Supports text-only and image-text-to-text models."""

    def __init__(self, path: str, gpu_device: int, memory_limit: str = "", max_context: int = 16384) -> None:
        if not path:
            raise ModelError("MODEL_PROVIDER=transformers needs MODEL_PATH (a local model directory)")
        self.path = path
        self.gpu_device = gpu_device
        self.memory_limit = memory_limit
        self.max_context = max_context
        self._model = None
        self._processor = None
        self._multimodal = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        device = f"cuda:{self.gpu_device}" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
                                  "local_files_only": True}
        if device != "cpu":
            kwargs["device_map"] = {"": device}
            if self.memory_limit:
                kwargs["max_memory"] = {self.gpu_device: self.memory_limit}
        cfg = AutoConfig.from_pretrained(self.path, local_files_only=True)
        self._multimodal = hasattr(cfg, "vision_config")
        if self._multimodal:
            from transformers import AutoModelForImageTextToText

            self._processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
            self._model = AutoModelForImageTextToText.from_pretrained(self.path, **kwargs)
        else:
            self._processor = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs)
        self._model.eval()
        logger.info("loaded %s on %s (multimodal=%s)", self.path, device, self._multimodal)

    def _generate(self, messages: list[ChatMessage], max_tokens: int, temperature: float) -> str:
        import torch

        with self._lock:
            self._load()
            proc, model = self._processor, self._model
            if self._multimodal:
                conv = [
                    {"role": m.role, "content": [*({"type": "image", "image": i} for i in m.images),
                                                 {"type": "text", "text": m.content}]}
                    for m in messages
                ]
                inputs = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt")
            else:
                conv = [{"role": m.role, "content": m.content} for m in messages]
                inputs = proc.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt",
                                                  return_dict=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            n_in = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                     temperature=max(temperature, 1e-5) if temperature > 0 else None)
            tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
            return tok.decode(out[0][n_in:], skip_special_tokens=True)

    async def chat(self, model: str, messages: list[ChatMessage], *, json_mode: bool, max_tokens: int,
                   temperature: float) -> Generation:
        t = time.perf_counter()
        text = await asyncio.to_thread(self._generate, messages, max_tokens, temperature)
        return Generation(text=text, model=self.path, duration_ms=(time.perf_counter() - t) * 1000)

    async def health(self, model: str) -> dict[str, Any]:
        import os

        ok = os.path.isdir(self.path)
        return {"ok": ok, "backend": "transformers", "model": self.path, "loaded": self._model is not None,
                "error": None if ok else f"MODEL_PATH {self.path} does not exist"}


class LocalLLMProvider:
    """What agents call. Serializes GPU work (one generation at a time) and records timing."""

    def __init__(self, backend: Backend, model: str, temperature: float = 0.1, max_tokens: int = 700) -> None:
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(1)
        self.calls = 0
        self.total_ms = 0.0
        self.last_ms = 0.0

    async def chat(self, messages: list[ChatMessage], *, json_mode: bool = True, max_tokens: int | None = None,
                   temperature: float | None = None) -> Generation:
        async with self._sem:
            gen = await self.backend.chat(
                self.model, messages, json_mode=json_mode, max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature if temperature is None else temperature,
            )
        self.calls += 1
        self.total_ms += gen.duration_ms
        self.last_ms = gen.duration_ms
        return gen

    async def ask(self, system: str, user: str, images: list[Image.Image] | None = None, **kw: Any) -> Generation:
        return await self.chat([ChatMessage("system", system), ChatMessage("user", user, images or [])], **kw)

    async def health(self) -> dict[str, Any]:
        return await self.backend.health(self.model)
