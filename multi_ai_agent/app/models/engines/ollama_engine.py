"""Ollama engine: a model served by the local Ollama server, managed like any other local model.

load()   -> asks Ollama to load the model into VRAM and keep it there (keep_alive=-1m: the ModelManager, not
            Ollama's idle timer, decides when it leaves the GPU)
unload() -> keep_alive=0, then waits until Ollama has actually released the VRAM
Inference stays on this computer: the URL is checked by app.core.local_guard.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

import httpx

from app.core.exceptions import ModelLoadError, ModelNotFoundError
from app.models.base import GenerationOutput, GenerationRequest, LocalModel

logger = logging.getLogger(__name__)

KEEP_FOREVER = "-1m"
_GiB = 1024**3
_tags_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}
_tags_lock = threading.Lock()


def _norm(tag: str) -> str:
    return tag if ":" in tag else f"{tag}:latest"


def ollama_tags(url: str, max_age: float = 15.0) -> dict[str, dict[str, Any]]:
    """Installed Ollama models (name -> info), cached briefly. Empty when Ollama is not running."""
    now = time.monotonic()
    with _tags_lock:
        hit = _tags_cache.get(url)
        if hit and now - hit[0] < max_age:
            return hit[1]
    try:
        models = httpx.get(f"{url}/api/tags", timeout=1.5).json().get("models", [])
        tags = {m["name"]: m for m in models}
    except (httpx.HTTPError, ValueError):
        tags = {}
    with _tags_lock:
        _tags_cache[url] = (now, tags)
    return tags


def ollama_loaded(url: str) -> dict[str, dict[str, Any]]:
    """Models currently in Ollama's memory (name -> info incl. size_vram)."""
    try:
        return {m["name"]: m for m in httpx.get(f"{url}/api/ps", timeout=2).json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return {}


def ollama_num_ctx(spec: Any, settings: Any) -> int:
    """Context size the model is loaded with. Every caller must send the same value: Ollama reloads the model when a
    request asks for a different num_ctx."""
    return int(spec.model_kwargs.get("num_ctx") or settings.inference.max_context_length)


def ollama_installed(url: str, tag: str) -> bool:
    return _norm(tag) in ollama_tags(url)


def ollama_size_gb(url: str, tag: str) -> float:
    info = ollama_tags(url).get(_norm(tag))
    return round(info.get("size", 0) / _GiB, 2) if info else 0.0


class OllamaModel(LocalModel):
    @property
    def tag(self) -> str:
        if not self.spec.ollama_model:
            raise ModelLoadError(f"Model '{self.name}' uses engine: ollama but has no ollama_model tag")
        return _norm(self.spec.ollama_model)

    @property
    def url(self) -> str:
        return self.settings.inference.ollama_url.rstrip("/")

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def load(self) -> None:
        if not ollama_installed(self.url, self.tag):
            if not ollama_tags(self.url, max_age=0):
                raise ModelLoadError(f"Ollama is not reachable at {self.url}. Start it with: ollama serve")
            raise ModelNotFoundError(
                f"Ollama model '{self.tag}' is not installed. Download it with: ollama pull {self.tag}",
                details={"model": self.name, "ollama_model": self.tag},
            )
        main_gpu = int(self.device.split(":", 1)[1]) if self.device.startswith("cuda:") else 0
        try:
            with self._client(600) as c:
                r = c.post(f"{self.url}/api/generate", json={
                    "model": self.tag, "prompt": "", "keep_alive": KEEP_FOREVER,
                    "options": {"main_gpu": main_gpu, "num_ctx": ollama_num_ctx(self.spec, self.settings)},
                })
        except httpx.HTTPError as exc:
            raise ModelLoadError(f"Ollama failed to load '{self.tag}': {exc}") from exc
        if r.status_code != 200:
            raise ModelLoadError(f"Ollama failed to load '{self.tag}': {r.status_code} {r.text[:200]}")
        info = ollama_loaded(self.url).get(self.tag, {})
        self.memory_gb = round(info.get("size_vram", 0) / _GiB, 2) or ollama_size_gb(self.url, self.tag)
        if not info.get("size_vram") and self.device.startswith("cuda"):
            logger.warning("Ollama loaded %s but reports no VRAM use: it may be running on the CPU", self.tag)
        logger.info("loaded Ollama model %s (%.2f GB VRAM)", self.tag, self.memory_gb)

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        if request.image is not None:
            buf = io.BytesIO()
            request.image.convert("RGB").save(buf, format="PNG")
            last_user = next(m for m in reversed(messages) if m["role"] == "user")
            last_user["images"] = [base64.b64encode(buf.getvalue()).decode()]
        payload: dict[str, Any] = {
            # think: off, except models configured with a reasoning level (gpt-oss: model_kwargs.think "low")
            "model": self.tag, "messages": messages, "stream": False, "think": self.spec.model_kwargs.get("think", False),
            "keep_alive": KEEP_FOREVER,
            "options": {"temperature": request.temperature, "top_p": request.top_p,
                        "repeat_penalty": request.repetition_penalty, "num_predict": request.max_new_tokens,
                        "num_ctx": ollama_num_ctx(self.spec, self.settings)},
        }
        start = time.perf_counter()
        with self._client(600) as c:
            r = c.post(f"{self.url}/api/chat", json=payload)
            if r.status_code == 400 and "think" in r.text:  # model without a thinking mode
                payload.pop("think")
                r = c.post(f"{self.url}/api/chat", json=payload)
        if r.status_code != 200:
            raise ModelLoadError(f"Ollama generation failed for '{self.tag}': {r.status_code} {r.text[:200]}")
        data = r.json()
        return GenerationOutput(
            text=(data.get("message", {}).get("content") or "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency=time.perf_counter() - start,
        )

    def unload(self) -> None:
        try:
            with self._client(60) as c:
                c.post(f"{self.url}/api/generate", json={"model": self.tag, "keep_alive": 0})
        except httpx.HTTPError as exc:
            logger.warning("Ollama unload of %s failed: %s", self.tag, exc)
            return
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and self.tag in ollama_loaded(self.url):
            time.sleep(0.25)  # wait for the VRAM to be released before the next model is loaded
        self.memory_gb = 0.0
