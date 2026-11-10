"""llama.cpp engine for GGUF models (requires llama-cpp-python built with CUDA)."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

from app.core.exceptions import ModelLoadError
from app.models.base import GenerationOutput, GenerationRequest, LocalModel

logger = logging.getLogger(__name__)


class LlamaCppModel(LocalModel):
    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelLoadError(
                f"Model '{self.name}' is GGUF and needs llama-cpp-python. Install a CUDA build: "
                'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python'
            ) from exc
        gguf = self.spec.gguf_file
        if gguf is None:
            raise ModelLoadError(f"No .gguf file found in {self.spec.path}")
        on_gpu = self.device.startswith("cuda")
        main_gpu = int(self.device.split(":", 1)[1]) if on_gpu and ":" in self.device else 0
        kwargs: dict[str, Any] = {
            "model_path": str(gguf),
            "n_gpu_layers": -1 if on_gpu else 0,  # all layers on the GPU
            "main_gpu": main_gpu,
            "n_ctx": self.settings.inference.max_context_length,
            "verbose": False,
            **self.spec.model_kwargs,
        }
        self.llm = Llama(**kwargs)
        self.memory_gb = self.spec.disk_size_gb()
        logger.info("loaded GGUF model %s on %s via llama.cpp (%.2f GB)", self.name, self.device, self.memory_gb)

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        if request.image is not None:
            raise ModelLoadError(f"llama.cpp model '{self.name}' does not accept images")
        start = time.perf_counter()
        resp = self.llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repeat_penalty=request.repetition_penalty,
        )
        latency = time.perf_counter() - start
        usage = resp.get("usage") or {}
        return GenerationOutput(
            text=(resp["choices"][0]["message"].get("content") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency=latency,
        )

    def unload(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None and hasattr(llm, "close"):
            llm.close()
        self.llm = None
        gc.collect()
