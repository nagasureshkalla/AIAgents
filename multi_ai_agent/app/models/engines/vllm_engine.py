"""vLLM engine (optional; high-throughput NVIDIA inference for larger Hugging Face models)."""

from __future__ import annotations

import gc
import logging
import time

from app.core.exceptions import ModelLoadError
from app.models.base import GenerationOutput, GenerationRequest, LocalModel

logger = logging.getLogger(__name__)


class VLLMModel(LocalModel):
    def load(self) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ModelLoadError(
                f"Model '{self.name}' is configured with engine: vllm, but vLLM is not installed. "
                "Install it with `pip install vllm` or set engine: transformers."
            ) from exc
        if not self.device.startswith("cuda"):
            raise ModelLoadError("The vLLM engine requires an NVIDIA CUDA GPU")
        dtype = self.spec.dtype if self.spec.dtype != "auto" else "auto"
        self.llm = LLM(
            model=str(self.spec.path),
            dtype=dtype,
            max_model_len=self.settings.inference.max_context_length,
            trust_remote_code=self.spec.trust_remote_code,
            gpu_memory_utilization=self.settings.runtime.vram_threshold,
            **self.spec.model_kwargs,
        )
        self.memory_gb = self.spec.estimated_memory_gb()
        logger.info("loaded model %s via vLLM on %s", self.name, self.device)

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        start = time.perf_counter()
        outputs = self.llm.chat(request.messages, params, use_tqdm=False)
        latency = time.perf_counter() - start
        out = outputs[0]
        return GenerationOutput(
            text=out.outputs[0].text.strip(),
            prompt_tokens=len(out.prompt_token_ids or []),
            completion_tokens=len(out.outputs[0].token_ids),
            latency=latency,
        )

    def unload(self) -> None:
        self.llm = None
        gc.collect()
