"""Image-text-to-text wrapper for local Hugging Face vision / multimodal models (e.g. SmolVLM, Qwen2-VL)."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

from app.core.exceptions import InvalidRequestError, ModelLoadError
from app.models.base import (
    GenerationOutput,
    GenerationRequest,
    LocalModel,
    dtype_kwarg,
    measure_weights_gb,
    resolve_torch_dtype,
    sequence_confidence,
)

logger = logging.getLogger(__name__)


def _vlm_auto_classes() -> list[Any]:
    classes: list[Any] = []
    try:
        from transformers import AutoModelForImageTextToText

        classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    return classes


class VisionModel(LocalModel):
    def load(self) -> None:
        from transformers import AutoProcessor

        common: dict[str, Any] = {
            "local_files_only": self.settings.runtime.offline_mode,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        self.torch_dtype = resolve_torch_dtype(self.spec.dtype, self.device)
        self.processor = AutoProcessor.from_pretrained(str(self.spec.path), **common, **self.spec.processor_kwargs)

        model_kwargs = {"low_cpu_mem_usage": True, **dtype_kwarg(self.torch_dtype), **self.spec.model_kwargs}
        last_error: Exception | None = None
        self.model = None
        for auto_cls in _vlm_auto_classes():
            try:
                self.model = auto_cls.from_pretrained(str(self.spec.path), **common, **model_kwargs)
                break
            except ValueError as exc:  # architecture not mapped by this auto class; try the next one
                last_error = exc
        if self.model is None:
            raise ModelLoadError(f"No image-text-to-text architecture found for {self.spec.path}: {last_error}")

        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.memory_gb = measure_weights_gb(self.model)
        logger.info("loaded vision model %s on %s (%.2f GB)", self.name, self.device, self.memory_gb)

    @property
    def _tokenizer(self) -> Any:
        return getattr(self.processor, "tokenizer", self.processor)

    def _build_prompt(self, request: GenerationRequest) -> str:
        # Many small VLM chat templates have no system role, so fold system text into the user turn.
        system = "\n".join(m["content"] for m in request.messages if m["role"] == "system")
        user_parts = [m["content"] for m in request.messages if m["role"] == "user"]
        user_text = "\n\n".join(p for p in (system, *user_parts) if p)
        if getattr(self.processor, "chat_template", None):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
            return self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return f"<image>\n{user_text}"

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        import torch

        if request.image is None:
            raise InvalidRequestError(f"Vision model '{self.name}' requires an image")

        prompt = self._build_prompt(request)
        inputs = self.processor(text=prompt, images=[request.image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        prompt_len = int(inputs["input_ids"].shape[1])

        tok = self._tokenizer
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": request.repetition_penalty,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if request.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=request.temperature, top_p=request.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - start

        new_tokens = outputs.sequences[0, prompt_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.shape[0]),
            latency=latency,
            confidence=sequence_confidence(self.model, outputs),
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
