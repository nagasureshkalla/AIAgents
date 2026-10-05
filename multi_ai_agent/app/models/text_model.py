"""Causal-LM wrapper for local Hugging Face text models."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

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


class TextModel(LocalModel):
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common: dict[str, Any] = {
            "local_files_only": self.settings.runtime.offline_mode,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        self.torch_dtype = resolve_torch_dtype(self.spec.dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.spec.path), **common)
        self.tokenizer.truncation_side = "left"  # keep the end of long prompts (the generation prompt)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"low_cpu_mem_usage": True, **dtype_kwarg(self.torch_dtype), **self.spec.model_kwargs}
        self.model = AutoModelForCausalLM.from_pretrained(str(self.spec.path), **common, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.memory_gb = measure_weights_gb(self.model)
        logger.info("loaded text model %s on %s (%.2f GB)", self.name, self.device, self.memory_gb)

    @property
    def _has_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if self._has_chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n\n".join(lines) + "\n\nAssistant:"

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        import torch

        prompt = self._render_prompt(request.messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.runtime.max_input_tokens,
            # Chat templates already contain BOS/special tokens.
            add_special_tokens=not self._has_chat_template,
        )
        inputs = inputs.to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": request.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
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
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.shape[0]),
            latency=latency,
            confidence=sequence_confidence(self.model, outputs),
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
