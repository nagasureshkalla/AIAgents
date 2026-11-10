"""Local-only inference guard: refuses cloud AI configuration and hides cloud LLM keys from the process."""

from __future__ import annotations

import logging
import os
import re

from app.core.config import Settings

logger = logging.getLogger(__name__)

LOCAL_NOTICE = "Cloud AI providers are disabled. All inference is running locally."

# Environment variables of hosted LLM/inference services. Search-provider keys are deliberately not listed.
CLOUD_AI_ENV_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^OPENAI_[A-Z_]*KEY$",
        r"^OPENAI_BASE_URL$",
        r"^AZURE_OPENAI_[A-Z_]*$",
        r"^ANTHROPIC_[A-Z_]*KEY$",
        r"^GEMINI_API_KEY$",
        r"^GOOGLE_API_KEY$",
        r"^GOOGLE_GENAI_[A-Z_]*$",
        r"^GROQ_API_KEY$",
        r"^TOGETHER_API_KEY$",
        r"^OPENROUTER_API_KEY$",
        r"^MISTRAL_API_KEY$",
        r"^COHERE_API_KEY$",
        r"^DEEPSEEK_API_KEY$",
        r"^FIREWORKS_API_KEY$",
        r"^REPLICATE_API_TOKEN$",
        r"^HF_INFERENCE_[A-Z_]*$",
        r"^HUGGINGFACE_INFERENCE_[A-Z_]*$",
    )
)


class CloudInferenceError(RuntimeError):
    """Raised when the configuration asks for anything other than local inference."""


def find_cloud_keys(environ: dict[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return sorted(k for k, v in env.items() if v and any(p.match(k) for p in CLOUD_AI_ENV_PATTERNS))


def enforce_local_inference(settings: Settings, environ: dict[str, str] | None = None) -> list[str]:
    """Validate local-only mode and drop cloud LLM credentials from the environment.

    Returns the names of the removed variables. Raises CloudInferenceError for a non-local ai_mode
    or for the not-yet-supported multi-GPU mode.
    """
    mode = (settings.inference.ai_mode or "").strip().lower()
    if mode != "local":
        raise CloudInferenceError(
            f"AI_MODE={settings.inference.ai_mode!r} is not supported. This application only performs local "
            "inference on your own GPU; set AI_MODE=local."
        )
    if settings.inference.gpu_mode != "single":
        raise CloudInferenceError("GPU_MODE=multi is reserved for future multi-GPU inference; use GPU_MODE=single.")

    from urllib.parse import urlparse

    host = (urlparse(settings.inference.ollama_url).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise CloudInferenceError(
            f"OLLAMA_URL must point at this computer (got {settings.inference.ollama_url!r}); remote inference servers "
            "are not allowed."
        )

    env = os.environ if environ is None else environ
    removed = find_cloud_keys(env)
    for key in removed:
        env.pop(key, None)
    if removed:
        logger.warning("ignoring cloud AI credentials (never used for inference): %s", ", ".join(removed))
    logger.info(LOCAL_NOTICE)
    return removed
