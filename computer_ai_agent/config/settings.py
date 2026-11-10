"""Runtime configuration, read from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass
class Settings:
    # --- models -----------------------------------------------------------------------------------------
    # "ollama": a local Ollama server (GPU). "transformers": load MODEL_PATH with Hugging Face on GPU_DEVICE.
    model_provider: str = "ollama"
    model_name: str = "qwen3.6:27b"           # planner / reasoning model (Ollama tag or HF id)
    model_path: str = ""                      # HF weights directory when model_provider=transformers
    vision_model_name: str = ""               # empty = reuse model_name (it must accept images)
    vision_model_path: str = ""
    fast_model_name: str = ""                 # optional small model for cheap calls (routing); empty = model_name
    ollama_url: str = "http://127.0.0.1:11434"
    gpu_device: int = 0
    gpu_memory_limit: str = ""                # e.g. "20GiB" for transformers; Ollama manages its own memory
    max_context: int = 16384
    temperature: float = 0.1
    model_timeout: float = 180.0
    keep_alive: str = "30m"                   # how long Ollama keeps the model in VRAM between calls

    # --- desktop ----------------------------------------------------------------------------------------
    display: str = ""                         # X display to control; empty = $DISPLAY
    screen_fps: float = 2.0                   # live preview frames per second in the web UI
    model_image_width: int = 1280             # screenshots are downscaled to this width before the model sees them
    vision_mode: str = "auto"                 # auto | always | never: when to send the screenshot to the model
    ocr_enabled: bool = True
    accessibility_enabled: bool = True
    action_delay: float = 0.35                # settle time after each GUI action before observing again
    settle_timeout: float = 1.2               # max wait for the target window to stop changing after an action
    nav_timeout: float = 5.0                  # max wait for a page/view change after a link click, Enter, etc.
    launch_timeout: float = 25.0

    # --- agent loop -------------------------------------------------------------------------------------
    max_action_retries: int = 3
    max_steps_per_subtask: int = 25
    require_confirmation: bool = True
    confirmation_timeout: float = 300.0

    # --- server / storage -------------------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8765
    debug: bool = False
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    logs_dir: Path = field(default_factory=lambda: ROOT / "logs")
    harness_dir: Path | None = field(default_factory=lambda: ROOT / "harness")  # planner/harness.py; None = off

    @property
    def vision_model(self) -> str:
        return self.vision_model_name or self.model_name

    @property
    def fast_model(self) -> str:
        return self.fast_model_name or self.model_name


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env", override=False)
    s = Settings(
        model_provider=_str("MODEL_PROVIDER", "ollama").lower(),
        model_name=_str("MODEL_NAME", Settings.model_name),
        model_path=_str("MODEL_PATH", ""),
        vision_model_name=_str("VISION_MODEL_NAME", ""),
        vision_model_path=_str("VISION_MODEL_PATH", ""),
        fast_model_name=_str("FAST_MODEL_NAME", ""),
        ollama_url=_str("OLLAMA_URL", Settings.ollama_url).rstrip("/"),
        gpu_device=_int("GPU_DEVICE", 0),
        gpu_memory_limit=_str("GPU_MEMORY_LIMIT", ""),
        max_context=_int("MAX_CONTEXT", Settings.max_context),
        temperature=_float("TEMPERATURE", Settings.temperature),
        model_timeout=_float("MODEL_TIMEOUT", Settings.model_timeout),
        keep_alive=_str("MODEL_KEEP_ALIVE", Settings.keep_alive),
        display=_str("AGENT_DISPLAY", os.getenv("DISPLAY", "")),
        screen_fps=_float("SCREEN_FPS", Settings.screen_fps),
        model_image_width=_int("MODEL_IMAGE_WIDTH", Settings.model_image_width),
        vision_mode=_str("VISION_MODE", "auto").lower(),
        ocr_enabled=_bool("OCR_ENABLED", True),
        accessibility_enabled=_bool("ACCESSIBILITY_ENABLED", True),
        action_delay=_float("ACTION_DELAY", Settings.action_delay),
        settle_timeout=_float("SETTLE_TIMEOUT", Settings.settle_timeout),
        nav_timeout=_float("NAV_TIMEOUT", Settings.nav_timeout),
        launch_timeout=_float("LAUNCH_TIMEOUT", Settings.launch_timeout),
        max_action_retries=_int("MAX_ACTION_RETRIES", 3),
        max_steps_per_subtask=_int("MAX_STEPS_PER_SUBTASK", Settings.max_steps_per_subtask),
        require_confirmation=_bool("REQUIRE_CONFIRMATION", True),
        confirmation_timeout=_float("CONFIRMATION_TIMEOUT", Settings.confirmation_timeout),
        host=_str("HOST", Settings.host),
        port=_int("PORT", Settings.port),
        debug=_bool("DEBUG", False),
    )
    if os.getenv("DATA_DIR"):
        s.data_dir = Path(os.environ["DATA_DIR"])
    if os.getenv("LOGS_DIR"):
        s.logs_dir = Path(os.environ["LOGS_DIR"])
    if os.getenv("HARNESS_DIR"):
        s.harness_dir = Path(os.environ["HARNESS_DIR"])
    if s.model_provider not in ("ollama", "transformers"):
        raise ValueError(f"MODEL_PROVIDER must be 'ollama' or 'transformers', got {s.model_provider!r}")
    if s.vision_mode not in ("auto", "always", "never"):
        raise ValueError(f"VISION_MODE must be auto, always or never, got {s.vision_mode!r}")
    return s
