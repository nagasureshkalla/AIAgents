"""Application settings loaded from config.yaml, .env and MAA_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

ModelTypeName = Literal["text", "vision", "multimodal"]
DeviceName = Literal["auto", "cuda", "mps", "cpu"]

_PUBLIC_HOSTS = {"0.0.0.0", "::", "[::]"}


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    allow_public: bool = False
    log_level: str = "info"
    json_logs: bool = False

    @model_validator(mode="after")
    def _check_bind_address(self) -> "ServerConfig":
        if self.host in _PUBLIC_HOSTS and not self.allow_public:
            raise ValueError(
                f"Refusing to bind to {self.host}: the model API is local-only by default. "
                "Set server.allow_public: true (or MAA_ALLOW_PUBLIC=true) to expose it."
            )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    path: str
    type: ModelTypeName
    hf_id: str | None = None
    priority: Literal["critical", "high", "normal", "low"] = "normal"
    max_tokens: int | None = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    est_size_gb: float | None = Field(
        default=None, description="Expected memory footprint; estimated from weight files when omitted."
    )
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    processor_kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str | None = None
    enabled: bool = True
    timeout_seconds: float | None = None


class RuntimeConfig(BaseModel):
    device: DeviceName = "auto"
    max_concurrent_agents: int = Field(default=4, ge=1)
    max_concurrent_tasks: int = Field(default=2, ge=1)
    gpu_concurrency: int = Field(default=1, ge=1)
    cpu_concurrency: int = Field(default=2, ge=1)
    offline_mode: bool = True
    allow_cpu_fallback: bool = True
    vram_threshold: float = Field(default=0.85, gt=0, le=1)
    ram_threshold: float = Field(default=0.90, gt=0, le=1)
    max_loaded_models: int = Field(default=3, ge=1)
    idle_unload_seconds: float = Field(default=0, ge=0, description="0 disables idle unloading.")
    agent_timeout_seconds: float = Field(default=180, gt=0)
    task_timeout_seconds: float = Field(default=900, gt=0)
    max_retries: int = Field(default=1, ge=0)
    max_input_tokens: int = Field(default=4096, ge=256)
    preload: list[str] = Field(default_factory=list)
    use_llm_routing: bool = False
    model_fallbacks: dict[str, list[str]] = Field(default_factory=dict)


class GenerationConfig(BaseModel):
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.2, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    repetition_penalty: float = Field(default=1.05, gt=0)


class SecurityConfig(BaseModel):
    max_request_mb: float = Field(default=25, gt=0)
    max_image_mb: float = Field(default=10, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    max_image_side: int = Field(default=1536, ge=64, description="Images are downscaled to this before inference.")
    allowed_image_formats: list[str] = Field(default_factory=lambda: ["PNG", "JPEG", "WEBP", "BMP", "GIF"])
    max_prompt_chars: int = Field(default=20000, gt=0)

    @field_validator("allowed_image_formats")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [f.upper() for f in v]


class PathsConfig(BaseModel):
    models_dir: str = "./models"
    input_dir: str = "./data/input"
    output_dir: str = "./data/output"
    cache_dir: str = "./data/cache"
    tasks_dir: str = "./data/tasks"


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    discover_models: bool = True
    base_dir: Path = PROJECT_ROOT

    def resolve_path(self, value: str | Path) -> Path:
        p = Path(value).expanduser()
        return (p if p.is_absolute() else self.base_dir / p).resolve()

    @property
    def models_dir(self) -> Path:
        return self.resolve_path(self.paths.models_dir)

    @property
    def input_dir(self) -> Path:
        return self.resolve_path(self.paths.input_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.paths.output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.resolve_path(self.paths.cache_dir)

    @property
    def tasks_dir(self) -> Path:
        return self.resolve_path(self.paths.tasks_dir)

    def agent_config(self, name: str) -> AgentConfig:
        return self.agents.get(name, AgentConfig())


# Environment variable -> (section, key). Values are parsed with yaml so "true"/"8000" get typed.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "MAA_HOST": ("server", "host"),
    "MAA_PORT": ("server", "port"),
    "MAA_ALLOW_PUBLIC": ("server", "allow_public"),
    "MAA_LOG_LEVEL": ("server", "log_level"),
    "MAA_JSON_LOGS": ("server", "json_logs"),
    "MAA_DEVICE": ("runtime", "device"),
    "MAA_OFFLINE_MODE": ("runtime", "offline_mode"),
    "MAA_MAX_CONCURRENT_AGENTS": ("runtime", "max_concurrent_agents"),
    "MAA_GPU_CONCURRENCY": ("runtime", "gpu_concurrency"),
    "MAA_VRAM_THRESHOLD": ("runtime", "vram_threshold"),
    "MAA_MODELS_DIR": ("paths", "models_dir"),
}


def load_dotenv(path: Path) -> None:
    """Minimal .env loader; existing environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ('"', "'") and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:  # inline comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key:
            os.environ.setdefault(key, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for env_key, (section, key) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        result.setdefault(section, {})[key] = yaml.safe_load(raw)
    return result


def load_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    use_env: bool = True,
) -> Settings:
    """Load settings: config.yaml < .env / MAA_* env vars < explicit overrides."""
    path = Path(config_path or os.environ.get("MAA_CONFIG") or DEFAULT_CONFIG_PATH).expanduser().resolve()
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif config_path is not None:
        raise FileNotFoundError(f"Config file not found: {path}")

    if use_env:
        load_dotenv(path.parent / ".env")
        data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)
    data.setdefault("base_dir", str(path.parent))
    return Settings.model_validate(data)
