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
EngineName = Literal["auto", "transformers", "llama_cpp", "vllm", "ollama"]

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

    path: str = Field(default="", description="Local model folder. Not used for engine: ollama.")
    type: ModelTypeName
    ollama_model: str | None = Field(default=None, description="engine: ollama -> the Ollama tag, e.g. qwen3.6:27b")
    hf_id: str | None = None
    priority: Literal["critical", "high", "normal", "low"] = "normal"
    max_tokens: int | None = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    est_size_gb: float | None = Field(
        default=None, description="Expected memory footprint; estimated from weight files when omitted."
    )
    trust_remote_code: bool = False
    engine: EngineName = Field(default="auto", description="auto picks llama_cpp for GGUF folders, else transformers.")
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
    model_policy: Literal["on_demand", "exclusive"] = Field(
        default="on_demand",
        description="Before a task runs, the manager unloads idle models the chosen agents don't need: only when "
        "room is needed (on_demand) or always (exclusive), then loads the models they do need.",
    )
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


class InferenceConfig(BaseModel):
    """Local-only inference policy. There is deliberately no cloud option."""

    ai_mode: str = Field(default="local", description="Must be 'local'; anything else refuses to start.")
    engine: Literal["transformers", "llama_cpp", "vllm"] = "transformers"
    cuda_device: int | None = Field(default=None, ge=0, description="Pin a CUDA index (default: most free memory).")
    gpu_mode: Literal["single", "multi"] = "single"
    max_context_length: int = Field(default=8192, ge=512)
    max_new_tokens: int = Field(default=2048, ge=1)
    ollama_url: str = Field(default="http://127.0.0.1:11434", description="Local Ollama server (engine: ollama).")


class ControlPanelConfig(BaseModel):
    """The Control Panel agent drives desktop applications through the computer_ai_agent service."""

    enabled: bool = True
    project_dir: str = "../computer_ai_agent"   # where computer_ai_agent (and its .venv) lives
    url: str = "http://127.0.0.1:8765"        # its API / live control panel
    autostart: bool = True                    # start the service on first use if it is not running
    display: str | None = None                # X display to control (None = the service's default, $DISPLAY)
    require_confirmation: bool = True         # ask before destructive actions (delete, send, buy...)
    startup_timeout: float = Field(default=60, gt=0)
    stop_after_task: bool = True              # shut the service down (free its port) when no desktop task is running
    # A failed desktop task is never retried automatically: repeating GUI actions is not safe.


class AirLLMConfig(BaseModel):
    """The AirLLM agent answers with gpt-oss-120b. Engine "ollama" (default) runs it in Ollama (agents.airllm.model,
    split between VRAM and RAM: several tokens/s). Engine "airllm" streams it through the GPU one layer at a time
    with the AirLLM service (~/Desktop/AI/AirLLM): tiny VRAM, ~40 s per token. "ollama" falls back to "airllm"
    when the Ollama model is not pulled."""

    enabled: bool = True
    engine: Literal["ollama", "airllm"] = "ollama"
    ollama_max_new_tokens: int = Field(default=2048, ge=1, le=16384)  # answer + reasoning budget on Ollama
    project_dir: str = "../../AirLLM"           # the AirLLM service (server.py, .venv, models/)
    url: str = "http://127.0.0.1:8770"         # its API (must be this computer)
    autostart: bool = True                     # start the service on first use
    startup_timeout: float = Field(default=120, gt=0)
    load_timeout: float = Field(default=7200, gt=0)      # the first load splits the checkpoint into layers
    max_new_tokens: int = Field(default=96, ge=1, le=4096)  # ~40 s per token on an RTX 3090 (measured)
    reasoning: Literal["low", "medium", "high"] = "low"  # gpt-oss reasoning effort: every token costs seconds
    free_gpu: bool = True                      # unload the other (idle) models first so the layers have room
    stop_after_task: bool = False              # stop the service after each request (frees its host memory)


class WebTesterConfig(BaseModel):
    """The Web Tester agent tests websites through the agent-browser CLI (Chrome over CDP)."""

    enabled: bool = True
    cli: str = "agent-browser"                 # the agent-browser binary (npm i -g agent-browser)
    browser_path: str | None = None           # Chrome/Chromium to drive (None = agent-browser's own lookup)
    headed: bool = False                      # show the browser window while testing
    max_steps: int = Field(default=30, ge=1)  # model-chosen browser commands per test run
    command_timeout: float = Field(default=45, gt=0)
    skills: list[str] = Field(default_factory=lambda: ["core", "dogfood"])  # `agent-browser skills get <name>`
    output_dir: str = "./data/output/web_tests"  # report.md + screenshots, one folder per task
    # TestingTree integration: the flows the agent tested become test cases + suites in a TestingTree project
    export_to_testing_tree: bool = True
    testing_tree_api: str = "http://127.0.0.1:5001"  # TestingTreeServer
    testing_tree_ui: str = "http://localhost:5000"   # TestingTree web app (links in the report)
    run_exported_tests: bool = True             # replay them with TestingTree's Playwright runner and report results
    run_timeout: float = Field(default=300, gt=0)


class JobAgentConfig(BaseModel):
    """The Job Agent searches jobs, scores them against your resume and drafts applications through the Auto-Apply
    AI backend (~/Desktop/AI/JobAgent/Auto-Apply-AI). It never sends an application you did not approve."""

    enabled: bool = True
    project_dir: str = "../../JobAgent/Auto-Apply-AI"  # the Auto-Apply-AI checkout (backend/.venv, run.sh)
    url: str = "http://127.0.0.1:8010"         # its backend API (must be this computer)
    web_url: str = "http://localhost:8000/job"  # its web app, served by this Control Room (profile, resume, Gmail)
    autostart: bool = True                     # start the backend on first use if it is not running
    startup_timeout: float = Field(default=60, gt=0)
    user_email: str | None = None              # Auto-Apply account to act for (ignored in its default local mode)
    resume_path: str | None = None             # resume used to set up the account when none is uploaded yet
    max_results: int = Field(default=8, ge=1, le=30)   # jobs scored and listed per request
    max_drafts: int = Field(default=3, ge=1, le=10)    # applications drafted (and offered for approval) per request
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    approval_timeout: float = Field(default=900, gt=0)  # seconds to wait for Approve/Skip on each application
    request_timeout: float = Field(default=600, gt=0)   # one backend call (scoring runs the local model per job)
    # Browser applications: LinkedIn (search + Easy Apply) and company websites, in a visible Chrome window.
    browser_apply: bool = True                 # apply in the browser where there is no email address
    linkedin: bool = True                      # also search LinkedIn (you log in yourself, once, in that window)
    linkedin_url: str = "https://www.linkedin.com"
    browser_channel: str = "chrome"            # Playwright channel: your installed Google Chrome
    browser_headless: bool = False             # visible, so you can log in, solve checks and watch it apply
    browser_profile_dir: str = "./data/job_browser/profile"   # its own Chrome profile (keeps the LinkedIn login)
    accounts_file: str = "./data/job_browser/accounts.json"   # logins it created on job sites (mode 600)
    answers_file: str = "./data/job_browser/answers.json"     # your answers to form questions, reused next time
    max_browser_applications: int = Field(default=5, ge=1, le=25)  # per request
    max_form_steps: int = Field(default=25, ge=3, le=60)
    page_timeout: float = Field(default=30, gt=0)
    browser_allow_hosts: list[str] = Field(default_factory=list)  # tests only: non-public hosts the browser may open


class ResearchConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    web_enabled: bool = Field(default=True, description="False = LOCAL AI + OFFLINE: no web requests at all.")
    max_iterations: int = Field(default=5, ge=1, le=20)
    max_search_results: int = Field(default=10, ge=1, le=50)
    max_sources: int = Field(default=10, ge=1, le=40)
    default_depth: Literal["quick", "normal", "deep"] = "normal"
    request_timeout: float = Field(default=20, gt=0)
    max_page_mb: float = Field(default=10, gt=0)
    max_concurrent_fetches: int = Field(default=4, ge=1, le=16)
    search_provider: Literal["ddgs", "searxng"] = "ddgs"
    searxng_url: str | None = None
    use_playwright: Literal["auto", "always", "never"] = "auto"
    respect_robots: bool = True
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) LocalResearchAgent/1.0 (+local; respects robots.txt)"
    fast_model: str = "fast_text"
    reasoning_model: str = "research"
    require_cuda: bool = Field(default=True, description="Refuse to run research inference without an NVIDIA GPU.")
    allow_cpu: bool = Field(default=False, description="Debug only: permit research inference on CPU.")


class SpeechConfig(BaseModel):
    """Voice input: the dashboard records your microphone and a local Whisper model turns it into text.
    The Speech agent listens for the wake word and replies with a local Piper voice; no audio or text leaves this computer."""

    enabled: bool = True
    path: str = "./models/speech/whisper"      # hf_id openai/whisper-large-v3-turbo (scripts/download_models.py)
    language: str | None = None                # e.g. "en"; None = detect the language
    max_audio_mb: float = Field(default=20, gt=0)
    max_seconds: float = Field(default=120, gt=0)
    idle_unload_seconds: float = Field(default=300, ge=0)  # free the model's VRAM after this long unused (0 = keep)
    min_rms: float = Field(default=0.008, ge=0)  # quieter recordings are treated as silence (not transcribed)
    # False = conversation mode: everything said (minus noise / filler) is a request; True = only after the wake phrase
    require_wake_word: bool = False
    wake_phrase: str = "Hey CC"                # shown in the dashboard; matching is in app/agents/speech_agent.py
    # "Hey, ..." with the name lost in transcription still wakes when Whisper scores the audio as " Hey CC" above
    # this (said: about -1; not said: below -7, measured on 16 kHz Bluetooth-like audio)
    wake_score_threshold: float = Field(default=-3.5, le=0)
    follow_up_seconds: float = Field(default=8, gt=0)  # after a bare "Hey CC", how long the next utterance counts
    voice: str = "./models/speech/piper"       # Piper voice: an .onnx file, or a folder holding one (+ .onnx.json)
    voice_length_scale: float = Field(default=1.1, gt=0.3, le=3)  # >1 speaks slower (clearer over Bluetooth)
    # silence before each spoken chunk: a Bluetooth headset link wakes up on the first audio and clips the first words
    lead_in_ms: int = Field(default=300, ge=0, le=2000)
    # hands-free: how long a pause ends what you are saying (pausing to think must not send half a request)
    end_of_speech_seconds: float = Field(default=4.0, ge=0.5, le=15)
    max_speak_chars: int = Field(default=4000, gt=0)


class PathsConfig(BaseModel):
    models_dir: str = "./models"
    input_dir: str = "./data/input"
    output_dir: str = "./data/output"
    cache_dir: str = "./data/cache"
    tasks_dir: str = "./data/tasks"


class HarnessConfig(BaseModel):
    """Per-agent prompt harnesses (app/core/harness.py); harness2 improves them from evidence."""

    enabled: bool = True
    root: str = "./harness"
    dirs: dict[str, str] = Field(default_factory=dict)  # agent -> harness directory, overriding root/<agent>


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    control_panel: ControlPanelConfig = Field(default_factory=ControlPanelConfig)
    web_tester: WebTesterConfig = Field(default_factory=WebTesterConfig)
    job_agent: JobAgentConfig = Field(default_factory=JobAgentConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    airllm: AirLLMConfig = Field(default_factory=AirLLMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
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
    "MAA_OLLAMA_URL": ("inference", "ollama_url"),
    "MAA_CONTROL_PANEL_URL": ("control_panel", "url"),
    "MAA_CONTROL_PANEL_DISPLAY": ("control_panel", "display"),
    "MAA_CONTROL_PANEL_DIR": ("control_panel", "project_dir"),
    "MAA_WEB_TESTER_BROWSER": ("web_tester", "browser_path"),
    "MAA_WEB_TESTER_HEADED": ("web_tester", "headed"),
    "MAA_JOB_AGENT_URL": ("job_agent", "url"),
    "MAA_JOB_AGENT_DIR": ("job_agent", "project_dir"),
    "MAA_JOB_AGENT_USER": ("job_agent", "user_email"),
    "MAA_JOB_AGENT_RESUME": ("job_agent", "resume_path"),
    "MAA_AIRLLM_URL": ("airllm", "url"),
    "MAA_AIRLLM_DIR": ("airllm", "project_dir"),
    "MAA_SPEECH_ENABLED": ("speech", "enabled"),
    "MAA_SPEECH_MODEL": ("speech", "path"),
    "MAA_SPEECH_LANGUAGE": ("speech", "language"),
    "MAA_SPEECH_VOICE": ("speech", "voice"),
    "MAA_HARNESS_ENABLED": ("harness", "enabled"),
    "MAA_HARNESS_DIR": ("harness", "root"),
    # Plain names used by the research agent's .env (see .env.example).
    "AI_MODE": ("inference", "ai_mode"),
    "AI_INFERENCE_MODE": ("inference", "ai_mode"),
    "INFERENCE_ENGINE": ("inference", "engine"),
    "DEVICE": ("runtime", "device"),
    "CUDA_DEVICE": ("inference", "cuda_device"),
    "GPU_MODE": ("inference", "gpu_mode"),
    "MAX_CONTEXT_LENGTH": ("inference", "max_context_length"),
    "MAX_NEW_TOKENS": ("inference", "max_new_tokens"),
    "MAX_RESEARCH_ITERATIONS": ("research", "max_iterations"),
    "MAX_SEARCH_RESULTS": ("research", "max_search_results"),
    "MAX_SOURCES": ("research", "max_sources"),
    "RESEARCH_DEPTH": ("research", "default_depth"),
    "REQUEST_TIMEOUT": ("research", "request_timeout"),
    "MAX_PAGE_SIZE_MB": ("research", "max_page_mb"),
    "WEB_ENABLED": ("research", "web_enabled"),
    "SEARCH_PROVIDER": ("research", "search_provider"),
    "SEARXNG_URL": ("research", "searxng_url"),
}

# Model path variables -> model key whose path they override. REASONING_MODEL_PATH wins over MODEL_PATH.
_MODEL_PATH_ENV: list[tuple[str, str]] = [
    ("MODEL_PATH", "research"),
    ("REASONING_MODEL_PATH", "research"),
    ("FAST_MODEL_PATH", "fast_text"),
]


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
        value = raw if key in ("ai_mode", "searxng_url") else yaml.safe_load(raw)
        result.setdefault(section, {})[key] = value
    for env_key, model_key in _MODEL_PATH_ENV:
        raw = os.environ.get(env_key)
        if raw:
            result.setdefault("models", {}).setdefault(model_key, {"type": "text"})["path"] = raw
    dtype = os.environ.get("DTYPE")
    if dtype:
        result.setdefault("models", {}).setdefault("research", {"type": "text"})["dtype"] = dtype
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
        # DTYPE alone must not invent a model entry without a path.
        models = data.get("models") or {}
        for key in [k for k, v in models.items()
                    if isinstance(v, dict) and "path" not in v and v.get("engine") != "ollama"]:
            models.pop(key)
    if overrides:
        data = _deep_merge(data, overrides)
    data.setdefault("base_dir", str(path.parent))
    return Settings.model_validate(data)
