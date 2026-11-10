"""Local-only inference guard, research/inference config overrides, and engine selection."""

from __future__ import annotations

import importlib.util

import pytest

from app.cli import render_banner
from app.core.config import load_settings
from app.core.exceptions import ModelLoadError
from app.core.local_guard import CloudInferenceError, enforce_local_inference, find_cloud_keys
from app.models.model_manager import create_local_model
from app.models.model_registry import ModelRegistry, ModelSpec, ModelType
from tests.fakes import CPU_DEVICE, CUDA_DEVICE, make_settings

ENV_KEYS = (
    "AI_MODE", "MODEL_PATH", "REASONING_MODEL_PATH", "FAST_MODEL_PATH", "DTYPE", "CUDA_DEVICE", "WEB_ENABLED",
    "MAX_RESEARCH_ITERATIONS", "GPU_MODE", "DEVICE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def write_config(tmp_path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n  fast_text:\n    path: ./models/text/fast_text\n    type: text\n" + body, encoding="utf-8"
    )
    return str(path)


# ---------------------------------------------------------------- guard
def test_cloud_keys_are_removed_and_never_used(tmp_path):
    env = {"OPENAI_API_KEY": "sk-x", "ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g", "HF_INFERENCE_ENDPOINT": "u",
           "BRAVE_API_KEY": "search-ok", "HF_TOKEN": "download-ok", "PATH": "/bin"}
    removed = enforce_local_inference(make_settings(tmp_path), env)
    assert removed == ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "HF_INFERENCE_ENDPOINT", "OPENAI_API_KEY"]
    assert find_cloud_keys(env) == []
    assert env["BRAVE_API_KEY"] == "search-ok" and env["HF_TOKEN"] == "download-ok"  # search/download keys stay


@pytest.mark.parametrize("mode", ["cloud", "openai", "remote", ""])
def test_non_local_ai_mode_refuses_to_start(tmp_path, mode):
    settings = make_settings(tmp_path)
    settings.inference.ai_mode = mode
    with pytest.raises(CloudInferenceError):
        enforce_local_inference(settings, {})


def test_multi_gpu_mode_is_reserved(tmp_path):
    settings = make_settings(tmp_path)
    settings.inference.gpu_mode = "multi"
    with pytest.raises(CloudInferenceError, match="multi"):
        enforce_local_inference(settings, {})


def test_banner_reports_gpu_and_local_only(tmp_path):
    settings = make_settings(tmp_path)
    registry = ModelRegistry.from_settings(settings)
    gpu = render_banner(settings, CUDA_DEVICE, registry, ["research"])
    assert "GPU: FakeGPU" in gpu and "CUDA: Available" in gpu and "Device: cuda:0" in gpu
    assert "Cloud AI providers are disabled. All inference is running locally." in gpu
    cpu = render_banner(settings, CPU_DEVICE, registry, ["research"])
    assert "NVIDIA CUDA GPU was not detected." in cpu and "cannot start" in cpu


# ---------------------------------------------------------------- config / env
def test_env_overrides_for_research_and_model_paths(tmp_path, monkeypatch):
    cfg = write_config(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "./models/text/Qwen3-8B")
    monkeypatch.setenv("FAST_MODEL_PATH", "./models/text/Qwen3-4B")
    monkeypatch.setenv("DTYPE", "float16")
    monkeypatch.setenv("CUDA_DEVICE", "1")
    monkeypatch.setenv("WEB_ENABLED", "false")
    monkeypatch.setenv("MAX_RESEARCH_ITERATIONS", "3")
    monkeypatch.setenv("AI_MODE", "local")
    s = load_settings(cfg)
    assert s.models["research"].path == "./models/text/Qwen3-8B" and s.models["research"].dtype == "float16"
    assert s.models["fast_text"].path == "./models/text/Qwen3-4B"
    assert s.inference.cuda_device == 1 and s.inference.ai_mode == "local"
    assert s.research.web_enabled is False and s.research.max_iterations == 3


def test_reasoning_model_path_wins_and_dtype_alone_is_ignored(tmp_path, monkeypatch):
    cfg = write_config(tmp_path)
    monkeypatch.setenv("DTYPE", "bfloat16")
    assert "research" not in load_settings(cfg).models  # no path -> no half-defined model
    monkeypatch.setenv("MODEL_PATH", "./a")
    monkeypatch.setenv("REASONING_MODEL_PATH", "./b")
    assert load_settings(cfg).models["research"].path == "./b"


# ---------------------------------------------------------------- engines / discovery
def test_gguf_models_are_discovered_with_llama_cpp_engine(tmp_path):
    folder = tmp_path / "models" / "text" / "qwen-gguf"
    folder.mkdir(parents=True)
    (folder / "qwen2.5-7b-instruct-q4_k_m.gguf").write_bytes(b"GGUF" + b"\0" * 2048)
    settings = make_settings(tmp_path)
    spec = ModelRegistry.from_settings(settings).get("qwen-gguf")
    assert spec.installed and spec.format == "gguf" and spec.resolved_engine() == "llama_cpp"
    assert spec.metadata()["quantization"] == "Q4_K_M"


def test_hf_metadata(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"}')
    (folder / "model.safetensors").write_bytes(b"\0" * 10)
    meta = ModelSpec(name="m", path=folder, type=ModelType.TEXT).metadata()
    assert meta["architecture"] == "Qwen2ForCausalLM" and meta["kind"] == "Causal LM" and meta["format"] == "hf"


@pytest.mark.skipif(importlib.util.find_spec("llama_cpp") is not None, reason="llama-cpp-python is installed")
def test_llama_cpp_engine_reports_missing_package(tmp_path):
    folder = tmp_path / "g"
    folder.mkdir()
    (folder / "m.gguf").write_bytes(b"GGUF")
    spec = ModelSpec(name="g", path=folder, type=ModelType.TEXT)
    with pytest.raises(ModelLoadError, match="llama-cpp-python"):
        create_local_model(spec, "cuda:0", make_settings(tmp_path))


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed")
def test_vllm_engine_reports_missing_package(tmp_path):
    spec = ModelSpec(name="v", path=tmp_path, type=ModelType.TEXT, engine="vllm")
    with pytest.raises(ModelLoadError, match="vLLM is not installed"):
        create_local_model(spec, "cuda:0", make_settings(tmp_path))
