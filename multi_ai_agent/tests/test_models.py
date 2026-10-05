from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core import device as device_mod
from app.core.exceptions import ModelNotFoundError
from app.models.model_manager import ModelState
from app.models.model_registry import ModelRegistry
from tests.fakes import CUDA_DEVICE, FakeFactory, FakeGPU, build_container, install_fake_model, make_settings

GB = 1024**3


# ---------------------------------------------------------------- discovery
def test_registry_reads_config_and_discovers_extra_models(tmp_path):
    settings = make_settings(tmp_path)
    install_fake_model(tmp_path / "models" / "text" / "my_extra_model")
    (tmp_path / "models" / "vision" / "not_a_model").mkdir(parents=True)

    registry = ModelRegistry.from_settings(settings)

    assert {"base", "fast_text", "vision", "multimodal", "my_extra_model"} <= set(registry.names())
    assert registry.get("my_extra_model").source == "discovered"
    assert registry.get("my_extra_model").type.value == "text"
    assert "not_a_model" not in registry.names()
    assert all(s.installed for s in registry.list())


def test_uninstalled_model_reported(tmp_path):
    registry = ModelRegistry.from_settings(make_settings(tmp_path, install=("fast_text",)))
    assert registry.get("fast_text").installed
    assert not registry.get("base").installed


# ---------------------------------------------------------------- loading / unloading
async def test_load_use_unload_lifecycle(container):
    models = container.models
    assert models.state("fast_text") == ModelState.NOT_LOADED

    await models.load_model("fast_text")
    assert models.state("fast_text") == ModelState.LOADED

    key, _ = await models.acquire("fast_text")
    assert models.state("fast_text") == ModelState.ACTIVE
    models.release(key)
    assert models.state("fast_text") == ModelState.IDLE

    assert await models.unload_model("fast_text") is True
    assert models.state("fast_text") == ModelState.UNLOADED
    assert "fast_text" not in models.loaded_models()


async def test_concurrent_loads_happen_once(tmp_path):
    factory = FakeFactory(load_delay=0.05)
    c = build_container(make_settings(tmp_path), factory=factory)
    await asyncio.gather(*(c.models.load_model("base") for _ in range(5)))
    assert factory.calls["base"] == 1


async def test_lru_eviction_when_vram_threshold_exceeded(tmp_path):
    factory = FakeFactory(gpu=FakeGPU(total_gb=4.0))
    settings = make_settings(tmp_path, est_size_gb=1.5, runtime={"device": "cuda", "vram_threshold": 0.85})
    c = build_container(settings, factory=factory, device=CUDA_DEVICE)

    await c.models.load_model("base")
    await c.models.load_model("fast_text")
    await c.models.load_model("vision")  # 4.5 GB > 3.4 GB budget -> evict LRU ("base")

    loaded = set(c.models.loaded_models())
    assert loaded == {"fast_text", "vision"}
    assert c.models.state("base") == ModelState.UNLOADED


async def test_active_model_is_never_evicted(tmp_path):
    factory = FakeFactory(gpu=FakeGPU(total_gb=4.0))
    settings = make_settings(tmp_path, est_size_gb=1.5, runtime={"device": "cuda"})
    c = build_container(settings, factory=factory, device=CUDA_DEVICE)

    key, _ = await c.models.acquire("base")  # in use
    await c.models.load_model("fast_text")
    await c.models.load_model("vision")
    assert "base" in c.models.loaded_models()
    assert "fast_text" not in c.models.loaded_models()
    c.models.release(key)


async def test_max_loaded_models_cap(tmp_path):
    settings = make_settings(tmp_path, runtime={"max_loaded_models": 2})
    c = build_container(settings)
    for name in ("base", "fast_text", "vision"):
        await c.models.load_model(name)
    assert len(c.models.loaded_models()) == 2


async def test_oom_on_load_falls_back_to_cpu(tmp_path):
    factory = FakeFactory(oom_on_load={"base"})
    settings = make_settings(tmp_path, runtime={"device": "cuda", "allow_cpu_fallback": True})
    c = build_container(settings, factory=factory, device=CUDA_DEVICE)

    model = await c.models.load_model("base")
    assert model.device == "cpu"
    assert ("base", "cpu") in factory.devices


async def test_missing_model_uses_fallback_or_fails_clearly(tmp_path):
    c = build_container(make_settings(tmp_path, install=("fast_text",)))
    assert c.models.resolve("base") == "fast_text"
    with pytest.raises(ModelNotFoundError) as exc:
        c.models.resolve("vision")
    assert "download_models.py" in str(exc.value)


# ---------------------------------------------------------------- device detection
def test_device_detection_without_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "_import_torch", lambda: None)
    info = device_mod.get_device("auto")
    assert info.type == "cpu"
    assert info.torch_device == "cpu"


def _fake_torch(cuda: bool, mps: bool = False):
    free = {0: 2 * GB, 1: 10 * GB}
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 2,
            mem_get_info=lambda idx=0: (free[idx], 24 * GB),
            get_device_properties=lambda idx: SimpleNamespace(name=f"FakeRTX-{idx}"),
            is_bf16_supported=lambda: True,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_device_detection_prefers_cuda_with_most_free_memory(monkeypatch):
    monkeypatch.setattr(device_mod, "_import_torch", lambda: _fake_torch(cuda=True))
    info = device_mod.get_device("auto")
    assert info.type == "cuda"
    assert info.torch_device == "cuda:1"
    assert info.name == "FakeRTX-1"
    assert info.vram_total_gb == 24.0


def test_device_priority_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "_import_torch", lambda: _fake_torch(cuda=False, mps=True))
    assert device_mod.get_device("auto").type == "mps"
    assert device_mod.get_device("cpu").type == "cpu"
    assert device_mod.get_device("cuda").type == "cpu"  # requested but unavailable -> CPU with warning


def test_oom_detection():
    assert device_mod.is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 20 MiB"))
    assert not device_mod.is_oom_error(RuntimeError("shape mismatch"))
