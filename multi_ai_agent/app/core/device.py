"""Compute device detection (CUDA -> MPS -> CPU) and memory probing."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)

_GB = 1024**3


def _import_torch() -> Any | None:
    try:
        import torch

        return torch
    except ImportError:  # torch is optional for API-only / test usage
        return None


@dataclass
class DeviceInfo:
    type: str  # "cuda" | "mps" | "cpu"
    torch_device: str  # e.g. "cuda:0", "mps", "cpu"
    name: str
    vram_total_gb: float | None = None
    vram_used_gb: float | None = None
    ram_total_gb: float | None = None
    ram_used_gb: float | None = None
    gpu_count: int = 0
    torch_available: bool = False
    bf16_supported: bool = False

    @property
    def is_gpu(self) -> bool:
        return self.type in ("cuda", "mps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_available() -> bool:
    torch = _import_torch()
    return bool(torch is not None and torch.cuda.is_available())


def mps_available() -> bool:
    torch = _import_torch()
    if torch is None:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def best_cuda_index() -> int:
    """Pick the CUDA device with the most free memory instead of hard-coding cuda:0."""
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return 0
    best, best_free = 0, -1
    for idx in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(idx)
        except Exception:  # noqa: BLE001 - some drivers refuse mem_get_info
            continue
        if free > best_free:
            best, best_free = idx, free
    return best


def detect_device_type(preference: str = "auto") -> str:
    preference = (preference or "auto").lower()
    if preference == "cuda":
        if cuda_available():
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if preference == "mps":
        if mps_available():
            return "mps"
        logger.warning("MPS requested but not available; falling back to CPU")
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def get_device(preference: str = "auto", cuda_index: int | None = None) -> DeviceInfo:
    """Detect the best available device according to `preference`; `cuda_index` pins a specific GPU."""
    torch = _import_torch()
    dev_type = detect_device_type(preference)
    vm = psutil.virtual_memory()
    info = DeviceInfo(
        type=dev_type,
        torch_device="cpu",
        name="CPU",
        ram_total_gb=round(vm.total / _GB, 2),
        ram_used_gb=round((vm.total - vm.available) / _GB, 2),
        torch_available=torch is not None,
    )
    if dev_type == "cuda" and torch is not None:
        idx = best_cuda_index()
        if cuda_index is not None:
            if cuda_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA_DEVICE={cuda_index} but only {torch.cuda.device_count()} CUDA device(s) are visible"
                )
            idx = cuda_index
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.torch_device = f"cuda:{idx}"
        info.name = props.name
        info.gpu_count = torch.cuda.device_count()
        info.vram_total_gb = round(total / _GB, 2)
        info.vram_used_gb = round((total - free) / _GB, 2)
        info.bf16_supported = bool(torch.cuda.is_bf16_supported())
    elif dev_type == "mps":
        info.torch_device = "mps"
        info.name = "Apple Metal (MPS)"
        info.gpu_count = 1
    return info


def memory_probe(device: str) -> tuple[float, float]:
    """Return (free_gb, total_gb) for a torch device string. Unified memory/CPU uses system RAM."""
    torch = _import_torch()
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        free, total = torch.cuda.mem_get_info(idx)
        return free / _GB, total / _GB
    vm = psutil.virtual_memory()
    return vm.available / _GB, vm.total / _GB


def empty_cache(device: str | None = None) -> None:
    torch = _import_torch()
    if torch is None:
        return
    try:
        if (device is None or device.startswith("cuda")) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps" and mps_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best-effort
        logger.debug("empty_cache failed", exc_info=True)


def is_oom_error(exc: BaseException) -> bool:
    torch = _import_torch()
    if torch is not None:
        oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def is_cuda_error(exc: BaseException) -> bool:
    text = str(exc)
    return "CUDA" in text or "cuDNN" in text or "CUBLAS" in text


def describe(info: DeviceInfo) -> str:
    lines = [f"Device: {info.type.upper()}"]
    if info.type == "cuda":
        lines.append(f"GPU: {info.name}")
        lines.append(f"VRAM: {info.vram_total_gb} GB")
    elif info.type == "mps":
        lines.append(f"GPU: {info.name}")
    lines.append(f"RAM: {info.ram_total_gb} GB")
    return "\n".join(lines)


def gpu_details(index: int | None = None) -> list[dict[str, Any]]:
    """Per-GPU report for /api/gpu: name, VRAM (driver view and this process' torch allocator), utilization."""
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return []
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - utilization is optional
        pynvml = None  # type: ignore[assignment]
    indices = [index] if index is not None else list(range(torch.cuda.device_count()))
    out: list[dict[str, Any]] = []
    for idx in indices:
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        entry: dict[str, Any] = {
            "index": idx,
            "device": f"cuda:{idx}",
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_total_gb": round(total / _GB, 2),
            "vram_used_gb": round((total - free) / _GB, 2),
            "vram_free_gb": round(free / _GB, 2),
            "torch_allocated_gb": round(torch.cuda.memory_allocated(idx) / _GB, 2),
            "torch_reserved_gb": round(torch.cuda.memory_reserved(idx) / _GB, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "utilization": None,
        }
        if pynvml is not None:
            try:
                # NVML indexes physical GPUs; map through CUDA_VISIBLE_DEVICES when it is a plain index list.
                visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip().isdigit()]
                nvml_idx = int(visible[idx]) if idx < len(visible) else idx
                handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
                entry["utilization"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:  # noqa: BLE001
                pass
        out.append(entry)
    return out


def cuda_version() -> str | None:
    torch = _import_torch()
    return getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
