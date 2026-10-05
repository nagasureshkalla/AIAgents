"""Inference / agent performance metrics and system resource snapshots."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import psutil

logger = logging.getLogger(__name__)
_GB = 1024**3


@dataclass
class MetricRecord:
    kind: str  # "inference" | "agent"
    task_id: str | None
    agent: str | None
    model: str | None
    device: str | None
    start_time: float
    end_time: float
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    cpu_percent: float | None = None
    ram_used_gb: float | None = None
    gpu_utilization: float | None = None
    vram_used_gb: float | None = None
    status: str = "ok"
    error: str | None = None


class _GpuProbe:
    """NVML-based GPU utilization (optional: pip install nvidia-ml-py)."""

    def __init__(self) -> None:
        self._nvml: Any = None
        self._lock = threading.Lock()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:  # noqa: BLE001 - NVML missing or no NVIDIA driver
            self._nvml = None

    def utilization(self) -> tuple[float | None, float | None]:
        """(gpu_util_percent, vram_used_gb) summed/averaged over visible GPUs."""
        if self._nvml is not None:
            try:
                with self._lock:
                    count = self._nvml.nvmlDeviceGetCount()
                    utils, used = [], 0.0
                    for i in range(count):
                        handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                        utils.append(float(self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                        used += self._nvml.nvmlDeviceGetMemoryInfo(handle).used / _GB
                return (round(mean(utils), 1) if utils else None), round(used, 2)
            except Exception:  # noqa: BLE001
                return None, None
        try:
            import torch

            if torch.cuda.is_available():
                used = 0
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    used += total - free
                return None, round(used / _GB, 2)
        except Exception:  # noqa: BLE001
            pass
        return None, None


class PerformanceMonitor:
    def __init__(self, max_records: int = 2000) -> None:
        self._records: deque[MetricRecord] = deque(maxlen=max_records)
        self._gpu = _GpuProbe()
        psutil.cpu_percent(interval=None)  # prime the non-blocking CPU counter

    def system_snapshot(self) -> dict[str, float | None]:
        vm = psutil.virtual_memory()
        gpu_util, vram_used = self._gpu.utilization()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round((vm.total - vm.available) / _GB, 2),
            "ram_total_gb": round(vm.total / _GB, 2),
            "gpu_utilization": gpu_util,
            "vram_used_gb": vram_used,
        }

    def record(
        self,
        *,
        kind: str,
        start_time: float,
        end_time: float,
        task_id: str | None = None,
        agent: str | None = None,
        model: str | None = None,
        device: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> MetricRecord:
        latency = max(0.0, end_time - start_time)
        snap = self.system_snapshot()
        rec = MetricRecord(
            kind=kind,
            task_id=task_id,
            agent=agent,
            model=model,
            device=device,
            start_time=start_time,
            end_time=end_time,
            latency=round(latency, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(completion_tokens / latency, 2) if latency > 0 and completion_tokens else 0.0,
            cpu_percent=snap["cpu_percent"],
            ram_used_gb=snap["ram_used_gb"],
            gpu_utilization=snap["gpu_utilization"],
            vram_used_gb=snap["vram_used_gb"],
            status=status,
            error=error,
        )
        self._records.append(rec)
        return rec

    def recent(self, limit: int = 50, task_id: str | None = None) -> list[dict[str, Any]]:
        records = [r for r in self._records if task_id is None or r.task_id == task_id]
        return [asdict(r) for r in records[-limit:]]

    def summary(self) -> dict[str, Any]:
        by_key: dict[tuple[str, str], list[MetricRecord]] = {}
        for rec in self._records:
            name = rec.agent if rec.kind == "agent" else (rec.model or "unknown")
            by_key.setdefault((rec.kind, name or "unknown"), []).append(rec)
        out: dict[str, Any] = {"agents": {}, "models": {}}
        for (kind, name), recs in by_key.items():
            ok = [r for r in recs if r.status == "ok"]
            tps = [r.tokens_per_second for r in ok if r.tokens_per_second]
            out["agents" if kind == "agent" else "models"][name] = {
                "calls": len(recs),
                "errors": len(recs) - len(ok),
                "avg_latency": round(mean(r.latency for r in recs), 3),
                "max_latency": round(max(r.latency for r in recs), 3),
                "avg_tokens_per_second": round(mean(tps), 2) if tps else None,
                "total_completion_tokens": sum(r.completion_tokens for r in recs),
            }
        out["records"] = len(self._records)
        out["generated_at"] = time.time()
        return out
