"""Health and performance endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.schemas.result import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


def _gb(value: float | None) -> str | None:
    return None if value is None else f"{value:.1f}GB"


@router.get("/health", response_model=HealthResponse, summary="Service, device and resource health")
async def health(container: ContainerDep) -> HealthResponse:
    info = container.device_info
    snap = container.monitor.system_snapshot()
    vram_total = vram_used = None
    if info.type == "cuda":
        free, total = container.models.memory(info.torch_device)
        vram_total, vram_used = total, total - free
    counts = container.store.counts()
    return HealthResponse(
        status="healthy",
        device=info.type,
        gpu=info.name if info.is_gpu else None,
        vram_total=_gb(vram_total),
        vram_used=_gb(vram_used),
        ram_total=_gb(snap["ram_total_gb"]),
        ram_used=_gb(snap["ram_used_gb"]),
        cpu_percent=snap["cpu_percent"],
        gpu_utilization=snap["gpu_utilization"],
        agents_active=container.orchestrator.active_agents,
        tasks_running=counts["running"],
        tasks_queued=counts["queued"],
        models_loaded=container.models.loaded_models(),
        offline_mode=container.settings.runtime.offline_mode,
        uptime_seconds=round(time.time() - container.started_at, 1),
    )


@router.get("/metrics", summary="Performance summary and recent inference/agent records")
async def metrics(
    container: ContainerDep,
    limit: int = Query(default=50, ge=1, le=1000),
    task_id: str | None = None,
) -> dict:
    return {
        "summary": container.monitor.summary(),
        "scheduler": container.scheduler.stats(),
        "tasks": container.store.counts(),
        "recent": container.monitor.recent(limit, task_id),
    }


@router.get("/device", summary="Detected compute device")
async def device(container: ContainerDep) -> dict:
    return container.device_info.to_dict()
