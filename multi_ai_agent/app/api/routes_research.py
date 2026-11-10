"""Web research endpoints, Server-Sent Events streaming, GPU and local-inference status."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.research_agent import NO_GPU_MESSAGE
from app.api.deps import ContainerDep
from app.core.container import Container
from app.core.device import cuda_version, gpu_details
from app.core.events import EventType, TaskEvent
from app.core.exceptions import InvalidRequestError
from app.core.local_guard import LOCAL_NOTICE
from app.schemas.common import Priority, TaskStatus
from app.schemas.task import TaskOptions, TaskRecord, TaskRequest

router = APIRouter(prefix="/api", tags=["research"])

HEARTBEAT_SECONDS = 15.0


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The question to research on the web.")
    depth: Literal["quick", "normal", "deep"] | None = Field(default=None, description="Defaults to research.default_depth.")
    priority: Priority = Priority.HIGH

    model_config = {"json_schema_extra": {"examples": [{"query": "What are the latest open source AI video models?", "depth": "deep"}]}}


class ResearchCreated(BaseModel):
    research_id: str
    status: TaskStatus
    stream: str
    result: str


class ResearchResponse(BaseModel):
    research_id: str
    status: TaskStatus
    query: str
    mode: str | None = None
    depth: str | None = None
    answer: str | None = None
    confidence: str | None = None
    confidence_reason: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    iterations: int = 0
    failed_sources: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    duration: float | None = None
    created_at: float | None = None


def _research_payload(record: TaskRecord) -> dict[str, Any]:
    for result in record.agent_results.values():
        if result.agent == "research" and result.ok:
            return result.result
    return {}


def _response(record: TaskRecord) -> ResearchResponse:
    payload = _research_payload(record)
    answer = (record.result or {}).get("answer") if record.result else None
    return ResearchResponse(
        research_id=record.task_id,
        status=record.status,
        query=record.request.prompt,
        mode=payload.get("mode"),
        depth=payload.get("depth"),
        answer=payload.get("text") or answer,
        confidence=payload.get("confidence"),
        confidence_reason=payload.get("confidence_reason"),
        sources=payload.get("sources", []),
        claims=payload.get("claims", []),
        queries=payload.get("queries", []),
        iterations=payload.get("iterations", 0),
        failed_sources=payload.get("failed_sources", []),
        error=record.error,
        duration=record.duration,
        created_at=record.created_at,
    )


@router.post("/research", response_model=ResearchCreated, status_code=202, summary="Start a web research task")
async def start_research(body: ResearchRequest, container: ContainerDep) -> ResearchCreated:
    if not container.agents.has("research"):
        raise InvalidRequestError("The research agent is disabled (agents.research.enabled: false)")
    depth = body.depth or container.settings.research.default_depth
    request = TaskRequest(
        prompt=body.query,
        priority=body.priority,
        options=TaskOptions(agents=["research"], skip_validation=True, params={"depth": depth}),
    )
    record = container.submit(request)
    return ResearchCreated(
        research_id=record.task_id,
        status=record.status,
        stream=f"/api/stream/{record.task_id}",
        result=f"/api/research/{record.task_id}",
    )


@router.get("/research", response_model=list[ResearchResponse], summary="Recent research tasks")
async def list_research(container: ContainerDep, limit: int = 30) -> list[ResearchResponse]:
    records = [r for r in container.store.list(None, 500) if r.request.options.agents == ["research"]]
    return [_response(r) for r in records[: max(1, min(limit, 200))]]


@router.get("/research/{research_id}", response_model=ResearchResponse, summary="Research status, answer and citations")
async def get_research(research_id: str, container: ContainerDep) -> ResearchResponse:
    return _response(container.store.get(research_id))


@router.get("/research/{research_id}/sources", summary="Every page the browser visited for this research")
async def research_sources(research_id: str, container: ContainerDep) -> dict[str, Any]:
    payload = _research_payload(container.store.get(research_id))
    return {
        "research_id": research_id,
        "cited": payload.get("cited", []),
        "sources": payload.get("sources", []),
        "failed": payload.get("failed_sources", []),
        "evidence": payload.get("evidence", []),
    }


# ---------------------------------------------------------------- SSE
def to_sse(evt: TaskEvent) -> dict[str, Any] | None:
    """Map internal task events to the research stream format: status | source | answer | error."""
    d = evt.data
    if evt.event == EventType.AGENT_PROGRESS:
        return {"type": "status", "message": d.get("message", ""), **{k: v for k, v in d.items() if k != "message"}}
    if evt.event == EventType.RESEARCH_SOURCE:
        return {"type": "source", **d}
    if evt.event == EventType.TASK_QUEUED:
        return {"type": "status", "message": "Queued", "stage": "queued"}
    if evt.event == EventType.TASK_STARTED:
        return {"type": "status", "message": "Local model loaded on the GPU; starting research", "stage": "start"}
    if evt.event == EventType.AGENT_RETRY:
        return {"type": "status", "message": f"Retrying ({d.get('action', 'retry')})", "stage": "retry"}
    if evt.event == EventType.TASK_COMPLETED:
        result = d.get("result") or {}
        return {"type": "answer", "content": result.get("answer", ""), "duration": d.get("duration")}
    if evt.event in (EventType.TASK_FAILED, EventType.TASK_CANCELLED):
        return {"type": "error", "message": d.get("error") or evt.event.value}
    if evt.event == EventType.AGENT_FAILED:
        return {"type": "status", "message": f"{evt.agent} failed: {d.get('error')}", "stage": "error"}
    return None


def _frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def _stream(container: Container, task_id: str) -> AsyncIterator[str]:
    queue = container.events.subscribe(task_id, replay=True)
    try:
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
            except TimeoutError:
                record = container.store.get(task_id)
                if record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    final = {"type": "answer", "content": (record.result or {}).get("answer", "")} if record.status == TaskStatus.COMPLETED else {"type": "error", "message": record.error or record.status.value}
                    yield _frame(final)
                    return
                yield ": keep-alive\n\n"
                continue
            payload = to_sse(evt)
            if payload is not None:
                yield _frame({"research_id": task_id, "timestamp": evt.timestamp, **payload})
            if evt.is_terminal:
                return
    finally:
        container.events.unsubscribe(task_id, queue)


@router.get("/stream/{task_id}", summary="Server-Sent Events stream of a research (or any) task")
async def stream(task_id: str, container: ContainerDep) -> StreamingResponse:
    container.store.get(task_id)  # 404 for unknown ids
    return StreamingResponse(
        _stream(container, task_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- GPU / status
@router.get("/gpu", summary="NVIDIA GPU, CUDA and VRAM details")
async def gpu(container: ContainerDep) -> dict[str, Any]:
    info = container.device_info
    active = int(info.torch_device.split(":", 1)[1]) if info.torch_device.startswith("cuda:") else None
    gpus = await asyncio.to_thread(gpu_details) if info.type == "cuda" else []
    return {
        "cuda_available": info.type == "cuda",
        "cuda_version": cuda_version(),
        "device": info.torch_device,
        "active_index": active,
        "name": info.name if info.type == "cuda" else None,
        "gpu_count": info.gpu_count,
        "gpu_mode": container.settings.inference.gpu_mode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bf16_supported": info.bf16_supported,
        "gpus": gpus,
        "message": None if info.type == "cuda" else NO_GPU_MESSAGE,
    }


@router.get("/status", summary="Local-only inference status: model, GPU, web mode, privacy")
async def status(container: ContainerDep) -> dict[str, Any]:
    s = container.settings
    info = container.device_info
    vram_used = vram_total = None
    if info.type == "cuda":
        free, total = container.models.memory(info.torch_device)
        vram_used, vram_total = round(total - free, 2), round(total, 2)
    snap = container.monitor.system_snapshot()

    def model_status(key: str) -> dict[str, Any]:
        try:
            resolved = container.models.resolve(key)
            spec = container.model_registry.get(resolved)
            return {"key": key, "resolved": resolved, "path": str(spec.path), "installed": spec.installed,
                    "loaded": container.models.is_loaded(resolved), "engine": spec.resolved_engine(s.inference.engine)}
        except Exception as exc:  # noqa: BLE001 - report instead of failing the status endpoint
            return {"key": key, "resolved": None, "installed": False, "loaded": False, "error": str(exc)}

    web = s.research.web_enabled
    return {
        "ai_inference": "local",
        "ai_mode": s.inference.ai_mode,
        "external_ai_apis": "disabled",
        "notice": LOCAL_NOTICE,
        "cloud_keys_ignored": container.cloud_keys_ignored,
        "mode": "LOCAL AI + WEB" if web else "LOCAL AI + OFFLINE",
        "web_enabled": web,
        "search_provider": s.research.search_provider if web else None,
        "device": info.type,
        "torch_device": info.torch_device,
        "cuda_available": info.type == "cuda",
        "gpu": info.name if info.type == "cuda" else None,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "gpu_utilization": snap.get("gpu_utilization"),
        "research_enabled": container.agents.has("research"),
        "research_model": model_status(s.research.reasoning_model),
        "fast_model": model_status(s.research.fast_model),
        "models_loaded": container.models.loaded_models(),
        "default_depth": s.research.default_depth,
        "max_iterations": s.research.max_iterations,
        "gpu_required": s.research.require_cuda and not s.research.allow_cpu,
        "gpu_message": None if info.type == "cuda" else NO_GPU_MESSAGE,
    }
