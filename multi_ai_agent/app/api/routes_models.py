"""Local model management endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.deps import ContainerDep
from app.core.exceptions import ModelNotFoundError
from app.schemas.result import ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelAction(BaseModel):
    name: str
    device: str | None = Field(default=None, pattern=r"^(cpu|mps|cuda(:\d+)?)$")
    force: bool = False


class DownloadRequest(BaseModel):
    hf_id: str = Field(..., description="Hugging Face repo id, e.g. Qwen/Qwen2.5-7B-Instruct")
    type: Literal["text", "vision", "multimodal"] = "text"
    name: str | None = Field(default=None, description="Folder name under models/<type>/")


async def _info(container: ContainerDep, name: str) -> ModelInfo:
    for info in await container.models.list_models():
        if info.name == name:
            return info
    raise ModelNotFoundError(f"Unknown model '{name}'")


@router.get("", response_model=list[ModelInfo], summary="List configured and discovered local models")
async def list_models(container: ContainerDep) -> list[ModelInfo]:
    return await container.models.list_models()


@router.post("/load", response_model=ModelInfo, summary="Load a model onto the GPU (body: {name})")
async def load_model_body(body: ModelAction, container: ContainerDep) -> ModelInfo:
    return await load_model(body.name, container, body.device)


@router.post("/unload", summary="Unload a model and free its VRAM (body: {name})")
async def unload_model_body(body: ModelAction, container: ContainerDep) -> dict:
    return await unload_model(body.name, container, body.force)


@router.post("/download", summary="Download a model from the Hugging Face Hub into models/ (no remote inference)")
async def download_model(body: DownloadRequest, container: ContainerDep) -> dict:
    assert container.downloads is not None
    return container.downloads.start(body.hf_id, body.type, body.name).to_dict()


@router.get("/downloads", summary="Model download jobs and their progress")
async def list_downloads(container: ContainerDep) -> list[dict]:
    return container.downloads.list() if container.downloads else []


@router.post("/refresh", summary="Re-scan models/ for newly added model folders")
async def refresh_models(container: ContainerDep) -> dict:
    return {"discovered": container.models.refresh()}


@router.get("/{name}", response_model=ModelInfo, summary="Get one model's status")
async def get_model(name: str, container: ContainerDep) -> ModelInfo:
    return await _info(container, name)


@router.post("/{name}/load", response_model=ModelInfo, summary="Load a model into memory")
async def load_model(
    name: str,
    container: ContainerDep,
    device: str | None = Query(default=None, pattern=r"^(cpu|mps|cuda(:\d+)?)$"),
) -> ModelInfo:
    container.model_registry.get(name)
    await container.models.load_model(name, device)
    return await _info(container, name if device is None or device == container.device_info.torch_device else f"{name}@{device}")


@router.post("/{name}/unload", summary="Unload a model and free its memory")
async def unload_model(name: str, container: ContainerDep, force: bool = False) -> dict:
    container.model_registry.get(name)
    unloaded = await container.models.unload_model(name, force=force)
    return {"model": name, "unloaded": unloaded}
