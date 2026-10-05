"""Local model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.core.exceptions import ModelNotFoundError
from app.schemas.result import ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])


async def _info(container: ContainerDep, name: str) -> ModelInfo:
    for info in await container.models.list_models():
        if info.name == name:
            return info
    raise ModelNotFoundError(f"Unknown model '{name}'")


@router.get("", response_model=list[ModelInfo], summary="List configured and discovered local models")
async def list_models(container: ContainerDep) -> list[ModelInfo]:
    return await container.models.list_models()


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
