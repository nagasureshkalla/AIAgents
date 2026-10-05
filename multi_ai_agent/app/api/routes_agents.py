"""Agent introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.schemas.agent import AgentInfo

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("", response_model=list[AgentInfo], summary="List registered agents and their capabilities")
async def list_agents(container: ContainerDep) -> list[AgentInfo]:
    return container.agents.info()


@router.get("/{name}", response_model=AgentInfo, summary="Get one agent")
async def get_agent(name: str, container: ContainerDep) -> AgentInfo:
    return container.agents.get(name).info()
