"""Agent introspection endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter

from app.agents.manager_agent import (
    BIG_MODEL_PHRASES,
    CONTROL_PHRASES,
    JOB_PHRASES,
    IMAGE_INTENT_RULES,
    INTENT_CAPABILITIES,
    RESEARCH_PHRASES,
    TEXT_INTENT_RULES,
    WEB_TEST_PHRASES,
)
from app.api.deps import ContainerDep
from app.schemas.agent import AgentInfo

router = APIRouter(prefix="/api/agents", tags=["agents"])
routing_router = APIRouter(prefix="/api", tags=["agents"])


def _phrases(pattern: re.Pattern[str]) -> list[str]:
    """Turn a routing regex such as r"\b(summari[sz]e|tl;?dr)\b" into readable phrases: ["summarize", "tl;dr"]."""
    body = pattern.pattern.replace(r"\b", "")
    if body.startswith("\\?|"):  # visual_qa also matches any question mark
        body = body[3:]
        extra = ["?"]
    else:
        extra = []
    body = body[1:-1] if body.startswith("(") and body.endswith(")") else body
    out = []
    for alt in re.split(r"\|(?![^(]*\))", body):
        alt = re.sub(r"\(\?:([^|)]*)(?:\|[^)]*)?\)\??", r"\1", alt)  # (?:'s| is) -> 's ; (?:it |this )? -> it
        alt = re.sub(r"\[(\w)\w*\]", lambda m: "z" if m.group(1) in "sz" else m.group(1), alt)  # [sz] -> z
        alt = re.sub(r"(\S)\?", r"\1", alt)  # tl;?dr -> tl;dr
        out.append(re.sub(r"\s+", " ", alt).strip())
    return extra + out


@routing_router.get("/routing", summary="How the manager routes requests: intents, trigger phrases and agents")
async def routing(container: ContainerDep) -> list[dict]:
    rules = [(intent, pat, "image") for intent, pat in IMAGE_INTENT_RULES]
    rules += [(intent, pat, "text") for intent, pat in TEXT_INTENT_RULES]
    rules.append(("generate", None, "text"))  # fallback when no text rule matches
    out = []
    for intent, pattern, input_type in rules:
        capabilities = INTENT_CAPABILITIES[intent]
        agents = []
        for cap in capabilities:
            agent = container.agents.first_for(cap)
            if agent and agent.name not in agents:
                agents.append(agent.name)
        out.append({
            "intent": intent,
            "input": input_type,
            "phrases": (RESEARCH_PHRASES if intent == "research" else CONTROL_PHRASES if intent == "computer_control"
                        else WEB_TEST_PHRASES if intent == "web_test" else JOB_PHRASES if intent == "job_search"
                        else BIG_MODEL_PHRASES if intent == "big_model"
                        else _phrases(pattern) if pattern else []),
            "capabilities": capabilities,
            "agents": agents,
        })
    return out


@router.get("", response_model=list[AgentInfo], summary="List registered agents and their capabilities")
async def list_agents(container: ContainerDep) -> list[AgentInfo]:
    return container.agents.info()


@router.get("/{name}", response_model=AgentInfo, summary="Get one agent")
async def get_agent(name: str, container: ContainerDep) -> AgentInfo:
    return container.agents.get(name).info()
