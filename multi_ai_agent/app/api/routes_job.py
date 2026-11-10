"""Job Agent endpoints: Auto-Apply backend status, answers to its per-application approval prompts, and the
same-origin proxy (/job-api) used by the Auto-Apply web app that the Control Room serves at /job/."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.agents.job_agent import PENDING_APPROVALS, AutoApplyClient, JobAgent
from app.api.deps import ContainerDep
from app.core.exceptions import AgentSystemError

router = APIRouter(prefix="/api/job-agent", tags=["job agent"])
proxy_router = APIRouter(prefix="/job-api", include_in_schema=False)

# Hop-by-hop headers (and ones httpx recomputes) that must not be copied across the proxy.
_SKIP_REQUEST = {"host", "content-length", "connection", "accept-encoding", "origin", "referer", "cookie"}
_SKIP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection"}
_start_lock = asyncio.Lock()


class Answer(BaseModel):
    answer: Any


@router.get("", summary="Job Agent / Auto-Apply backend status")
async def job_agent_status(container: ContainerDep) -> dict[str, Any]:
    cfg = container.settings.job_agent
    client = AutoApplyClient(cfg, container.settings)
    return {"enabled": cfg.enabled, "url": client.url, "web_url": cfg.web_url, "running": await client.healthy(),
            "pending_requests": list(PENDING_APPROVALS)}


@router.post("/requests/{request_id}", summary="Answer a Job Agent prompt (approve / deny, a choice or typed text)")
async def answer_request(request_id: str, body: Answer) -> dict[str, Any]:
    fut = PENDING_APPROVALS.get(request_id)
    if fut is None or fut.done():
        raise HTTPException(404, "No pending Job Agent request with that id (already answered or expired)")
    if body.answer == "approve":
        fut.set_result("approve")
    elif isinstance(body.answer, str) and body.answer.strip():
        fut.set_result(body.answer.strip()[:2000])  # a choice or a typed answer to a form question
    else:
        fut.set_result("deny")
    return {"ok": True}


@proxy_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request, container: ContainerDep) -> Response:
    """Forward the web app's /job-api/<path> calls to the local Auto-Apply backend, starting it on first use."""
    cfg = container.settings.job_agent
    client = AutoApplyClient(cfg, container.settings)
    if not await client.healthy():
        agent = container.agents.get("job_agent")
        if not isinstance(agent, JobAgent):
            raise HTTPException(503, "The Job Agent is not registered")
        async with _start_lock:
            try:
                await agent.ensure_backend()
            except AgentSystemError as exc:
                raise HTTPException(503, str(exc)) from exc
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQUEST}
    try:
        async with httpx.AsyncClient(timeout=cfg.request_timeout) as c:
            r = await c.request(request.method, f"{client.url}/{path}", params=request.query_params,
                                headers=headers, content=await request.body())
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Auto-Apply backend request failed: {exc}") from exc
    out = {k: v for k, v in r.headers.items() if k.lower() not in _SKIP_RESPONSE}
    return Response(r.content, status_code=r.status_code, headers=out)
