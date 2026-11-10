"""Control Panel endpoints: service status and answers to its approval prompts / questions."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.agents.control_panel_agent import PENDING_REQUESTS, ControlPanelClient
from app.api.deps import ContainerDep

router = APIRouter(prefix="/api/control-panel", tags=["control panel"])


class Answer(BaseModel):
    answer: Any


@router.get("", summary="Control Panel service status")
async def control_panel_status(container: ContainerDep) -> dict[str, Any]:
    cfg = container.settings.control_panel
    client = ControlPanelClient(cfg, container.settings)
    st = await client.status()
    return {
        "enabled": cfg.enabled,
        "url": client.url,
        "running": st is not None,
        "model": ((st or {}).get("models") or {}).get("planner", {}).get("model"),
        "display": (st or {}).get("desktop", {}).get("display"),
        "pending_requests": list(PENDING_REQUESTS),
    }


@router.get("/screen.jpg", summary="Current screen of the controlled desktop (for the dashboard's Screen panel)")
async def control_panel_screen(container: ContainerDep) -> Response:
    # Only the dashboard's view polls this: the agent captures the screen itself, so tasks run with the panel hidden.
    url = ControlPanelClient(container.settings.control_panel, container.settings).url
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{url}/api/screen.jpg")
            r.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"The Control Panel service at {url} is not reachable") from exc
    return Response(r.content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/requests/{request_id}", summary="Approve/deny or answer a Control Panel question")
async def answer_request(request_id: str, body: Answer, container: ContainerDep) -> dict[str, Any]:
    url = PENDING_REQUESTS.get(request_id)
    if url is None:
        raise HTTPException(404, "No pending Control Panel request with that id (already answered or expired)")
    client = ControlPanelClient(container.settings.control_panel.model_copy(update={"url": url}), container.settings)
    if not await client.respond(request_id, body.answer):
        raise HTTPException(409, "The Control Panel no longer waits for that answer")
    PENDING_REQUESTS.pop(request_id, None)
    return {"ok": True}
