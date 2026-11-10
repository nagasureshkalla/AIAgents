"""REST API + static web UI."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.master_agent import TaskManager
from api.websocket import ScreenStreamer, agent_socket
from config.settings import Settings
from core.runtime import Runtime, build_runtime

logger = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / "web"


class TaskRequest(BaseModel):
    command: str = Field(min_length=1, max_length=4000)


class Answer(BaseModel):
    answer: Any


def create_app(settings: Settings, runtime: Runtime | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        rt = runtime or build_runtime(settings)
        state["rt"] = rt
        state["tasks"] = TaskManager(rt)
        state["screen"] = ScreenStreamer(rt)
        state["screen"].start()
        if os.getenv("MODEL_WARMUP", "true").lower() != "false":  # the multi-agent pipeline manages loading itself
            asyncio.create_task(rt.models.warmup())  # load the model into VRAM while the user types
        yield
        await state["screen"].stop()

    app = FastAPI(title="Computer Master Agent", version="1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

    def rt() -> Runtime:
        return state["rt"]

    def tm() -> TaskManager:
        return state["tasks"]

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB / "templates" / "index.html")

    # ---- tasks --------------------------------------------------------------------------------------------
    @app.post("/api/task", status_code=202)
    async def create_task(req: TaskRequest) -> dict[str, Any]:
        return tm().submit(req.command).to_dict()

    @app.get("/api/tasks")
    async def list_tasks() -> list[dict[str, Any]]:
        return [h.to_dict() for h in sorted(tm().tasks.values(), key=lambda h: h.created, reverse=True)]

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        h = tm().get(task_id)
        if h is not None:
            return h.to_dict() | {"record": h.record, "events": rt().bus.recent(task_id)}
        rec = rt().task_memory.load(task_id)
        if rec is None:
            raise HTTPException(404, "task not found")
        return {"task_id": task_id, "status": rec.get("result"), "record": rec, "events": []}

    for verb in ("pause", "resume", "stop"):
        def _make(v: str):  # noqa: ANN202
            async def control(task_id: str) -> dict[str, Any]:
                if not getattr(tm(), v)(task_id):
                    raise HTTPException(409, f"cannot {v} task {task_id}")
                return {"task_id": task_id, "ok": True, "action": v}
            return control
        app.post(f"/api/tasks/{{task_id}}/{verb}", name=f"{verb}_task")(_make(verb))

    @app.get("/api/history")
    async def history(limit: int = 50) -> list[dict[str, Any]]:
        return rt().task_memory.list(limit)

    # ---- human in the loop --------------------------------------------------------------------------------
    @app.get("/api/confirmations")
    async def pending() -> list[dict[str, Any]]:
        return rt().broker.list()

    @app.post("/api/confirmations/{request_id}")
    async def respond(request_id: str, body: Answer) -> dict[str, Any]:
        if not rt().broker.respond(request_id, body.answer):
            raise HTTPException(404, "no such pending request")
        return {"ok": True}

    # ---- system -------------------------------------------------------------------------------------------
    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        r = rt()
        win = await asyncio.to_thread(r.desktop.windows.active_window)
        running = tm().running()
        return {
            "desktop": {"backend": r.desktop.name, "display": r.desktop.display, "accessibility": r.desktop.accessibility,
                        "screen": list(r.desktop.screen.size())},
            "active_window": win.to_dict() if win else None,
            "models": await r.models.health(),
            "model_stats": r.models.stats(),
            "agents": tm().master.agents.list(),
            "plugins": tm().master.agents.plugin_names(),
            "running_task": running.to_dict() if running else None,
            "master_state": tm().master.state.state.value,
            "settings": {"vision_mode": r.settings.vision_mode, "screen_fps": r.settings.screen_fps,
                         "require_confirmation": r.settings.require_confirmation,
                         "max_action_retries": r.settings.max_action_retries, "debug": r.settings.debug},
        }

    @app.get("/api/applications")
    async def applications() -> list[dict[str, Any]]:
        return [a.to_dict() for a in rt().apps.installed()]

    @app.get("/api/agents")
    async def agents() -> list[dict[str, Any]]:
        return tm().master.agents.list()

    @app.get("/api/agents/{desktop_id}/memory")
    async def agent_memory(desktop_id: str) -> dict[str, Any]:
        agent = tm().master.agents.agents.get(desktop_id)
        if agent is None:
            raise HTTPException(404, "no such agent")
        m = agent.memory
        return {"profile": m.profile, "knowledge": m.knowledge, "ui_elements": len(m.ui.get("elements", {})),
                "successes": m.successes[-10:], "failures": m.failures[-10:], "ui_model": agent.ui_model}

    @app.get("/api/screen.jpg")
    async def screen() -> Response:
        return Response(await state["screen"].snapshot_jpeg(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/observe")
    async def observe() -> dict[str, Any]:
        obs = await asyncio.to_thread(rt().detector.observe, False, False)
        return obs.to_dict() | {"timing": rt().detector.last_timing}

    @app.websocket("/ws/agent")
    async def ws_agent(ws: WebSocket) -> None:
        await agent_socket(ws, rt(), tm(), state["screen"])

    return app
