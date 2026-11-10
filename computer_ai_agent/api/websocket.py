"""/ws/agent: streams agent events plus live screen frames; accepts control messages from the UI."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect

from vision.screen_capture import to_jpeg

if TYPE_CHECKING:
    from agents.master_agent import TaskManager
    from core.runtime import Runtime

logger = logging.getLogger(__name__)


class ScreenStreamer:
    """Publishes `screen_updated` at SCREEN_FPS while at least one viewer is connected."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.viewers = 0
        self.last_jpeg: bytes | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _frame(self) -> dict[str, Any]:
        img = self.rt.desktop.screen.grab()
        jpeg = to_jpeg(img, 1280, 68)
        self.last_jpeg = jpeg
        win = self.rt.desktop.windows.active_window()
        try:
            cursor = list(self.rt.desktop.mouse.position)
        except Exception:  # noqa: BLE001
            cursor = None
        return {"image": base64.b64encode(jpeg).decode(), "width": img.width, "height": img.height,
                "active_window": win.to_dict() if win else None, "cursor": cursor}

    async def _loop(self) -> None:
        while True:
            fps = max(0.2, min(self.rt.settings.screen_fps, 10.0))
            if self.viewers > 0:
                try:
                    frame = await asyncio.to_thread(self._frame)
                    self.rt.bus.publish("screen_updated", **frame)
                except Exception as exc:  # noqa: BLE001 - display gone for a moment; keep the loop alive
                    logger.debug("screen frame failed: %s", exc)
            await asyncio.sleep(1.0 / fps)

    async def snapshot_jpeg(self) -> bytes:
        await asyncio.to_thread(self._frame)
        assert self.last_jpeg is not None
        return self.last_jpeg


async def agent_socket(ws: WebSocket, rt: Runtime, tasks: TaskManager, streamer: ScreenStreamer) -> None:
    await ws.accept()
    q = rt.bus.subscribe(maxsize=400)
    streamer.viewers += 1
    try:
        await ws.send_json({"event": "hello", "history": list(rt.bus.recent(limit=400)),
                            "pending": rt.broker.list()})

        async def pump() -> None:
            while True:
                await ws.send_json(await q.get())

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "respond":
                    rt.broker.respond(str(msg.get("id")), msg.get("answer"))
                elif kind in ("pause", "resume", "stop") and msg.get("task_id"):
                    getattr(tasks, kind)(str(msg["task_id"]))
                elif kind == "command" and msg.get("command"):
                    h = tasks.submit(str(msg["command"]))
                    await ws.send_json({"event": "task_accepted", "task_id": h.task_id, "command": h.command})
        finally:
            sender.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        streamer.viewers -= 1
        rt.bus.unsubscribe(q)
