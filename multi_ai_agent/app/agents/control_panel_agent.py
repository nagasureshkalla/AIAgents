"""Control Panel agent: operates desktop applications (open apps, click, type, browse...) by delegating to the
computer_ai_agent service, which observes the screen, acts one step at a time and verifies every step.

Model lifecycle stays here: the agent's model (an Ollama vision model, `models.control_panel` in config.yaml) is
loaded through the ModelManager and held resident for the whole desktop task, so the manager can make room for it
by unloading other models and can unload it again when other agents need the GPU.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from app.agents.base_agent import BaseAgent, _models_used
from app.core.config import ControlPanelConfig, Settings
from app.core.exceptions import AgentSystemError
from app.models.engines.ollama_engine import ollama_num_ctx
from app.schemas.agent import AgentTask

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class ControlPanelError(AgentSystemError):
    code = "control_panel_failed"
    error_type = "control_failed"  # never retried automatically: repeating GUI actions is unsafe


# Pending approvals/questions from the Control Panel: request id -> service URL (answered via /api/control-panel).
PENDING_REQUESTS: dict[str, str] = {}


class ControlPanelClient:
    """Starts (if needed) and talks to the computer_ai_agent service over REST + WebSocket."""

    _proc: subprocess.Popen | None = None  # one service per process, shared by all tasks

    def __init__(self, cfg: ControlPanelConfig, settings: Settings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.url = cfg.url.rstrip("/")

    @property
    def project_dir(self) -> Path:
        return self.settings.resolve_path(self.cfg.project_dir)

    async def status(self) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.url}/api/status")
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def ensure_running(self, model_tag: str, num_ctx: int = 16384) -> dict[str, Any]:
        st = await self.status()
        if st is None:
            if not self.cfg.autostart:
                raise ControlPanelError(f"The Control Panel service is not running at {self.url}. Start it with: "
                                        f"cd {self.project_dir} && .venv/bin/python app.py")
            self._start(model_tag, num_ctx)
            deadline = time.monotonic() + self.cfg.startup_timeout
            while st is None and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                if ControlPanelClient._proc is not None and ControlPanelClient._proc.poll() is not None:
                    raise ControlPanelError(
                        f"The Control Panel service exited during startup; see {self._log_path()}")
                st = await self.status()
            if st is None:
                raise ControlPanelError(f"The Control Panel service did not start within {self.cfg.startup_timeout:.0f}s; "
                                        f"see {self._log_path()}")
        served = ((st.get("models") or {}).get("planner") or {}).get("model", "")
        if served and model_tag and served.split(":latest")[0] != model_tag.split(":latest")[0]:
            raise ControlPanelError(
                f"The Control Panel service at {self.url} uses model {served!r}, but this pipeline manages "
                f"{model_tag!r}. Restart it with MODEL_NAME={model_tag}, or set models.control_panel.ollama_model.")
        return st

    def _log_path(self) -> Path:
        return self.settings.cache_dir / "control_panel_service.log"

    def _start(self, model_tag: str, num_ctx: int) -> None:
        project = self.project_dir
        python = project / ".venv" / "bin" / "python"
        if not (project / "app.py").is_file() or not python.is_file():
            raise ControlPanelError(f"computer_ai_agent not found at {project} (expected app.py and .venv). "
                                    "Set control_panel.project_dir in config.yaml.")
        from urllib.parse import urlparse

        port = urlparse(self.url).port or 8765
        cmd = [str(python), "app.py", "--port", str(port)]
        if self.cfg.display:
            cmd += ["--display", self.cfg.display]
        env = {
            **os.environ,
            "MODEL_PROVIDER": "ollama",
            "MODEL_NAME": model_tag,
            "MAX_CONTEXT": str(num_ctx),     # same context as the ModelManager loaded it with (no Ollama reload)
            "OLLAMA_URL": self.settings.inference.ollama_url,
            "MODEL_KEEP_ALIVE": "-1m",       # the pipeline's ModelManager decides when the model leaves the GPU
            "MODEL_WARMUP": "false",         # ...and when it is loaded
            "REQUIRE_CONFIRMATION": "true" if self.cfg.require_confirmation else "false",
        }
        self._log_path().parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path(), "ab")
        ControlPanelClient._proc = subprocess.Popen(
            cmd, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        atexit.register(ControlPanelClient.shutdown)
        logger.info("started Control Panel service pid=%s: %s", ControlPanelClient._proc.pid, " ".join(cmd))

    async def stop_service(self) -> None:
        """Stop the service and free its port: the one this process started, or one started by hand from
        control_panel.project_dir (checked by its command line and working directory, so nothing else is touched)."""
        if ControlPanelClient._proc is not None:
            await asyncio.to_thread(ControlPanelClient.shutdown)
            logger.info("stopped the Control Panel service after the desktop task")
            return
        await asyncio.to_thread(self._stop_external)

    def _stop_external(self) -> None:
        from urllib.parse import urlparse

        import psutil

        port = urlparse(self.url).port or 8765
        project = self.project_dir.resolve()
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != psutil.CONN_LISTEN or conn.laddr.port != port or conn.pid is None:
                continue
            try:
                proc = psutil.Process(conn.pid)
                if "app.py" not in proc.cmdline() or Path(proc.cwd()).resolve() != project:
                    logger.warning("port %s is held by pid %s, which is not computer_ai_agent; leaving it", port, conn.pid)
                    return
                proc.terminate()
                proc.wait(5)
                logger.info("stopped the Control Panel service (pid %s) after the desktop task", conn.pid)
            except psutil.TimeoutExpired:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                logger.warning("could not stop the Control Panel service on port %s: %s", port, exc)
            return

    @staticmethod
    def shutdown() -> None:
        proc = ControlPanelClient._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(5)
        ControlPanelClient._proc = None

    async def run(self, command: str, on_event: EventHandler) -> dict[str, Any]:
        """Submit the command and stream its events until it finishes. Returns the terminal event."""
        import websockets

        ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/agent"
        cp_task: str | None = None
        async with websockets.connect(ws_url, max_size=32 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(json.dumps({"type": "command", "command": command}))
            try:
                async for raw in ws:
                    e = json.loads(raw)
                    ev = e.get("event")
                    if ev == "task_accepted" and cp_task is None:
                        cp_task = e["task_id"]
                        await on_event(e)
                        continue
                    if cp_task is None or e.get("task_id") != cp_task:
                        continue  # hello/history, frames and other tasks
                    await on_event(e)
                    if ev in ("task_completed", "task_failed", "task_stopped"):
                        return e
            except asyncio.CancelledError:
                if cp_task:
                    with contextlib.suppress(Exception):
                        await ws.send(json.dumps({"type": "stop", "task_id": cp_task}))
                        await asyncio.sleep(0.2)
                raise
        raise ControlPanelError("Lost the connection to the Control Panel service before the task finished")

    async def respond(self, request_id: str, answer: Any) -> bool:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(f"{self.url}/api/confirmations/{request_id}", json={"answer": answer})
        return r.status_code == 200


class ControlPanelAgent(BaseAgent):
    name = "control_panel"
    description = (
        "Control Panel: operates applications on this computer (Chrome, VS Code, Terminal, Files, Calculator, any app): "
        "opens them, clicks, types and verifies every step on screen. Asks before destructive actions."
    )
    capabilities = ["computer_control", "gui_automation", "app_automation"]
    keywords: list[str] = []
    default_model = "control_panel"

    client_factory: Callable[[ControlPanelConfig, Settings], ControlPanelClient] = ControlPanelClient
    _active = 0  # desktop tasks in flight in this process; the service is stopped only when the last one ends

    async def run(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.ctx.settings.control_panel
        ControlPanelAgent._active += 1
        try:
            return await self._run(task)
        finally:
            ControlPanelAgent._active -= 1
            if cfg.enabled and cfg.stop_after_task and ControlPanelAgent._active == 0:
                with contextlib.suppress(Exception):  # never let cleanup replace the task's own result or error
                    await asyncio.shield(self.client_factory(cfg, self.ctx.settings).stop_service())

    async def _run(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.ctx.settings.control_panel
        if not cfg.enabled:
            raise ControlPanelError("The Control Panel agent is disabled (control_panel.enabled: false)")
        key = self.model_key
        if not key:
            raise ControlPanelError("No model configured for the Control Panel agent (agents.control_panel.model)")
        models = self.ctx.models
        resolved = models.resolve_required(key)
        spec = models.registry.get(resolved)
        if not spec.is_ollama:
            raise ControlPanelError(f"The Control Panel needs an Ollama vision model (engine: ollama); "
                                    f"'{resolved}' uses {spec.engine}")
        client = self.client_factory(cfg, self.ctx.settings)

        if not models.is_loaded(resolved):
            self.progress(task, f"Loading model {resolved} ({spec.ollama_model})", model=resolved)
        # Hold the model for the whole desktop task: while active it cannot be evicted by other agents' loads.
        async with models.use(resolved):
            used = _models_used.get()
            if used is not None and resolved not in used:
                used.append(resolved)
            self.progress(task, "Connecting to the Control Panel", panel_url=client.url)
            await client.ensure_running(spec.ollama_model or "", ollama_num_ctx(spec, self.ctx.settings))
            actions: list[dict[str, Any]] = []
            started = time.time()

            async def forward(e: dict[str, Any]) -> None:
                msg = self._describe(e, actions)
                if e.get("event") in ("confirmation_required", "user_input_required"):
                    PENDING_REQUESTS[e["request"]["id"]] = client.url
                elif e.get("event") == "user_response":
                    PENDING_REQUESTS.pop(e.get("request_id", ""), None)
                if msg is not None:
                    self.progress(task, msg, control=self._control_payload(e), panel_url=client.url)

            final = await client.run(task.prompt, forward)

        result = final.get("result") or ""
        if final.get("event") != "task_completed":
            what = "stopped" if final.get("event") == "task_stopped" else "failed"
            raise ControlPanelError(f"Desktop task {what}: {result or 'no details'}")
        return {
            "text": result,
            "actions": actions,
            "action_count": final.get("actions", len(actions)),
            "duration": final.get("duration", round(time.time() - started, 1)),
            "control_task_id": final.get("task_id"),
            "panel_url": client.url,
            "model": resolved,
        }

    @staticmethod
    def _describe(e: dict[str, Any], actions: list[dict[str, Any]]) -> str | None:
        ev = e.get("event")
        if ev == "plan_created":
            return "Plan: " + " → ".join(f"[{s['app']}] {s['instruction'] or 'open'}" for s in e.get("subtasks", []))
        if ev == "application_opened":
            return f"{'Opened' if e.get('launched') else 'Switched to'} {e.get('application')}"
        if ev == "agent_created":
            return f"Created {e.get('agent')}"
        if ev == "action_started":
            return f"▶ {str(e.get('label', '')).replace(chr(10), '⏎')}"
        if ev == "action_completed":
            actions.append({"step": e.get("step"), "label": e.get("label"), "ok": e.get("ok"),
                            "message": e.get("message"), "agent": e.get("agent"), "duration_ms": e.get("duration_ms")})
            return f"{'✓' if e.get('ok') else '✗'} {str(e.get('label', '')).replace(chr(10), '⏎')} — {e.get('message', '')}"
        if ev == "recovery":
            return f"↻ {e.get('reason', '')}"
        if ev == "verification_completed" and e.get("what") == "goal":
            return f"Goal check: {'complete' if e.get('ok') else 'not complete'} — {e.get('detail', '')}"
        if ev in ("confirmation_required", "user_input_required"):
            return "Waiting for you: " + e["request"]["message"]
        if ev == "user_response":
            return f"Answered: {e.get('answer')}"
        if ev in ("task_paused", "task_resumed"):
            return ev.replace("task_", "").capitalize()
        return None

    @staticmethod
    def _control_payload(e: dict[str, Any]) -> dict[str, Any]:
        ev = e.get("event")
        out: dict[str, Any] = {"event": ev}
        if ev in ("confirmation_required", "user_input_required"):
            out["request"] = e["request"]  # includes a screenshot (base64 JPEG) for the approval prompt
        elif ev == "user_response":
            out["request_id"] = e.get("request_id")
        elif ev in ("action_completed", "action_started"):
            out.update({k: e.get(k) for k in ("step", "label", "ok", "agent")})
        return out
