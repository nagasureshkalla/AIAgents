"""Computer Master Agent: understands the command, picks/launches applications, delegates to sub-agents,
monitors them, recovers from launch failures and reports a concise result."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.agent_manager import AgentManager
from agents.application_agent import TaskContext
from applications.application_manager import ApplicationNotFound, LaunchFailed
from core.logging_setup import TaskDebugRecorder
from core.state_machine import AgentState, StateMachine
from core.types import ActionResult, AgentMessage
from safety.confirmation import TaskControl, TaskStopped

if TYPE_CHECKING:
    from core.runtime import Runtime
    from planner.task_planner import Subtask

logger = logging.getLogger(__name__)
S = AgentState
MASTER = "MasterAgent"


@dataclass
class TaskHandle:
    task_id: str
    command: str
    control: TaskControl
    status: str = "queued"           # queued | running | completed | failed | stopped
    result: str = ""
    created: float = field(default_factory=time.time)
    finished: float | None = None
    future: asyncio.Task | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "command": self.command, "status": self.status, "result": self.result,
                "created": self.created, "finished": self.finished, "paused": self.control.paused,
                "subtasks": self.record.get("subtasks", []), "actions": len(self.record.get("actions", []))}


class MasterAgent:
    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime
        self.agents = AgentManager(runtime)
        self.state = StateMachine(MASTER, runtime.bus)
        self.current_task: str | None = None
        # sub-agents may open, switch to or close other applications through the action engine
        runtime.engine.open_app = self._open_app
        runtime.engine.switch_app = self._switch_app
        runtime.engine.close_app = self._close_app

    async def _open_app(self, name: str) -> ActionResult:
        try:
            entry = self.rt.apps.find(name)
            args, env = self.agents.launch_options(entry)
            entry, win, launched = await self.rt.apps.aensure_open(entry, args, env)
            return ActionResult(True, f"{'launched' if launched else 'switched to'} {entry.name}: {win.title}")
        except (ApplicationNotFound, LaunchFailed) as exc:
            return ActionResult(False, str(exc))

    async def _switch_app(self, name: str) -> ActionResult:
        try:
            entry = self.rt.apps.find(name)
            if not await asyncio.to_thread(self.rt.apps.app_windows, entry):
                return await self._open_app(name)
            win = await self.rt.apps.aactivate(entry)
            return ActionResult(True, f"switched to {entry.name}: {win.title}")
        except ApplicationNotFound as exc:
            return ActionResult(False, str(exc))

    async def _close_app(self, name: str) -> ActionResult:
        try:
            n = await self.rt.apps.aclose(name)
            return ActionResult(n > 0, f"closed {n} window(s) of {name}" if n else f"{name} has no open window")
        except ApplicationNotFound as exc:
            return ActionResult(False, str(exc))

    def _send(self, ctx: TaskContext, msg: AgentMessage) -> None:
        d = msg.to_dict()
        ctx.record["messages"].append(d)
        self.rt.bus.publish("agent_message", task_id=ctx.task_id, message=d)
        ctx.recorder.record("message", **d)

    async def run(self, handle: TaskHandle) -> dict[str, Any]:
        rt = self.rt
        t0 = time.time()
        record = rt.task_memory.new_record(handle.task_id, handle.command)
        handle.record = record
        ctx = TaskContext(handle.task_id, handle.command, handle.control,
                          TaskDebugRecorder(rt.settings.logs_dir, handle.task_id, rt.settings.debug), record)
        self.current_task = handle.task_id
        self.state = StateMachine(MASTER, rt.bus, handle.task_id)
        rt.bus.publish("task_started", task_id=handle.task_id, command=handle.command)
        ctx.recorder.record("task_started", command=handle.command)
        outcome, summary = "failed", ""
        try:
            # ---------- UNDERSTANDING ----------
            self.state.to(S.UNDERSTANDING, handle.command)
            installed = [a.name for a in rt.apps.installed()]
            similar = rt.task_memory.similar_successes(handle.command)
            subtasks, source = await rt.task_planner.plan(handle.command, installed, similar)
            record["subtasks"] = [s.to_dict() for s in subtasks]
            record["plan_source"] = source
            rt.bus.publish("plan_created", task_id=handle.task_id, source=source,
                           subtasks=[s.to_dict() for s in subtasks])
            ctx.recorder.record("plan", source=source, subtasks=record["subtasks"])

            results: list[dict[str, Any]] = []
            ctx.shared["results"] = results
            for sub in subtasks:
                await handle.control.checkpoint()
                res = await self._run_subtask(ctx, sub)
                results.append(res)
                if res["status"] != "completed":
                    outcome = "stopped" if res["status"] == "stopped" else "failed"
                    summary = f"[{res['app']}] {res['result']}"
                    break
            else:
                outcome = "success"
                summary = self._summarize(results)
            if outcome == "success":
                self.state.to(S.COMPLETED, summary)
            else:
                self.state.force(S.FAILED, summary)
        except TaskStopped:
            outcome, summary = "stopped", "Stopped by the user"
            self.state.force(S.FAILED, summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s crashed", handle.task_id)
            outcome, summary = "failed", f"{type(exc).__name__}: {exc}"
            self.state.force(S.FAILED, summary)
        finally:
            self.current_task = None

        duration = round(time.time() - t0, 1)
        record.update(result=outcome, summary=summary, duration=duration, finished_at=time.time(),
                      application=", ".join(dict.fromkeys(s["app"] for s in record["subtasks"])) or None)
        rt.task_memory.save(record)
        ctx.recorder.record("task_finished", result=outcome, summary=summary, duration=duration)
        event = {"success": "task_completed", "stopped": "task_stopped"}.get(outcome, "task_failed")
        rt.bus.publish(event, task_id=handle.task_id, result=summary, duration=duration,
                       actions=len(record["actions"]))
        return record

    async def _run_subtask(self, ctx: TaskContext, sub: Subtask) -> dict[str, Any]:
        rt = self.rt
        rt.bus.publish("subtask_started", task_id=ctx.task_id, subtask=sub.to_dict())

        # ---------- APPLICATION_DETECTION ----------
        if self.state.state not in (S.UNDERSTANDING, S.COMPLETED, S.FAILED, S.WAITING_FOR_USER):
            self.state.force(S.UNDERSTANDING)
        self.state.to(S.APPLICATION_DETECTION, sub.app)
        app_name = sub.app
        entry = None
        for _attempt in range(2):
            try:
                entry = rt.apps.find(app_name)
                break
            except ApplicationNotFound:
                candidates = [a.name for a in rt.apps.installed()
                              if app_name.lower()[:3] in a.name.lower()][:6]
                self.state.to(S.WAITING_FOR_USER, f"{app_name} not installed")
                answer = await rt.broker.choose(
                    ctx.task_id, f"I can't find an application called \"{app_name}\". Which one should I use?",
                    candidates + ["Cancel"] if candidates else [], MASTER)
                if not answer or answer == "Cancel":
                    return self._sub_result(sub, "failed", f"Application not found: {app_name}")
                app_name = answer
                self.state.to(S.APPLICATION_DETECTION, app_name)
        if entry is None:
            return self._sub_result(sub, "failed", f"Application not found: {app_name}")
        running = await asyncio.to_thread(rt.apps.app_windows, entry)
        if sub.close:
            return await self._close_subtask(ctx, sub, entry, running)
        rt.bus.publish("application_detected", task_id=ctx.task_id, application=entry.name, desktop_id=entry.id,
                       running=bool(running), windows=[w.to_dict() for w in running])

        # ---------- APPLICATION_LAUNCH ----------
        self.state.to(S.APPLICATION_LAUNCH, entry.name)
        extra_args, env = self.agents.launch_options(entry)
        window = None
        for attempt in range(1, 3):
            await ctx.control.checkpoint()
            try:
                entry, window, launched = await rt.apps.aensure_open(entry, extra_args, env)
                rt.bus.publish("application_opened", task_id=ctx.task_id, application=entry.name, launched=launched,
                               window=window.to_dict())
                break
            except LaunchFailed as exc:
                logger.warning("launch attempt %d for %s failed: %s", attempt, entry.name, exc)
                if attempt == 1:
                    self.state.to(S.RECOVERING, str(exc))
                    self.state.to(S.APPLICATION_DETECTION, "retry")
                    self.state.to(S.APPLICATION_LAUNCH, "retry")
                    await asyncio.sleep(1.5)
                    continue
                self.state.to(S.WAITING_FOR_USER, str(exc))
                answer = await rt.broker.choose(ctx.task_id, f"{exc}. Open it yourself, then press Continue.",
                                                ["Continue", "Cancel"], MASTER)
                if answer != "Continue":
                    return self._sub_result(sub, "failed", str(exc))
                wins = await asyncio.to_thread(rt.apps.app_windows, entry)
                if not wins:
                    return self._sub_result(sub, "failed", f"{entry.name} still has no window")
                window = await rt.apps.aactivate(entry)
                self.state.to(S.APPLICATION_LAUNCH, "user opened it")
        assert window is not None

        # ---------- AGENT_CREATION ----------
        self.state.to(S.AGENT_CREATION, entry.name)
        agent, created = self.agents.get(entry)
        agent.attach(window)
        if created:
            ui = await agent.inspect(ctx)
            rt.bus.publish("agent_created", task_id=ctx.task_id, agent=agent.name, application=entry.name,
                           plugin=agent.plugin_name or None, ui_summary={
                               "title": ui["window"]["title"], "element_counts": ui["element_counts"],
                               "sources": ui["sources"]})
        self.state.to(S.OBSERVING, f"delegating to {agent.name}")

        # ---------- delegate ----------
        instruction = sub.instruction
        if sub.uses_previous and ctx.shared.get("results"):
            prev = ctx.shared["results"][-1]
            instruction += f" (Result of the previous step in {prev['app']}: {prev['result'][:300]})"
        msg = AgentMessage(sender=MASTER, recipient=agent.name, task=instruction or f"Open {entry.name}",
                           context={"subtask": sub.id, "steps": sub.steps, "expected_result": sub.expected_result,
                                    "window": window.to_dict()})
        self._send(ctx, msg)
        sub.instruction = instruction
        response = await agent.run(msg, sub, ctx)
        self._send(ctx, response)
        rt.apps.app_windows(entry)  # refresh tracked windows/pids after the sub-agent's work
        self.state.force(S.COMPLETED if response.status == "completed" else S.FAILED, response.result[:200])
        res = self._sub_result(sub, response.status, response.result, agent=agent.name, application=entry.name)
        rt.bus.publish("subtask_completed", task_id=ctx.task_id, **res)
        return res

    async def _close_subtask(self, ctx: TaskContext, sub: Subtask, entry: Any, windows: list[Any]) -> dict[str, Any]:
        """Quit an app the reliable way: ask the window manager to close its windows, then check they are gone.

        Clicking a title-bar "Close" button is fragile (client-side decorations, wrong coordinates) and the old
        check accepted "another window became active" as success while the app kept running."""
        rt = self.rt
        if not windows:
            res = self._sub_result(sub, "completed", f"{entry.name} wasn't open", application=entry.name)
        else:
            self.state.to(S.APPLICATION_LAUNCH, f"closing {entry.name}")
            n = await rt.apps.aclose(entry)
            ctx.recorder.record("action", action="close_application", application=entry.name, windows=n)
            left = windows
            for _ in range(24):  # up to ~6 s: apps may finish saving state first
                await asyncio.sleep(0.25)
                left = await asyncio.to_thread(rt.apps.app_windows, entry)
                if not left:
                    break
            if not left:
                res = self._sub_result(sub, "completed", f"Closed {entry.name}", application=entry.name)
            else:
                titles = ", ".join(f'"{w.title}"' for w in left[:3])
                res = self._sub_result(sub, "failed", f"{entry.name} is still open ({titles}); it may be asking "
                                       "to save changes", application=entry.name)
        self.state.force(S.COMPLETED if res["status"] == "completed" else S.FAILED, res["result"][:200])
        rt.bus.publish("subtask_completed", task_id=ctx.task_id, **res)
        return res

    @staticmethod
    def _sub_result(sub: Subtask, status: str, result: str, **extra: Any) -> dict[str, Any]:
        return {"subtask": sub.id, "app": sub.app, "status": status, "result": result, **extra}

    @staticmethod
    def _summarize(results: list[dict[str, Any]]) -> str:
        if len(results) == 1:
            return results[0]["result"]
        return " · ".join(f"{r.get('application', r['app'])}: {r['result']}" for r in results)


class TaskManager:
    """Queues commands and runs them one at a time (one mouse, one keyboard)."""

    def __init__(self, runtime: Runtime, master: MasterAgent | None = None) -> None:
        self.rt = runtime
        self.master = master or MasterAgent(runtime)
        self.tasks: dict[str, TaskHandle] = {}
        self._lock = asyncio.Lock()

    def submit(self, command: str) -> TaskHandle:
        task_id = f"task_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        handle = TaskHandle(task_id, command.strip(), TaskControl(self.rt.bus, task_id))
        self.tasks[task_id] = handle
        handle.future = asyncio.create_task(self._run(handle))
        return handle

    async def _run(self, handle: TaskHandle) -> None:
        async with self._lock:
            if handle.control.stopped:
                handle.status, handle.result, handle.finished = "stopped", "Stopped before it started", time.time()
                return
            handle.status = "running"
            record = await self.master.run(handle)
            handle.status = {"success": "completed"}.get(record["result"], record["result"])
            handle.result = record["summary"]
            handle.finished = time.time()

    async def run_and_wait(self, command: str) -> TaskHandle:
        handle = self.submit(command)
        assert handle.future is not None
        await handle.future
        return handle

    def get(self, task_id: str) -> TaskHandle | None:
        return self.tasks.get(task_id)

    def pause(self, task_id: str) -> bool:
        h = self.tasks.get(task_id)
        if h and h.status in ("queued", "running"):
            h.control.pause()
            return True
        return False

    def resume(self, task_id: str) -> bool:
        h = self.tasks.get(task_id)
        if h:
            h.control.resume()
            return True
        return False

    def stop(self, task_id: str) -> bool:
        h = self.tasks.get(task_id)
        if h and h.status in ("queued", "running"):
            h.control.stop()
            self.rt.broker.cancel_task(task_id)
            return True
        return False

    def running(self) -> TaskHandle | None:
        return next((h for h in self.tasks.values() if h.status == "running"), None)
