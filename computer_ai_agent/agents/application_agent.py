"""GenericApplicationAgent: controls ANY application through OBSERVE -> PLAN -> ACT -> OBSERVE -> VERIFY.

Specialized sub-agents (applications/plugins/<app>/agent.py) subclass it to add knowledge, launch flags and
app-specific state readers. Unknown apps get a class created on the fly: type("XyzSubAgent", (GenericApplicationAgent,), {}).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.logging_setup import TaskDebugRecorder
from core.state_machine import AgentState, StateMachine
from core.types import Action, ActionResult, AgentMessage, Observation, WindowInfo
from memory.application_memory import ApplicationMemory
from planner.action_planner import StepRecord
from safety.confirmation import TaskControl, TaskStopped
from safety.permission_manager import Risk
from vision.screen_capture import to_jpeg
from vision.ui_detector import draw_marks

if TYPE_CHECKING:
    from applications.application_detector import AppEntry
    from core.runtime import Runtime
    from planner.task_planner import Subtask

logger = logging.getLogger(__name__)
S = AgentState


@dataclass
class TaskContext:
    task_id: str
    command: str
    control: TaskControl
    recorder: TaskDebugRecorder
    record: dict[str, Any]
    step_counter: list[int] = field(default_factory=lambda: [0])
    shared: dict[str, Any] = field(default_factory=dict)   # results passed between sub-agents

    def next_step(self) -> int:
        self.step_counter[0] += 1
        return self.step_counter[0]


class GenericApplicationAgent:
    plugin_name: str = ""           # set by the agent manager for plugin classes
    knowledge_seed: dict[str, Any] = {}

    def __init__(self, runtime: Runtime, entry: AppEntry) -> None:
        self.rt = runtime
        self.entry = entry
        self.app_name = entry.name
        self.name = type(self).__name__
        self.pid: int | None = None
        self.window_id: int | None = None
        self.window: WindowInfo | None = None
        self.memory = ApplicationMemory(runtime.settings.data_dir, entry.name, self.knowledge())
        self.memory.update_profile(application=entry.name, desktop_id=entry.id, exec=entry.exec,
                                   wm_class=entry.wm_class, agent=self.name, plugin=self.plugin_name or None)
        self.task_history: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.ui_model: dict[str, Any] = {}
        self.last_obs: Observation | None = None
        self.status = "idle"
        self.current_action = ""

    # ---- plugin hooks -----------------------------------------------------------------------------------------
    def knowledge(self) -> dict[str, Any]:
        return dict(self.knowledge_seed) or {"application": self.app_name, "tips": [], "shortcuts": {}}

    def launch_args(self) -> list[str]:
        return []

    def launch_env(self) -> dict[str, str]:
        return {}

    def describe_state(self, obs: Observation) -> str:
        """Extra app-specific facts for the planner (e.g. the browser URL)."""
        return ""

    def preparation(self, obs: Observation) -> list[Action]:
        """Deterministic actions to run before the first planning step (no model call). Default: none."""
        return []

    async def _prepare_window(self, obs: Observation, ctx: TaskContext, sm: StateMachine) -> Observation:
        actions = self.preparation(obs)
        if not actions:
            return obs
        sm.to(S.PLANNING, "preparing the window")
        sm.to(S.EXECUTING, "preparing the window")
        for a in actions:
            self._publish("action_started", ctx, step=0, action=a.to_dict(), label=a.short(), reason=a.reason)
            res = await self.rt.engine.execute(a, obs)
            await self._wait_until_stable()
            self._publish("action_completed", ctx, step=0, action=a.to_dict(), label=a.short(), ok=res.ok,
                          message=a.reason or res.message, duration_ms=round(res.duration_ms))
        sm.to(S.OBSERVING)
        return await self.observe(ctx, use_cache=False)

    # ---- helpers ----------------------------------------------------------------------------------------------
    def attach(self, window: WindowInfo) -> None:
        self.window, self.window_id, self.pid = window, window.id, window.pid

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "application": self.app_name, "desktop_id": self.entry.id, "pid": self.pid,
                "window_id": self.window_id, "status": self.status, "current_action": self.current_action,
                "plugin": self.plugin_name or None, "tasks": len(self.task_history),
                "actions": len(self.action_history), "known_ui": len(self.memory.ui.get("elements", {}))}

    def _publish(self, event: str, ctx: TaskContext, **data: Any) -> None:
        self.rt.bus.publish(event, task_id=ctx.task_id, agent=self.name, application=self.app_name, **data)

    async def observe(self, ctx: TaskContext, force_ocr: bool = False, use_cache: bool = True) -> Observation:
        obs = await asyncio.to_thread(self.rt.detector.observe, force_ocr, use_cache)
        self.last_obs = obs
        self._publish("observation", ctx, window=obs.active_window.to_dict() if obs.active_window else None,
                      elements=[e.to_dict() for e in obs.elements[:250]], sources=obs.sources,
                      timing={k: round(v, 1) for k, v in self.rt.detector.last_timing.items()})
        return obs

    def _belongs(self, obs: Observation) -> bool:
        w = obs.active_window
        if w is None:
            return False
        if self.pid and w.pid == self.pid:
            return True
        return self.rt.apps.detector.matches_window(self.entry, w)

    def _want_image(self, obs: Observation, last_problem: bool, step_in_task: int) -> bool:
        mode = self.rt.settings.vision_mode
        if mode == "never":
            return False
        if mode == "always":
            return True
        # A screenshot adds ~1-2k tokens (seconds per step). Send it for the first look, after a problem, or when
        # there is too little text information (no accessibility and almost no OCR) to act on.
        return step_in_task == 0 or last_problem or len(obs.elements) < 6

    def _image_for_model(self, obs: Observation):  # noqa: ANN202
        region = obs.active_window.geometry if obs.active_window and obs.active_window.geometry[2] > 50 else None
        img, _, _ = draw_marks(obs, self.rt.settings.model_image_width, region)
        return img

    def expects_navigation(self, action: Action, el) -> bool:  # noqa: ANN001
        """Actions after which the app may load something new (a page, a view): wait for it, not just for pixels."""
        if action.action in ("click", "double_click") and el is not None and el.type in ("link", "menuitem", "tab"):
            return True
        if action.action == "key_press" and any(k.lower() in ("enter", "return") for k in action.keys):
            return True
        if action.action == "type_text" and action.text.endswith("\n"):
            return True
        return action.action == "hotkey" and {k.lower() for k in action.keys} & {"f5", "left", "right"} != set()

    def _region_grab(self, region: tuple[int, int, int, int] | None):  # noqa: ANN202
        img = self.rt.desktop.screen.grab()
        if region and region[2] > 20 and region[3] > 20:
            x, y, w, h = region
            return img.crop((max(0, x), max(0, y), x + w, y + h))
        return img

    async def _settle(self, action: Action, el, before: Observation) -> float:  # noqa: ANN001
        """After an action: if navigation is expected, wait (max NAV_TIMEOUT) for evidence it happened: a title
        change or a repaint of a large part of the window. Then wait briefly (max SETTLE_TIMEOUT) for the window
        to stop changing. Only the target window is compared, so clocks or other monitors never delay the agent,
        and a window that animates forever costs at most SETTLE_TIMEOUT."""
        from vision.screen_capture import diff_ratio

        t0 = time.monotonic()
        win = before.active_window
        region = win.geometry if win else None
        if self.expects_navigation(action, el):
            title0 = win.title if win else ""
            ref = await asyncio.to_thread(self._region_grab, region) if before.image is not None else None
            deadline = t0 + self.rt.settings.nav_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(0.2)
                now_win = await asyncio.to_thread(self.rt.desktop.windows.active_window)
                if now_win is None or win is None or now_win.id != win.id or now_win.title != title0:
                    region = now_win.geometry if now_win else region
                    break
                cur = await asyncio.to_thread(self._region_grab, region)
                if ref is not None and diff_ratio(ref, cur) > 0.25:
                    break
        await self._wait_until_stable(region=region)
        return time.monotonic() - t0

    async def _wait_until_stable(self, max_wait: float | None = None, interval: float = 0.15,
                                 region: tuple[int, int, int, int] | None = None) -> float:
        """Poll the target window until two consecutive frames match. ~20 ms per frame; a static UI returns after
        one interval."""
        from vision.screen_capture import diff_ratio

        limit = self.rt.settings.settle_timeout if max_wait is None else max_wait
        start = time.monotonic()
        prev = await asyncio.to_thread(self._region_grab, region)
        while time.monotonic() - start < limit:
            await asyncio.sleep(interval)
            cur = await asyncio.to_thread(self._region_grab, region)
            if diff_ratio(prev, cur) < 0.004:
                break
            prev = cur
        return time.monotonic() - start

    # ---- inspection (dynamic sub-agent creation, spec section 6) --------------------------------------------
    async def inspect(self, ctx: TaskContext) -> dict[str, Any]:
        obs = await self.observe(ctx, force_ocr=False, use_cache=False)
        groups: dict[str, int] = {}
        for e in obs.elements:
            groups[e.type] = groups.get(e.type, 0) + 1
        win = obs.active_window
        self.ui_model = {
            "application": self.app_name,
            "window": {
                "title": win.title if win else "",
                "geometry": list(win.geometry) if win else None,
                "elements": [{"id": e.id, "type": e.type, "text": e.text[:80], "location": list(e.center),
                              "source": e.source} for e in obs.elements[:300]],
            },
            "element_counts": groups,
            "dialogs": [e.text for e in obs.elements if e.type == "dialog"],
            "sources": obs.sources,
        }
        self.memory.remember_ui(obs)
        return self.ui_model

    # ---- the loop ---------------------------------------------------------------------------------------------
    async def run(self, msg: AgentMessage, subtask: Subtask, ctx: TaskContext) -> AgentMessage:
        sm = StateMachine(self.name, self.rt.bus, ctx.task_id)
        sm.to(S.UNDERSTANDING, subtask.instruction or "focus the application")
        self.status = "running"
        started = time.time()
        goal = subtask.instruction
        actions_done: list[dict[str, Any]] = []
        task_entry = {"task_id": ctx.task_id, "goal": goal, "started": started}
        self.task_history.append(task_entry)

        def reply(status: str, result: str, **context: Any) -> AgentMessage:
            self.status = "idle"
            self.current_action = ""
            task_entry.update(status=status, result=result, duration=round(time.time() - started, 1))
            return AgentMessage(sender=self.name, recipient=msg.sender, status=status, result=result,
                                context={"actions": actions_done, **context})

        try:
            sm.to(S.PLANNING, "checking application window")
            sm.to(S.OBSERVING)
            obs = await self.observe(ctx, use_cache=False)
            if not goal:  # "Open X": success = the app's window is in front
                if self._belongs(obs):
                    sm.to(S.VERIFYING, "window is active")
                    sm.to(S.COMPLETED)
                    return reply("completed", f"{self.app_name} is open ({obs.active_window.title})")
                sm.to(S.FAILED, "window not active")
                return reply("failed", f"{self.app_name} window is not in front")

            hints = self.memory.hints(goal)  # constant for this subtask -> part of the cacheable system prompt
            history: list[StepRecord] = []
            retries: dict[str, int] = {}
            planning_errors = 0
            note = ""
            last_problem = False
            force_ocr = False
            step_in_task = 0
            max_steps = self.rt.settings.max_steps_per_subtask

            while True:
                await ctx.control.checkpoint()
                if step_in_task == 0 and not history:
                    obs = await self._prepare_window(obs, ctx, sm)
                if step_in_task >= max_steps:
                    sm.to(S.WAITING_FOR_USER, "step limit reached")
                    answer = await self.rt.broker.choose(
                        ctx.task_id, f"{self.name} used {max_steps} steps on \"{goal[:80]}\" without finishing. "
                        "Keep going?", ["Continue", "Stop"], self.name, to_jpeg(obs.image, 960))
                    if answer != "Continue":
                        sm.to(S.FAILED, "step limit")
                        return reply("failed", f"Stopped after {step_in_task} steps without completing: {goal}")
                    max_steps += self.rt.settings.max_steps_per_subtask
                    sm.to(S.OBSERVING)

                # ---------- PLAN ----------
                sm.to(S.PLANNING)
                extra = self.describe_state(obs)
                if not self._belongs(obs) and obs.active_window is not None:
                    extra += (f"\nThe active window is \"{obs.active_window.title}\" ({obs.active_window.wm_class}), not "
                              f"{self.app_name}: handle it (it may be a dialog) or switch_application.")
                image = self._image_for_model(obs) if self._want_image(obs, last_problem, step_in_task) else None
                decision = await self.rt.action_planner.next_action(
                    self.app_name, goal, obs, history, hints=hints, checkpoints=subtask.steps,
                    context=self._context_text(ctx), note=note, image=image, app_state=extra.strip())
                step = ctx.next_step()
                gen = decision.generation
                ctx.recorder.record("decision", step=step, agent=self.name, raw=decision.raw, error=decision.error,
                                    model_ms=round(gen.duration_ms) if gen else None, image=decision.used_image)
                ctx.recorder.screenshot(step, obs.image)
                ctx.recorder.dump(step, "observation", obs.to_dict())
                note = ""
                if decision.action is None:
                    planning_errors += 1
                    self._publish("recovery", ctx, step=step, reason=f"planner: {decision.error}")
                    if planning_errors >= self.rt.settings.max_action_retries:
                        sm.to(S.WAITING_FOR_USER, "model cannot decide")
                        answer = await self.rt.broker.choose(
                            ctx.task_id, f"I could not work out the next step for \"{goal[:80]}\" "
                            f"({decision.error}). What should I do?", ["Try again", "Stop"], self.name,
                            to_jpeg(obs.image, 960))
                        if answer != "Try again":
                            sm.to(S.FAILED, "model cannot decide")
                            return reply("failed", f"Could not plan the next action: {decision.error}")
                        planning_errors = 0
                        sm.to(S.OBSERVING)
                    else:
                        sm.to(S.RECOVERING, decision.error)
                        sm.to(S.OBSERVING)
                    note = f"Your last reply was invalid ({decision.error}). Reply with one valid JSON action."
                    obs = await self.observe(ctx, force_ocr=True, use_cache=False)
                    last_problem = True
                    continue
                planning_errors = 0
                action = decision.action
                step_in_task += 1

                # ---------- control actions ----------
                if action.action == "done":
                    sm.to(S.VERIFYING, "goal check")
                    self._publish("verification_started", ctx, step=step, what="goal")
                    # text is enough when the accessibility tree describes the window; a screenshot costs ~2 s
                    need_img = self.rt.settings.vision_mode == "always" or (
                        self.rt.settings.vision_mode == "auto" and "accessibility" not in obs.sources)
                    vimg = self._image_for_model(obs) if need_img else None
                    done_summary = "; ".join(
                        f"{h.action.get('action')} {h.action.get('target_label', '')} {h.action.get('text', '')[:40]!r}"
                        f" -> {'ok' if h.ok else 'problem'}" for h in history[-8:])
                    app_state = self.describe_state(obs)
                    complete, evidence, answer = await self.rt.verifier.check_goal(
                        goal, obs, vimg, action.reason, done_summary + (f"\n{app_state}" if app_state else ""))
                    self._publish("verification_completed", ctx, step=step, what="goal", ok=complete, detail=evidence)
                    if complete:
                        sm.to(S.COMPLETED)
                        self.memory.record_success(goal, actions_done, time.time() - started)
                        self.memory.remember_ui(obs)
                        result = action.text or answer or action.reason or "Done"
                        return reply("completed", result, evidence=evidence, answer=action.text or answer)
                    history.append(StepRecord(step, action.to_dict(), False, f"goal check says NOT complete: {evidence}"))
                    note = f"An independent check says the goal is not complete yet: {evidence}"
                    sm.to(S.PLANNING)
                    sm.to(S.OBSERVING)
                    obs = await self.observe(ctx, use_cache=False)
                    continue

                if action.action == "fail":
                    sm.to(S.WAITING_FOR_USER, "agent wants to give up")
                    answer = await self.rt.broker.choose(
                        ctx.task_id, f"{self.name} thinks this cannot be done: {action.reason}", ["Keep trying", "Stop"],
                        self.name, to_jpeg(obs.image, 960))
                    if answer != "Keep trying":
                        sm.to(S.FAILED, action.reason)
                        self.memory.record_task_failure()
                        return reply("failed", action.reason or "The agent could not complete the task")
                    history.append(StepRecord(step, action.to_dict(), False, "user asked to keep trying"))
                    note = "The user wants you to keep trying with a different approach."
                    sm.to(S.OBSERVING)
                    obs = await self.observe(ctx, use_cache=False)
                    continue

                if action.action == "ask_user":
                    sm.to(S.WAITING_FOR_USER, action.question)
                    answer = await self.rt.broker.choose(ctx.task_id, action.question or action.reason,
                                                         action.choices, self.name, to_jpeg(obs.image, 960))
                    history.append(StepRecord(step, action.to_dict(), True, f"user answered: {answer}"))
                    if answer is None:
                        note = "The user did not answer; make the most reasonable choice yourself."
                    sm.to(S.OBSERVING)
                    obs = await self.observe(ctx, use_cache=False)
                    continue

                # ---------- SAFETY ----------
                verdict = self.rt.permissions.check(action, obs, self.app_name, ctx.command)
                if verdict.risk == Risk.BLOCK:
                    history.append(StepRecord(step, action.to_dict(), False, f"BLOCKED by safety: {verdict.reason}"))
                    self._publish("action_completed", ctx, step=step, action=action.to_dict(), ok=False,
                                  message=f"blocked: {verdict.reason}")
                    note = f"That action is not allowed ({verdict.reason}). Use ask_user if the user must act."
                    continue
                if self.rt.permissions.needs_prompt(verdict):
                    sm.to(S.WAITING_FOR_USER, verdict.reason)
                    approved = await self.rt.broker.confirm(
                        ctx.task_id, f"{verdict.reason} Do you want me to continue?", self.name, verdict.category,
                        to_jpeg(obs.image, 960))
                    sm.to(S.PLANNING)
                    if not approved:
                        history.append(StepRecord(step, action.to_dict(), False, "user DENIED this action"))
                        note = "The user denied that action. Do not repeat it; find another way or use fail."
                        ctx.recorder.record("denied", step=step, action=action.to_dict(), reason=verdict.reason)
                        sm.to(S.OBSERVING)
                        continue

                # ---------- ACT + OBSERVE + VERIFY ----------
                obs, v = await self._perform(action, obs, step, ctx, sm, history, actions_done, force_ocr)
                force_ocr = False
                # Chained keyboard follow-ups ("then"): each one is safety-checked and verified; stop at the first
                # problem and let the planner look at the screen again.
                for follow in decision.followups if v.ok else []:
                    await ctx.control.checkpoint()
                    fv = self.rt.permissions.check(follow, obs, self.app_name, ctx.command)
                    if fv.risk != Risk.SAFE:
                        note = f"Chain stopped before {follow.short()}: it needs separate approval. Send it alone."
                        break
                    step = ctx.next_step()
                    step_in_task += 1
                    action = follow
                    obs, v = await self._perform(follow, obs, step, ctx, sm, history, actions_done, False)
                    if not v.ok:
                        break

                if v.ok:
                    last_problem = False
                    self.current_action = ""
                    continue

                # ---------- RECOVER ----------
                sig = f"{action.action}|{action.target_label or action.element}|{action.text[:30]}|{'+'.join(action.keys)}"
                retries[sig] = retries.get(sig, 0) + 1
                self.memory.record_failure(goal, action.to_dict(), v.detail)
                sm.to(S.RECOVERING, v.detail)
                self._publish("recovery", ctx, step=step, reason=v.detail, attempt=retries[sig])
                last_problem, force_ocr = True, True
                if retries[sig] >= self.rt.settings.max_action_retries:
                    sm.to(S.WAITING_FOR_USER, "action keeps failing")
                    answer = await self.rt.broker.choose(
                        ctx.task_id, f"\"{action.short()}\" failed {retries[sig]} times ({v.detail}). What now?",
                        ["Try another approach", "Stop"], self.name, to_jpeg(obs.image, 960))
                    if answer != "Try another approach":
                        sm.to(S.FAILED, "repeated failure")
                        self.memory.record_task_failure()
                        return reply("failed", f"Stopped: {action.short()} kept failing ({v.detail})")
                    retries[sig] = 0
                    note = "The user asked you to try a completely different approach for this step."
                else:
                    note = (f"Your previous action ({action.short()}) {v.status.replace('_', ' ')}: {v.detail}. "
                            "Look at the screen again and try an alternative (different element, keyboard shortcut, "
                            "waiting for loading, or closing a popup).")
                sm.to(S.OBSERVING)
                obs = await self.observe(ctx, force_ocr=True, use_cache=False)

        except TaskStopped:
            sm.force(S.FAILED, "stopped by user")
            return reply("stopped", "Stopped by the user")
        except Exception as exc:  # noqa: BLE001 - report, never crash the master
            logger.exception("%s crashed", self.name)
            sm.force(S.FAILED, str(exc))
            self.memory.record_task_failure()
            return reply("failed", f"{type(exc).__name__}: {exc}")

    async def _perform(self, action: Action, obs: Observation, step: int, ctx: TaskContext, sm: StateMachine,
                       history: list[StepRecord], actions_done: list[dict[str, Any]],
                       force_ocr: bool) -> tuple[Observation, Any]:
        """Execute one action, let the window settle, observe again and verify. Returns (new observation, verdict)."""
        sm.to(S.EXECUTING, action.short())
        point, el, _ = self.rt.engine.resolve(action, obs)
        if el is not None:
            action.target_label = f'{el.type} "{el.text[:40]}"'
        self.current_action = action.short()
        self._publish("action_started", ctx, step=step, action=action.to_dict(), label=action.short(),
                      reason=action.reason, expect=action.expect, point=list(point) if point else None,
                      bbox=list(el.bbox) if el else None)
        result: ActionResult = await self.rt.engine.execute(action, obs)
        settle_s = 0.0
        if result.ok and action.action not in ("wait", "screenshot", "mouse_move"):
            settle_s = await self._settle(action, el, obs)

        sm.to(S.OBSERVING)
        after = await self.observe(ctx, force_ocr=force_ocr, use_cache=False)
        sm.to(S.VERIFYING)
        self._publish("verification_started", ctx, step=step, what="action")
        v = self.rt.verifier.check_action(action, result, obs, after)
        # the engine's own report matters to the model (e.g. "pasted all 412 characters")
        detail = v.detail if not result.ok or result.message in v.detail else f"{result.message}; {v.detail}"
        entry = {**action.to_dict(), "step": step, "ok": v.ok, "verification": v.status, "detail": detail,
                 "point": list(result.point) if result.point else None, "duration_ms": round(result.duration_ms),
                 "settle_ms": round(settle_s * 1000), "observe_ms": round(self.rt.detector.last_timing.get("total_ms", 0)),
                 "agent": self.name, "application": self.app_name, "ts": time.time()}
        actions_done.append(entry)
        self.action_history.append(entry)
        ctx.record["actions"].append(entry)
        ctx.recorder.record("action", **entry)
        self.rt.action_log.log(self.app_name, self.name, action.action,
                               action.target_label or action.text[:60], result.point, v.ok, detail)
        self._publish("verification_completed", ctx, step=step, what="action", ok=v.ok, status=v.status, detail=detail)
        self._publish("action_completed", ctx, step=step, action=action.to_dict(), label=action.short(),
                      ok=v.ok, status=v.status, message=detail, point=entry["point"],
                      duration_ms=entry["duration_ms"])
        history.append(StepRecord(step, action.to_dict(), v.ok, f"{v.status}: {detail}", action.reason))
        return after, v

    @staticmethod
    def _context_text(ctx: TaskContext) -> str:
        items = ctx.shared.get("results", [])
        return " | ".join(f"{r['app']}: {r['result'][:200]}" for r in items[-3:])
