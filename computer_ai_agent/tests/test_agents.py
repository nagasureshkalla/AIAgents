"""End-to-end agent tests on the simulated desktop: master -> sub-agent -> observe/plan/act/verify."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agents.master_agent import TaskManager
from tests.fakes import make_runtime


async def _run(rt, command: str, responder=None):  # noqa: ANN001, ANN202
    tm = TaskManager(rt)
    events: list[dict] = []
    rt.bus.on(events.append)
    if responder:
        rt.bus.on(lambda e: responder(rt, e))
    handle = await asyncio.wait_for(tm.run_and_wait(command), timeout=20)
    return handle, events


async def test_calculator_task_completes_with_verification(tmp_path: Path):
    rt, app, backend = make_runtime(tmp_path, [
        {"reason": "type the expression", "action": "type_text", "text": "25*89\n", "expect": "result shown"},
        {"reason": "result 2225 is shown", "action": "done", "text": "2225"},
    ])
    handle, events = await _run(rt, "Open Calculator and calculate 25 * 89")
    assert handle.status == "completed", handle.result
    assert handle.result == "2225" and app.display == "2225"
    names = [e["event"] for e in events]
    for expected in ("task_started", "plan_created", "application_detected", "application_opened", "agent_created",
                     "action_started", "action_completed", "verification_completed", "task_completed"):
        assert expected in names, expected
    states = [e["state"] for e in events if e["event"] == "state_changed" and e["agent"] == "CalculatorSubAgent"]
    assert states[:2] == ["UNDERSTANDING", "PLANNING"] and "EXECUTING" in states and states[-1] == "COMPLETED"
    # persistent memory + task history were written
    assert (tmp_path / "data" / "applications" / "calculator" / "successful_actions.json").exists()
    assert rt.task_memory.load(handle.task_id)["result"] == "success"
    assert backend.calls == ["action", "action", "goal"]  # single-app command: no planning call


async def test_failed_action_triggers_recovery_then_success(tmp_path: Path):
    rt, app, _ = make_runtime(tmp_path, [
        {"action": "click", "element": 999},                       # invalid element -> planner error, re-plan
        {"action": "hotkey", "keys": ["ctrl", "k"]},                # no visible effect -> RECOVERING
        {"action": "click", "element": 4},                          # button "7" -> display "7"
        {"action": "done", "reason": "7 shown", "text": "7"},
    ])
    handle, events = await _run(rt, "Open Calculator and enter 7")
    assert handle.status == "completed", handle.result
    assert app.display == "7"
    recov = [e for e in events if e["event"] == "recovery"]
    assert len(recov) == 2  # bad element id + no-change hotkey
    assert any(e["event"] == "state_changed" and e["state"] == "RECOVERING" for e in events)


async def test_goal_check_rejects_premature_done(tmp_path: Path):
    rt, app, backend = make_runtime(tmp_path, [{"action": "done", "reason": "looks done"}], goal_complete=False)

    def answer(rt, e):  # noqa: ANN001
        if e["event"] == "user_input_required":
            rt.broker.respond(e["request"]["id"], "Stop")

    handle, events = await _run(rt, "Open Calculator and calculate 6*7", answer)
    assert handle.status == "failed"
    assert any(e["event"] == "verification_completed" and e.get("what") == "goal" and not e["ok"] for e in events)


async def test_destructive_click_waits_for_confirmation_and_respects_deny(tmp_path: Path):
    rt, app, _ = make_runtime(tmp_path, [
        {"action": "click", "element": 3, "reason": "clear history"},  # "Delete all history" -> CONFIRM
        {"action": "fail", "reason": "user denied"},
    ])
    asked: list[dict] = []

    def answer(rt, e):  # noqa: ANN001
        if e["event"] == "confirmation_required":
            asked.append(e["request"])
            rt.broker.respond(e["request"]["id"], "deny")
        elif e["event"] == "user_input_required":
            rt.broker.respond(e["request"]["id"], "Stop")

    handle, _ = await _run(rt, "Open Calculator and delete all history", answer)
    assert asked and "irreversible" in asked[0]["message"]
    assert app.deleted is False
    assert handle.status == "failed"


async def test_destructive_click_runs_after_approval(tmp_path: Path):
    rt, app, _ = make_runtime(tmp_path, [
        {"action": "click", "element": 3},
        {"action": "done", "reason": "cleared"},
    ])

    def answer(rt, e):  # noqa: ANN001
        if e["event"] == "confirmation_required":
            rt.broker.respond(e["request"]["id"], "approve")

    handle, _ = await _run(rt, "Open Calculator and delete all history", answer)
    assert handle.status == "completed" and app.deleted is True


async def test_stop_and_pause(tmp_path: Path):
    rt, app, _ = make_runtime(tmp_path, [{"action": "wait", "amount": 1}] * 5)
    tm = TaskManager(rt)
    h = tm.submit("Open Calculator and wait")
    await asyncio.sleep(0.3)
    assert tm.pause(h.task_id)
    await asyncio.sleep(0.3)
    tm.resume(h.task_id)
    assert tm.stop(h.task_id)
    await asyncio.wait_for(h.future, 10)
    assert h.status == "stopped"


async def test_dynamic_subagent_for_unknown_app(tmp_path: Path):
    rt, app, _ = make_runtime(tmp_path, [{"action": "done", "reason": "open"}])
    from applications.application_detector import AppEntry

    rt.apps.detector.apps = {"xyz-studio": AppEntry("xyz-studio", "XYZ Studio", "xyz", wm_class="xyzstudio")}
    rt.apps.detector.matches_window = lambda *a, **k: True  # the fake desktop's only window plays XYZ Studio
    tm = TaskManager(rt)
    h = await asyncio.wait_for(tm.run_and_wait("Open XYZ Studio"), 10)
    assert h.status == "completed", h.result
    agents = tm.master.agents.list()
    assert agents[0]["name"] == "XYZStudioSubAgent" and agents[0]["plugin"] is None


async def test_keyboard_chain_runs_in_one_decision_and_stops_on_failure(tmp_path: Path):
    rt, app, backend = make_runtime(tmp_path, [
        # one decision: type, then two chained keyboard steps
        {"action": "type_text", "text": "6*7", "then": [{"action": "key_press", "keys": ["Enter"]},
                                                         {"action": "hotkey", "keys": ["ctrl", "k"]},  # no effect
                                                         {"action": "key_press", "keys": ["Escape"]}]},
        {"action": "done", "text": "42"},
    ])
    handle, events = await _run(rt, "Open Calculator and calculate 6*7")
    assert handle.status == "completed" and handle.result == "42"
    done = [e for e in events if e["event"] == "action_completed"]
    assert [e["label"] for e in done] == ['type_text "6*7"', "key_press Enter", "hotkey ctrl+k"]
    assert done[-1]["ok"] is False                        # the chain stopped there: Escape never ran
    assert ("press", "Escape") not in rt.desktop.keyboard.log
    assert backend.calls.count("action") == 2             # 4 actions planned with 2 model calls


async def test_close_app_uses_the_window_manager_and_verifies(tmp_path: Path):
    rt, app, backend = make_runtime(tmp_path, [])
    app.running = True
    handle, events = await _run(rt, "can you close the calculator app?")
    assert handle.status == "completed", handle.result
    assert handle.result == "Closed Calculator" and not app.running
    assert rt.apps.windows.closed  # _NET_CLOSE_WINDOW, not a click
    assert backend.calls == []     # no model call at all
    assert not any(e["event"] == "action_started" for e in events)


async def test_close_app_that_is_not_open_says_so(tmp_path: Path):
    rt, app, backend = make_runtime(tmp_path, [])
    app.running = False
    handle, _ = await _run(rt, "close calculator")
    assert handle.status == "completed" and handle.result == "Calculator wasn't open"
    assert not rt.apps.windows.closed and backend.calls == []


async def test_close_app_that_stays_open_is_reported_as_failed(tmp_path: Path, monkeypatch):
    rt, app, _ = make_runtime(tmp_path, [])
    app.running = True
    monkeypatch.setattr(type(rt.apps.windows), "close", lambda self, wid: True)  # e.g. a "save changes?" dialog
    handle, _ = await _run(rt, "close calculator")
    assert handle.status == "failed" and "still open" in handle.result
