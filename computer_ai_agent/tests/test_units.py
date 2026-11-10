"""Unit tests for the pieces that need no desktop: parsing, state machine, safety, planning, memory, detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from actions.keyboard import normalize_key
from applications.application_detector import ApplicationDetector, _parse_desktop
from core.jsonutil import extract_json
from core.state_machine import AgentState, InvalidTransition, StateMachine
from core.types import Action, Element, Observation, WindowInfo
from memory.application_memory import ApplicationMemory
from memory.task_memory import TaskMemory
from planner.task_planner import TaskPlanner
from safety.permission_manager import PermissionManager, Risk


# ---------- actions / parsing ----------
def test_action_aliases_and_coordinates():
    a = Action.from_dict({"action": "Click", "coordinates": [512, 72]})
    assert a.action == "click" and (a.x, a.y) == (512, 72)
    assert Action.from_dict({"action": "type", "text": "Hello World"}).action == "type_text"
    h = Action.from_dict({"action": "hotkey", "keys": "CTRL+S"})
    assert h.keys == ["CTRL", "S"]
    assert Action.from_dict({"action": "press", "key": "Enter"}).keys == ["Enter"]
    with pytest.raises(ValueError):
        Action.from_dict({"action": "teleport"})


def test_extract_json_handles_think_fences_and_trailing_commas():
    assert extract_json('<think>hmm</think>{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"action": "click", "element": 3,}\n```') == {"action": "click", "element": 3}
    assert extract_json('noise {"x": {"y": "}"}} tail') == {"x": {"y": "}"}}
    assert extract_json("no json here") is None


def test_key_normalization():
    assert normalize_key("CTRL") == "ctrl"
    assert normalize_key("Return") == "enter"
    assert normalize_key("PageDown") == "page_down"
    assert normalize_key("A") == "a"
    assert normalize_key("F5") == "f5"
    with pytest.raises(ValueError):
        normalize_key("hyperdrive")


# ---------- state machine ----------
def test_state_machine_transitions():
    sm = StateMachine("t")
    for s in ("UNDERSTANDING", "APPLICATION_DETECTION", "APPLICATION_LAUNCH", "AGENT_CREATION", "OBSERVING",
              "PLANNING", "EXECUTING", "VERIFYING", "COMPLETED"):
        sm.to(AgentState(s))
    assert sm.state == AgentState.COMPLETED
    with pytest.raises(InvalidTransition):
        StateMachine("t").to(AgentState.EXECUTING)


# ---------- safety ----------
def _obs(*els: Element, title: str = "App") -> Observation:
    win = WindowInfo(1, title, "app", 1, (0, 0, 800, 600), True)
    return Observation(0, (800, 600), win, list(els), focused=next((e for e in els if "focused" in e.states), None))


def test_safety_rules():
    pm = PermissionManager(require_confirmation=True)
    pw = Element(1, "password", "", (10, 10, 100, 20), states=["focused"])
    assert pm.check(Action("type_text", text="hunter2"), _obs(pw), "Chrome").risk == Risk.BLOCK
    delete = Element(2, "button", "Delete permanently", (10, 40, 100, 20))
    v = pm.check(Action("click", element=2), _obs(delete), "Files")
    assert v.risk == Risk.CONFIRM and v.category == "delete"
    send = Element(3, "button", "Send", (10, 70, 60, 20))
    assert pm.check(Action("click", element=3), _obs(send), "Mail").category == "send"
    ok = Element(4, "button", "Search", (10, 90, 60, 20))
    assert pm.check(Action("click", element=4), _obs(ok), "Chrome").risk == Risk.SAFE
    term = _obs(title="user@host: ~")
    assert pm.check(Action("type_text", text="rm -rf ~/old\n"), term, "Terminal").risk == Risk.CONFIRM
    assert pm.check(Action("type_text", text="sudo apt update\n"), term, "Terminal").risk == Risk.CONFIRM
    assert pm.check(Action("type_text", text="nvidia-smi\n"), term, "Terminal").risk == Risk.SAFE
    assert pm.check(Action("close_application", app="gedit"), _obs(), "gedit", "Open gedit").risk == Risk.CONFIRM
    assert pm.check(Action("close_application", app="gedit"), _obs(), "gedit", "Close gedit").risk == Risk.SAFE
    assert pm.check(Action("key_press", keys=["Delete"]), _obs(), "Files").risk == Risk.CONFIRM


# ---------- planning ----------
def test_quick_and_rule_plans():
    q = TaskPlanner.quick_plan("Open Chrome.")
    assert q and q[0].app == "Chrome" and q[0].instruction == ""
    assert TaskPlanner.quick_plan("Open Chrome and search for OpenAI") is None
    plan = TaskPlanner.rule_plan("Open Chrome and download the file, then open VS Code and create a project")
    assert [s.app for s in plan] == ["Chrome", "VS Code"]
    assert plan[1].instruction == "create a project"


# ---------- application detection ----------
def test_desktop_entry_parsing_and_matching(tmp_path: Path):
    (tmp_path / "google-chrome.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Google Chrome\nGenericName=Web Browser\n"
        "Exec=/usr/bin/google-chrome-stable %U\n\n[Desktop Action new]\nName=New Window\nExec=other\n")
    (tmp_path / "code_code.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Visual Studio Code\nExec=env A=1 /snap/bin/code --force-user-env %F\n"
        "StartupWMClass=code_code\nKeywords=vscode;\n")
    (tmp_path / "org.gnome.Calculator.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Calculator\nExec=gnome-calculator\n")
    det = ApplicationDetector([tmp_path])
    det.discover()
    chrome = det.find("chrome")
    assert chrome and chrome.id == "google-chrome" and chrome.argv == ["/usr/bin/google-chrome-stable"]
    assert det.find("VS Code").id == "code_code"
    assert det.find("calculator").binary == "gnome-calculator"
    assert det.matches_window(chrome, WindowInfo(1, "x", "Google-chrome", 1, (0, 0, 1, 1)))
    assert not det.matches_window(chrome, WindowInfo(2, "x", "Gnome-terminal", 2, (0, 0, 1, 1)))
    assert _parse_desktop(tmp_path / "missing.desktop") is None


# ---------- memory ----------
def test_application_memory_learns_and_recalls(tmp_path: Path):
    mem = ApplicationMemory(tmp_path, "Google Chrome", {"tips": ["ctrl+l focuses the address bar"]})
    mem.record_success("search google for rtx 3090", [{"action": "hotkey", "keys": ["ctrl", "l"]},
                                                       {"action": "type_text", "text": "rtx 3090\n"}], 4.2)
    mem2 = ApplicationMemory(tmp_path, "Google Chrome")  # reload from disk
    hints = mem2.hints("search google for nvidia rtx 3090 reviews")
    assert "ctrl+l focuses the address bar" in hints
    assert "succeeded with" in hints and "ctrl+l" in mem2.knowledge["learned_shortcuts"]
    assert (tmp_path / "applications" / "google_chrome" / "successful_actions.json").exists()


def test_task_memory_similarity(tmp_path: Path):
    tm = TaskMemory(tmp_path)
    rec = TaskMemory.new_record("t1", "Open Chrome and search for Python tutorials")
    rec["result"] = "success"
    tm.save(rec)
    assert tm.similar_successes("open chrome and search for python docs")
    assert not tm.similar_successes("calculate 2+2 in calculator")
    assert tm.list()[0]["task_id"] == "t1"


# ---------- speed / correctness fixes ----------
def test_planner_repairs_broken_json_and_reads_chains():
    from planner.action_planner import parse_reply

    broken = '```json\n{\n "reason": "x",\n "action": "select_text": "element": 5,\n "expect": "y"\n}\n```'
    assert parse_reply(broken) == {"reason": "x", "action": "select_text", "element": 5, "expect": "y"}


def test_paste_is_used_for_long_or_structured_text():
    from actions.action_engine import ActionEngine

    assert not ActionEngine.should_paste("google.com\n")
    assert not ActionEngine.should_paste("25*89\n")
    assert ActionEngine.should_paste('[{"title": "Inception"}]')
    assert ActionEngine.should_paste("line one\nline two")
    assert ActionEngine.should_paste("x" * 60)


def test_long_text_is_marked_as_a_shortened_view():
    e = Element(1, "textbox", "a" * 500, (0, 0, 10, 10))
    d = e.describe()
    assert "shortened view of 500 characters" in d and len(d) < 200
    assert Element(2, "button", "Save", (0, 0, 1, 1)).describe() == '[2] button "Save" at (0,0,1x1) <accessibility>'


def test_single_app_commands_skip_the_planning_model():
    one = TaskPlanner.single_app_plan('open visual studio and create a json file and save it on desktop')
    assert one and one[0].app == "visual studio" and "save it on desktop" in one[0].instruction
    assert TaskPlanner.single_app_plan("Open Chrome, go to Google, search for X, and open the first result.")
    assert TaskPlanner.single_app_plan("Open Chrome and download the file, then open VS Code") is None
    assert TaskPlanner.single_app_plan("Open Chrome and open VS Code") is None


def test_close_plan():
    plan = TaskPlanner.close_plan("can you close the calculator app?")
    assert plan and plan[0].app == "calculator" and plan[0].close
    assert TaskPlanner.close_plan("exit vs code")[0].app == "vs code"
    for cmd in ("close the tab in chrome", "close the file", "close this dialog", "close chrome and open vs code"):
        assert TaskPlanner.close_plan(cmd) is None, cmd
