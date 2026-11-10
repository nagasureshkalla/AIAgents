"""The planner harness: editable guidance appended to the fixed action protocol."""

from __future__ import annotations

import os
from pathlib import Path

from planner.action_planner import SYSTEM, ActionPlanner
from tests.fakes import make_runtime


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_no_harness_keeps_the_builtin_prompt(tmp_path: Path):
    planner = ActionPlanner(llm=None, harness_dir=tmp_path / "missing")  # type: ignore[arg-type]
    assert planner.system_prompt("Calculator") == SYSTEM.format(app="Calculator", knowledge="")


def test_harness_text_is_appended_and_app_skills_are_scoped(tmp_path: Path):
    _write(tmp_path, {
        "systemprompt.md": "Prefer typing whole expressions.",
        "guardrails.md": "- When operating calculator, load the `calculator` skill.\n- Clear stale input first.",
        "LongTermMEMORY.md": "- Escape clears the calculator display.",
        "skills/calculator/SKILL.md": "---\nname: calculator\ndescription: operating calculator\n---\nPress Escape first.",
    })
    planner = ActionPlanner(llm=None, harness_dir=tmp_path)  # type: ignore[arg-type]
    calc = planner.system_prompt("Calculator")
    assert calc.startswith(SYSTEM.format(app="Calculator", knowledge=""))
    assert "Prefer typing whole expressions." in calc
    assert "More rules:\n- Clear stale input first." in calc and "load the `calculator` skill" not in calc
    assert "Guidance for Calculator:\nPress Escape first." in calc
    assert calc.endswith("Lessons from earlier tasks:\n- Escape clears the calculator display.")
    assert "Press Escape first." not in planner.system_prompt("Text Editor")


def test_harness_reloads_when_a_file_changes(tmp_path: Path):
    _write(tmp_path, {"systemprompt.md": "One."})
    planner = ActionPlanner(llm=None, harness_dir=tmp_path)  # type: ignore[arg-type]
    assert "One." in planner.system_prompt("Calculator")
    p = tmp_path / "systemprompt.md"
    p.write_text("Two.")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert "Two." in planner.system_prompt("Calculator")


def test_runtime_planner_reads_settings_harness_dir(tmp_path: Path):
    _write(tmp_path / "harness", {"guardrails.md": "- Never use the mouse for digits."})
    rt, _, _ = make_runtime(tmp_path, [])
    assert "Never use the mouse for digits." in rt.action_planner.system_prompt("Calculator")
