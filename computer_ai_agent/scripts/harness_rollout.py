#!/usr/bin/env python3
"""Run one task on the simulated calculator desktop with the real local model and a candidate planner harness.

harness2's executor for this agent (../GoogleHrness/backend/agents). Nothing touches the real desktop: the screen,
mouse, keyboard and windows are the test doubles from tests/fakes.py; only the model calls are real.

    .venv/bin/python scripts/harness_rollout.py --task harness/tasks.json --task-id stale-display \\
        --harness-dir harness --out /tmp/run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actions.platform import Desktop  # noqa: E402
from agents.master_agent import TaskManager  # noqa: E402
from applications.application_detector import AppEntry, ApplicationDetector  # noqa: E402
from applications.application_manager import ApplicationManager  # noqa: E402
from config.settings import load_settings  # noqa: E402
from core.runtime import build_runtime  # noqa: E402
from models.model_manager import ModelManager  # noqa: E402
from tests.fakes import FakeApp, FakeDetector, FakeKeyboard, FakeMouse, FakeScreen, FakeWindows  # noqa: E402


def _trajectory(journal: Path) -> list[str]:
    lines: list[str] = []
    if not journal.is_file():
        return lines
    for raw in journal.read_text(encoding="utf-8").splitlines():
        e = json.loads(raw)
        kind = e.get("kind")
        if kind == "decision":
            lines.append(f"[ACTION] step {e.get('step')}: model decided {json.dumps(e.get('raw'), default=str)[:600]}")
            if e.get("error"):
                lines.append(f"[ERROR] planner: {e['error']}")
        elif kind == "action":
            fields = {k: e[k] for k in ("text", "keys", "element", "target_label") if e.get(k)}
            lines.append(f"[OBSERVATION] {e.get('action')} {json.dumps(fields, default=str)[:300]} -> "
                         f"{'OK' if e.get('ok') else 'PROBLEM'} ({e.get('verification')}): {str(e.get('detail'))[:300]}")
        elif kind == "denied":
            lines.append(f"[OBSERVATION] action denied by the user: {e.get('reason')}")
        elif kind == "task_finished":
            lines.append(f"[OBSERVATION] task finished: {e.get('result')} {str(e.get('summary', ''))[:300]}")
    return lines


async def _run(spec: dict, harness_dir: Path, work: Path, timeout: float) -> dict:
    settings = load_settings()
    settings.data_dir, settings.logs_dir, settings.harness_dir = work / "data", work / "logs", harness_dir
    settings.action_delay = settings.settle_timeout = settings.nav_timeout = 0.0
    settings.vision_mode = "never"
    settings.max_steps_per_subtask = int(spec.get("max_steps", 10))
    settings.confirmation_timeout = 5

    app = FakeApp()
    app.display = spec.get("initial_display", "")
    desktop = Desktop("fake", ":99", FakeScreen(app), FakeMouse(app), FakeKeyboard(app), FakeWindows(app), True)
    detector_apps = ApplicationDetector(dirs=[])
    detector_apps.apps = {"org.gnome.Calculator": AppEntry("org.gnome.Calculator", "Calculator", "gnome-calculator",
                                                           wm_class="gnome-calculator")}
    rt = build_runtime(settings, desktop=desktop, models=ModelManager(settings),
                       apps=ApplicationManager(desktop.windows, ":99", 2.0, detector_apps),
                       detector=FakeDetector(desktop, app))

    def answer(e: dict) -> None:  # the simulated user denies destructive actions and stops on questions
        if e["event"] == "confirmation_required":
            rt.broker.respond(e["request"]["id"], "deny")
        elif e["event"] == "user_input_required":
            rt.broker.respond(e["request"]["id"], "Stop")

    rt.bus.on(answer)
    t0 = time.perf_counter()
    handle = await asyncio.wait_for(TaskManager(rt).run_and_wait(spec["command"]), timeout=timeout)
    journal = settings.logs_dir / "tasks" / f"{handle.task_id}.jsonl"
    return {
        "status": handle.status,
        "result": handle.result,
        "display": app.display,
        "history_deleted": app.deleted,
        "seconds": round(time.perf_counter() - t0, 1),
        "model": settings.model_name,
        "system": rt.action_planner.system_prompt("Calculator"),
        "trajectory": _trajectory(journal),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    spec = next((t for t in json.loads(Path(args.task).read_text()) if t["id"] == args.task_id), None)
    if spec is None:
        raise SystemExit(f"task {args.task_id!r} not in {args.task}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cua-harness-") as tmp:
        try:
            r = asyncio.run(_run(spec, Path(args.harness_dir).resolve(), Path(tmp), args.timeout))
            error = ""
        except Exception as exc:  # noqa: BLE001 - becomes the rollout's [ERROR]
            r, error = {}, f"{type(exc).__name__}: {exc}"
    lines = [f"[ACTION] command: {spec['command']}"]
    if spec.get("initial_display"):
        lines.append(f"[OBSERVATION] the calculator display already shows {spec['initial_display']!r}")
    if error:
        lines.append(f"[ERROR] {error}")
    else:
        lines += r["trajectory"]
        lines.append(f"[OBSERVATION] final display {r['display']!r}; task {r['status']}; reported result {r['result']!r}")
        answer = (f"status: {r['status']}\nreported result: {r['result']}\ncalculator display: {r['display']}\n"
                  f"history deleted: {r['history_deleted']}\n")
        (out / "answer.txt").write_text(answer, encoding="utf-8")
    (out / "trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps({**r, "error": error}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ok": not error, "status": r.get("status"), "display": r.get("display"), "error": error}))
    return 0 if not error else 1


if __name__ == "__main__":
    sys.exit(main())
