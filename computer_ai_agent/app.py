"""Computer Master Agent entry point.

    python app.py                                   # web control panel on http://127.0.0.1:8765
    python app.py --command "Open Calculator and calculate 25 * 89"   # run one task in the terminal, no web UI
    python app.py --display :7 --command "..."      # drive another X display (e.g. a nested Xephyr test screen)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUA_SESSION_DISPLAY", os.environ.get("DISPLAY", ""))  # the user's own screen

from config.settings import load_settings  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("computer_agent")

COLORS = {"ok": "\033[32m", "bad": "\033[31m", "warn": "\033[33m", "dim": "\033[2m", "bold": "\033[1m", "end": "\033[0m"}


def _c(kind: str, text: str) -> str:
    return f"{COLORS[kind]}{text}{COLORS['end']}" if sys.stdout.isatty() else text


def print_event(e: dict) -> None:
    ev = e["event"]
    if ev in ("screen_updated", "observation", "verification_started", "log"):
        return
    agent = e.get("agent", "")
    if ev == "state_changed":
        print(_c("dim", f"  [{agent}] {e['old']} → {e['state']}" + (f"  ({e['detail'][:70]})" if e.get("detail") else "")))
    elif ev == "plan_created":
        print(_c("bold", f"Plan ({e['source']}):"))
        for s in e["subtasks"]:
            print(f"  {s['id']}. [{s['app']}] {s['instruction'] or '(open)'}")
    elif ev == "application_detected":
        print(f"  application: {e['application']} ({'running' if e['running'] else 'not running'})")
    elif ev == "application_opened":
        print(f"  {'launched' if e['launched'] else 'focused'} {e['application']}: {e['window']['title']}")
    elif ev == "agent_created":
        print(_c("bold", f"  created {e['agent']}") + f" plugin={e['plugin']} ui={e['ui_summary']['element_counts']}")
    elif ev == "agent_message":
        m = e["message"]
        body = m["task"] or f"{m['status']}: {m['result']}"
        print(_c("dim", f"  ✉ {m['from']} → {m['to']}: {body[:140]}"))
    elif ev == "action_started":
        why = f"  — {e['reason'][:90]}" if e.get("reason") else ""
        print(f"  ▶ step {e['step']}: {e['label']}{_c('dim', why)}")
    elif ev == "action_completed":
        mark = _c("ok", "✓") if e["ok"] else _c("warn", "✗")
        print(f"    {mark} {e.get('message', '')[:120]}" + _c("dim", f"  ({e.get('duration_ms', 0)} ms)"))
    elif ev == "verification_completed" and e.get("what") == "goal":
        print(f"  goal check: {'complete' if e['ok'] else 'NOT complete'} — {e['detail'][:140]}")
    elif ev == "recovery":
        print(_c("warn", f"  ↻ recovering: {e['reason'][:140]}"))
    elif ev in ("task_completed", "task_failed", "task_stopped"):
        kind = "ok" if ev == "task_completed" else "bad"
        print(_c(kind, f"\n{ev.replace('_', ' ').upper()} in {e['duration']}s ({e['actions']} actions)"))
        print(textwrap.fill(e["result"] or "", 100))


async def run_cli(command: str, settings) -> int:  # noqa: ANN001
    from agents.master_agent import TaskManager
    from core.runtime import build_runtime

    rt = build_runtime(settings)
    health = await rt.models.health()
    if not health["ok"]:
        print(_c("bad", f"Model not ready: {health}"))
        return 2
    rt.bus.on(print_event)
    tm = TaskManager(rt)
    q = rt.bus.subscribe()

    async def answer_prompts() -> None:
        while True:
            e = await q.get()
            if e["event"] not in ("confirmation_required", "user_input_required"):
                continue
            req = e["request"]
            print(_c("warn", f"\n? {req['message']}"))
            if req["choices"]:
                for i, c in enumerate(req["choices"], 1):
                    print(f"   {i}. {c}")
            if not sys.stdin.isatty():  # nobody to ask: deny / stop (safe default)
                print(_c("dim", "   (no terminal for input: answering with the safe default)"))
                rt.broker.respond(req["id"], False if req["kind"] == "confirm" else
                                  next((c for c in req["choices"] if c.lower() in ("stop", "cancel", "deny")), None))
                continue
            try:
                raw = (await asyncio.to_thread(input, "> ")).strip()
            except EOFError:
                raw = ""
            if req["kind"] == "confirm":
                ans = raw.lower() in ("1", "y", "yes", "approve")
            elif req["choices"] and raw.isdigit() and 1 <= int(raw) <= len(req["choices"]):
                ans = req["choices"][int(raw) - 1]
            else:
                ans = raw
            rt.broker.respond(req["id"], ans)

    prompter = asyncio.create_task(answer_prompts())
    print(_c("bold", f"Command: {command}") + _c("dim", f"   (model {settings.model_name}, display {rt.desktop.display})"))
    handle = await tm.run_and_wait(command)
    prompter.cancel()
    return 0 if handle.status == "completed" else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Local AI computer-use multi-agent system")
    ap.add_argument("--command", "-c", help="run one task in the terminal and exit (no web UI)")
    ap.add_argument("--display", help="X display to control, e.g. :0 (default: $DISPLAY)")
    ap.add_argument("--host", help="web UI host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="web UI port (default 8765)")
    ap.add_argument("--debug", action="store_true", help="record screenshots/OCR/decisions in logs/tasks/<id>/")
    ap.add_argument("--no-confirm", action="store_true", help="do not ask before risky actions (NOT recommended)")
    args = ap.parse_args()

    settings = load_settings()
    if args.display:
        settings.display = args.display
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.debug:
        settings.debug = True
    if args.no_confirm:
        settings.require_confirmation = False
    setup_logging(settings.logs_dir, settings.debug)
    if not args.command:
        logging.getLogger().handlers[-1].setLevel(logging.WARNING if not settings.debug else logging.DEBUG)

    if args.command:
        logging.getLogger().handlers[-1].setLevel(logging.WARNING)  # the CLI prints events itself
        sys.exit(asyncio.run(run_cli(args.command, settings)))

    import uvicorn

    from api.routes import create_app

    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        print(_c("warn", f"WARNING: the control panel can move your mouse and type; it is listening on {settings.host}"))
    print(_c("bold", f"Computer Agent control panel: http://{settings.host}:{settings.port}"))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
