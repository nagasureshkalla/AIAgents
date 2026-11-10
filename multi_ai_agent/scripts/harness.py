#!/usr/bin/env python3
"""Manage the agents' prompt harnesses (app/core/harness.py).

Examples:
    python scripts/harness.py init                      # write harness/<agent>/ from the built-in prompts
    python scripts/harness.py init --agent text --force # reset one agent's harness
    python scripts/harness.py show text --mode summarize  # the system prompt the agent would use now
    python scripts/harness.py rollout text --task harness/tasks/text.json --task-id summarize-incident \\
        --harness-dir /tmp/candidate --out /tmp/run       # one real run, used by harness2's executor

Improving a harness from evidence is done by ../GoogleHrness:
    cd ../../GoogleHrness && .venv/bin/python -m backend.agents improve multi_ai_agent/text --editor ollama
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.base_agent import BaseAgent
from app.core.config import Settings, load_settings
from app.core.harness import seed_tree, write_tree

# Agents whose prompts live in a harness, and the class that owns them.
HARNESSED = ("text", "reasoning", "validation", "aggregator")


def agent_class(name: str) -> type[BaseAgent]:
    from app.agents.reasoning_agent import ReasoningAgent
    from app.agents.text_agent import TextAgent
    from app.agents.validation_agent import ValidationAgent
    from app.orchestration.result_aggregator import ResultAggregator

    classes = {c.name: c for c in (TextAgent, ReasoningAgent, ValidationAgent, ResultAggregator)}
    if name not in classes:
        raise SystemExit(f"unknown agent {name!r}; harnessed agents: {', '.join(HARNESSED)}")
    return classes[name]


def harness_root(settings: Settings) -> Path:
    return settings.resolve_path(settings.harness.root)


def cmd_init(args: argparse.Namespace, settings: Settings) -> int:
    root = harness_root(settings)
    for name in [args.agent] if args.agent else HARNESSED:
        cls = agent_class(name)
        d = root / name
        if d.exists() and not args.force:
            print(f"{d} exists; keep it (use --force to reset to the built-in prompts)")
            continue
        write_tree(seed_tree(name, cls.prompts), d)
        print(f"wrote {d} ({len(cls.prompts)} skills)")
    return 0


def cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    from app.core.harness import HarnessStore

    cls = agent_class(args.agent)
    store = HarnessStore(harness_root(settings), {k: settings.resolve_path(v) for k, v in settings.harness.dirs.items()})
    mode = args.mode or next(iter(cls.prompts))
    print(store.get(args.agent).compose(cls.prompts.get(mode, ""), mode))
    return 0


def load_task(path: Path, task_id: str) -> dict:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    for t in tasks:
        if t["id"] == task_id:
            return t
    raise SystemExit(f"task {task_id!r} not in {path}")


async def _rollout(args: argparse.Namespace, settings: Settings) -> dict:
    from app.core.container import Container
    from app.schemas.agent import (
        AgentMessage,
        AgentResult,
        AgentStatus,
        AgentTask,
        MessageType,
    )

    spec = load_task(Path(args.task), args.task_id)
    container = Container.build(settings, persist_tasks=False)
    await container.start()
    try:
        if args.agent == "aggregator":
            agent: BaseAgent = container.aggregator
        else:
            agent = container.agents.get(args.agent)
        task_id = f"harness-{uuid.uuid4().hex[:8]}"
        mode = spec.get("mode") or next(iter(agent.prompts))
        upstream = [
            AgentMessage(
                task_id=task_id,
                sender=u.get("agent", "text"),
                receiver=agent.name,
                message_type=MessageType.RESULT,
                payload={"step_id": f"u{i}", "text": u["text"]},
            )
            for i, u in enumerate(spec.get("upstream", []))
        ]
        task = AgentTask(
            task_id=task_id,
            step_id="s1",
            agent=agent.name,
            instruction=spec.get("instruction", spec["prompt"]),
            prompt=spec["prompt"],
            params={"mode": mode},
            messages=upstream,
        )
        system = agent.system_prompt(mode)
        t0 = time.perf_counter()
        if args.agent == "aggregator":
            content = [
                (f"u{i}", AgentResult(agent=m.sender, status=AgentStatus.COMPLETED, result={"text": m.payload["text"]}))
                for i, m in enumerate(upstream)
            ]
            try:
                text = await agent._synthesize(task, content, spec.get("issues", []))
                result = {"ok": True, "payload": {"text": text}, "error": None}
            except Exception as exc:  # noqa: BLE001 - recorded as the rollout error
                result = {"ok": False, "payload": {}, "error": str(exc)}
        else:
            res = await agent.execute(task)
            result = {"ok": res.ok, "payload": res.result, "error": res.error}
        result.update(
            agent=agent.name, mode=mode, model=agent.model_key, system=system, seconds=round(time.perf_counter() - t0, 2)
        )
        return result
    finally:
        await container.stop()


def cmd_rollout(args: argparse.Namespace, settings: Settings) -> int:
    agent_class(args.agent)
    settings.harness.enabled = True
    settings.harness.dirs[args.agent] = str(Path(args.harness_dir).resolve())
    settings.runtime.preload = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_task(Path(args.task), args.task_id)
    result = asyncio.run(_rollout(args, settings))
    payload = result["payload"]
    answer = payload.get("text") if isinstance(payload.get("text"), str) else json.dumps(payload, indent=2, default=str)
    lines = [
        f"[ACTION] {result['agent']} agent, mode={result['mode']}, model={result['model']}",
        "[OBSERVATION] system prompt in effect:",
        result["system"],
        "[ACTION] user request:",
        spec["prompt"],
    ]
    for u in spec.get("upstream", []):
        lines += [f"[OBSERVATION] upstream {u.get('agent', 'text')} agent output:", u["text"]]
    if result["ok"]:
        lines += [f"[OBSERVATION] reply after {result['seconds']}s:", answer]
        (out / "answer.txt").write_text(answer, encoding="utf-8")
    else:
        lines += ["[ERROR]", str(result["error"])]
    (out / "trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "seconds": result["seconds"], "error": result["error"]}))
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config.yaml to load (default: MAA_CONFIG or ./config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write harness/<agent>/ from the built-in prompts")
    p.add_argument("--agent", choices=HARNESSED)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("show", help="print the composed system prompt for an agent")
    p.add_argument("agent", choices=HARNESSED)
    p.add_argument("--mode")
    p = sub.add_parser("rollout", help="run one task with a candidate harness (harness2 executor)")
    p.add_argument("agent", choices=HARNESSED)
    p.add_argument("--task", required=True, help="JSON list of tasks (harness/tasks/<agent>.json)")
    p.add_argument("--task-id", required=True)
    p.add_argument("--harness-dir", required=True)
    p.add_argument("--out", required=True, help="run directory: answer.txt, trajectory.md, result.json")
    args = parser.parse_args()
    settings = load_settings(args.config)
    return {"init": cmd_init, "show": cmd_show, "rollout": cmd_rollout}[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
