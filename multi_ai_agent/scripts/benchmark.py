#!/usr/bin/env python3
"""Benchmark the full multi-agent pipeline in-process (no HTTP server needed).

Examples:
    python scripts/benchmark.py                                   # 3 text tasks
    python scripts/benchmark.py --runs 5 --concurrency 2
    python scripts/benchmark.py --image ./data/input/photo.jpg    # also benchmark the image pipeline
    python scripts/benchmark.py --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.core.container import Container  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.schemas.common import TaskStatus  # noqa: E402
from app.schemas.task import TaskRequest  # noqa: E402

TEXT_PROMPTS = [
    "Summarize: Local inference keeps data on the machine, reduces latency for small models, and avoids per-token "
    "API costs, but it requires managing GPU memory and model files.",
    "Explain why running several large models on one GPU at the same time can cause out-of-memory errors.",
    "Write a haiku about asynchronous programming.",
]


async def run(args: argparse.Namespace) -> None:
    overrides = {"runtime": {"device": args.device}} if args.device else None
    settings = load_settings(args.config, overrides)
    setup_logging("warning")
    container = Container.build(settings, persist_tasks=False)
    await container.start()

    requests: list[tuple[str, TaskRequest]] = []
    for i in range(args.runs):
        requests.append(("text", TaskRequest(prompt=TEXT_PROMPTS[i % len(TEXT_PROMPTS)])))
    if args.image:
        src = Path(args.image).resolve()
        dest = settings.input_dir / src.name
        if src != dest.resolve():
            shutil.copy(src, dest)
        for _ in range(args.runs):
            requests.append(("image", TaskRequest(prompt="Analyze this image and explain what is happening.", image_path=str(dest))))

    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[tuple[str, str, float, str]] = []

    async def one(kind: str, req: TaskRequest) -> None:
        async with semaphore:
            start = time.perf_counter()
            record = await container.run_and_wait(req, timeout=args.timeout)
            rows.append((kind, record.task_id, time.perf_counter() - start, record.status.value))
            if record.status != TaskStatus.COMPLETED:
                print(f"  {record.task_id} {record.status.value}: {record.error}")

    wall = time.perf_counter()
    try:
        await asyncio.gather(*(one(kind, req) for kind, req in requests))
    finally:
        wall = time.perf_counter() - wall
        summary = container.monitor.summary()
        await container.stop()

    print(f"\nDevice: {container.device_info.type} ({container.device_info.name})")
    print(f"{'kind':<6} {'task':<18} {'latency(s)':>10}  status")
    for kind, task_id, latency, status in rows:
        print(f"{kind:<6} {task_id:<18} {latency:>10.2f}  {status}")
    for kind in ("text", "image"):
        lat = [r[2] for r in rows if r[0] == kind]
        if lat:
            print(f"{kind}: mean {statistics.mean(lat):.2f}s  min {min(lat):.2f}s  max {max(lat):.2f}s")
    print(f"wall time: {wall:.2f}s for {len(rows)} tasks (concurrency {args.concurrency})")

    print("\nPer-model inference:")
    for name, stats in summary["models"].items():
        print(
            f"  {name:<16} calls={stats['calls']:<4} avg={stats['avg_latency']:.2f}s "
            f"tok/s={stats['avg_tokens_per_second']} errors={stats['errors']}"
        )
    print("Per-agent:")
    for name, stats in summary["agents"].items():
        print(f"  {name:<12} calls={stats['calls']:<4} avg={stats['avg_latency']:.2f}s errors={stats['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--image", help="Image file to benchmark the vision pipeline with")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--config")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
