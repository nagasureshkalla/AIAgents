"""Command-line launcher: parses options, prints the startup banner and runs uvicorn."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from app.agents.registry import discover_agent_classes
from app.core.config import Settings, load_settings
from app.core.device import DeviceInfo, get_device
from app.models.model_registry import ModelRegistry

WIDTH = 40


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run.py", description="Local Multi-AI Agent server")
    parser.add_argument("--host", help="Bind address (default from config: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Port (default 8000)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="Inference device")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], help="Log level")
    parser.add_argument("--allow-public", action="store_true", help="Permit binding to 0.0.0.0")
    parser.add_argument("--online", action="store_true", help="Disable offline_mode (allow HF Hub access)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    return parser.parse_args(argv)


def apply_args_to_env(args: argparse.Namespace) -> None:
    """Pass CLI options to the app via MAA_* variables so `uvicorn app.main:app` (and --reload) sees them."""
    mapping = {
        "MAA_HOST": args.host,
        "MAA_PORT": args.port,
        "MAA_DEVICE": args.device,
        "MAA_CONFIG": os.path.abspath(args.config) if args.config else None,
        "MAA_LOG_LEVEL": args.log_level,
        "MAA_ALLOW_PUBLIC": "true" if args.allow_public else None,
        "MAA_OFFLINE_MODE": "false" if args.online else None,
    }
    for key, value in mapping.items():
        if value is not None:
            os.environ[key] = str(value)


def render_banner(settings: Settings, device: DeviceInfo, registry: ModelRegistry, agent_names: list[str]) -> str:
    host = settings.server.host
    url = f"http://{host}:{settings.server.port}"
    line = "=" * WIDTH
    out = [line, "LOCAL MULTI AI AGENT".center(WIDTH), line, "", f"Server: {url}", f"API Docs: {url}/docs", ""]
    out.append(f"Device: {device.type.upper()}")
    if device.type == "cuda":
        out.append(f"GPU: {device.name}" + (f" (+{device.gpu_count - 1} more)" if device.gpu_count > 1 else ""))
        out.append(f"VRAM: {device.vram_total_gb} GB (used {device.vram_used_gb} GB)")
    elif device.type == "mps":
        out.append(f"GPU: {device.name}")
    out.append(f"RAM: {device.ram_total_gb} GB")
    out.append(f"Offline mode: {'on' if settings.runtime.offline_mode else 'off'}")

    out += ["", "Models:"]
    for spec in registry.list():
        mark = "✓" if spec.installed else "✗"
        detail = spec.hf_id or str(spec.path)
        suffix = "" if spec.installed else "  (not downloaded)"
        out.append(f"  {mark} {spec.name:<12} {spec.type.value:<10} {detail}{suffix}")

    out += ["", "Agents:"]
    out += [f"  ✓ {name.capitalize()}" for name in agent_names]

    has_text = bool(registry.installed())
    status = "READY" if has_text else "DEGRADED (no models installed: python scripts/download_models.py --all)"
    out += ["", f"Status: {status}", line]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    apply_args_to_env(args)
    settings = load_settings()
    device = get_device(settings.runtime.device)
    registry = ModelRegistry.from_settings(settings)
    agent_names = [c.name for c in discover_agent_classes() if settings.agent_config(c.name).enabled]
    print(render_banner(settings, device, registry, agent_names), flush=True)

    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=args.reload,
        log_level=settings.server.log_level,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
