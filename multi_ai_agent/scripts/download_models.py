#!/usr/bin/env python3
"""Download Hugging Face models into the local models/ directory.

Examples:
    python scripts/download_models.py --all                      # every model in config.yaml with an hf_id
    python scripts/download_models.py --model Qwen/Qwen2.5-0.5B-Instruct --type text
    python scripts/download_models.py --model HuggingFaceTB/SmolVLM-256M-Instruct --type vision --name fast_vision
    python scripts/download_models.py --model openai/whisper-large-v3-turbo --type speech --name whisper
    python scripts/download_models.py --list

This is the only component that contacts the Hugging Face Hub. The server itself loads models from
disk only (offline_mode), and never downloads at startup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# This script is the explicit "go online" step: undo offline flags that may be exported in the shell.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(_var, None)

from app.core.config import Settings, load_settings  # noqa: E402
from app.models.model_registry import ModelRegistry, ModelSpec, ModelType  # noqa: E402

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "tokenizer*",
]


def default_dir_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].lower().replace(".", "_")


def target_for(settings: Settings, model_id: str, model_type: str, name: str | None) -> Path:
    for cfg in settings.models.values():
        if cfg.hf_id == model_id and cfg.type == model_type and name is None:
            return settings.resolve_path(cfg.path)
    return settings.models_dir / model_type / (name or default_dir_name(model_id))


def is_complete(path: Path) -> bool:
    spec = ModelSpec(name=path.name, path=path, type=ModelType.TEXT)
    return spec.installed


def download(
    model_id: str,
    target: Path,
    *,
    revision: str | None,
    force: bool,
    allow_bin: bool,
    allow_code: bool,
    token: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    if is_complete(target) and not force:
        print(f"✓ {model_id} already present at {target} (use --force to re-download)")
        return target
    patterns = list(ALLOW_PATTERNS)
    if allow_bin:
        patterns.append("*.bin")
    if allow_code:
        patterns.append("*.py")
    target.mkdir(parents=True, exist_ok=True)
    print(f"↓ downloading {model_id} -> {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        revision=revision,
        allow_patterns=patterns,
        token=token,
    )
    if not is_complete(target):
        print(
            f"! {model_id}: no safetensors weights found. Retry with --allow-bin if the repo only ships .bin files.",
            file=sys.stderr,
        )
    else:
        print(f"✓ {model_id} ready at {target}")
    return target


def list_models(settings: Settings) -> None:
    registry = ModelRegistry.from_settings(settings)
    for spec in registry.list():
        mark = "✓" if spec.installed else "✗"
        print(f"{mark} {spec.name:<16} {spec.type.value:<11} {spec.hf_id or '-':<42} {spec.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--type", choices=[*(t.value for t in ModelType), "speech"], help="Capability folder (speech = voice input)")
    parser.add_argument("--name", help="Folder name under models/<type>/ (default: derived from the id)")
    parser.add_argument("--revision", help="Branch, tag or commit")
    parser.add_argument("--all", action="store_true", help="Download every model in config.yaml that has an hf_id")
    parser.add_argument("--list", action="store_true", help="Show configured/discovered models and their status")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--allow-bin", action="store_true", help="Also fetch PyTorch .bin weights")
    parser.add_argument("--allow-code", action="store_true", help="Also fetch *.py (needed for trust_remote_code models)")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    token = os.environ.get("HF_TOKEN")

    if args.list:
        list_models(settings)
        return 0

    if args.all:
        failures = 0
        for name, cfg in settings.models.items():
            if not cfg.hf_id:
                print(f"- {name}: no hf_id in config, skipping")
                continue
            try:
                download(
                    cfg.hf_id,
                    settings.resolve_path(cfg.path),
                    revision=args.revision,
                    force=args.force,
                    allow_bin=args.allow_bin,
                    allow_code=args.allow_code or cfg.trust_remote_code,
                    token=token,
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"✗ {name} ({cfg.hf_id}) failed: {exc}", file=sys.stderr)
        return 1 if failures else 0

    if not args.model or not args.type:
        parser.error("--model and --type are required (or use --all / --list)")

    target = target_for(settings, args.model, args.type, args.name)
    download(
        args.model,
        target,
        revision=args.revision,
        force=args.force,
        allow_bin=args.allow_bin,
        allow_code=args.allow_code,
        token=token,
    )
    configured = any(settings.resolve_path(c.path) == target.resolve() for c in settings.models.values())
    if not configured:
        try:
            shown = f"./{target.resolve().relative_to(settings.base_dir)}"
        except ValueError:
            shown = str(target)
        print(
            "\nThe model will be auto-discovered as "
            f"'{target.name}'. To give it a stable key and assign it to an agent, add to config.yaml:\n\n"
            f"models:\n  {target.name}:\n    path: {shown}\n    type: {args.type}\n    hf_id: {args.model}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
