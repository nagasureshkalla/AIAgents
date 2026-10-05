"""Filesystem helpers with path sanitization."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.core.exceptions import InvalidRequestError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_join(root: Path, user_path: str | Path) -> Path:
    """Resolve `user_path` under `root`, rejecting traversal and symlinks that escape it."""
    root = root.resolve()
    candidate = Path(user_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise InvalidRequestError(f"Path '{user_path}' is outside the allowed directory")
    return resolved


def resolve_input_path(user_path: str, input_dir: Path, project_root: Path) -> Path:
    """Accept './data/input/x.png' (project-relative), 'x.png' (input-relative) or an absolute path in input_dir."""
    if "\x00" in user_path:
        raise InvalidRequestError("Invalid characters in path")
    input_dir = input_dir.resolve()
    raw = Path(user_path)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [project_root / raw, input_dir / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(input_dir) and resolved.is_file():
            return resolved
    for candidate in candidates:
        if not candidate.resolve().is_relative_to(input_dir):
            continue
        raise InvalidRequestError(f"Image not found: {user_path}")
    raise InvalidRequestError(f"Image path must be inside {input_dir} (got '{user_path}')")


def sanitize_filename(name: str, default: str = "upload") -> str:
    base = Path(name or default).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._")
    return cleaned[:100] or default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
