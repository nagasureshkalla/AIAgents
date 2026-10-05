"""Structured logging with task/agent context carried through contextvars."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any

task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)
agent_var: ContextVar[str | None] = ContextVar("agent", default=None)

_STRUCTURED_FIELDS = ("event", "duration", "status", "model", "device", "step")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "task_id", None) is None:
            record.task_id = task_id_var.get()
        if getattr(record, "agent", None) is None:
            record.agent = agent_var.get()
        return True


class KeyValueFormatter(logging.Formatter):
    """[2026-10-04 14:20:01] level=INFO task=abc agent=vision event=completed duration=1.21s msg"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{ts}]", f"level={record.levelname}", f"logger={record.name}"]
        if record.task_id:
            parts.append(f"task={record.task_id}")
        if record.agent:
            parts.append(f"agent={record.agent}")
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is None:
                continue
            if field == "duration" and isinstance(value, (int, float)):
                value = f"{value:.2f}s"
            parts.append(f"{field}={value}")
        message = record.getMessage()
        if message:
            parts.append(message)
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "task_id": record.task_id,
            "agent": record.agent,
            "message": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "info", json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else KeyValueFormatter())
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def bind_context(task_id: str | None = None, agent: str | None = None) -> list[Token]:
    tokens: list[Token] = []
    if task_id is not None:
        tokens.append(task_id_var.set(task_id))
    if agent is not None:
        tokens.append(agent_var.set(agent))
    return tokens


def reset_context(tokens: list[Token]) -> None:
    for token in reversed(tokens):
        token.var.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    task_id: str | None = None,
    agent: str | None = None,
    duration: float | None = None,
    status: str | None = None,
    message: str = "",
    **fields: Any,
) -> None:
    extra: dict[str, Any] = {"event": event, "duration": duration, "status": status}
    if task_id is not None:
        extra["task_id"] = task_id
    if agent is not None:
        extra["agent"] = agent
    for key in _STRUCTURED_FIELDS:
        if key in fields:
            extra[key] = fields.pop(key)
    if fields:
        message = (message + " " if message else "") + " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, message, extra=extra)
