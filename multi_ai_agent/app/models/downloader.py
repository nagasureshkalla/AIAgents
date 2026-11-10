"""Background model downloads from the Hugging Face Hub (model repository only, never inference).

Runs scripts/download_models.py in a subprocess so the server process itself stays in offline mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}$")
_NAME = re.compile(r"^[\w.-]{1,64}$")
_PERCENT = re.compile(r"(\d{1,3})%")


@dataclass
class DownloadJob:
    id: str
    hf_id: str
    type: str
    name: str | None
    status: str = "running"  # running | completed | failed | cancelled
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: float | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=40))
    error: str | None = None
    process: asyncio.subprocess.Process | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hf_id": self.hf_id,
            "type": self.type,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log": list(self.log)[-8:],
        }


class DownloadManager:
    def __init__(self, on_complete: Callable[[], object], config_path: str | None = None) -> None:
        self.jobs: dict[str, DownloadJob] = {}
        self._on_complete = on_complete
        self._config_path = config_path
        self._tasks: set[asyncio.Task] = set()

    def start(self, hf_id: str, model_type: str, name: str | None = None) -> DownloadJob:
        if not _REPO_ID.match(hf_id or ""):
            raise InvalidRequestError(f"Invalid Hugging Face model id {hf_id!r} (expected 'owner/name')")
        if model_type not in ("text", "vision", "multimodal"):
            raise InvalidRequestError("type must be text, vision or multimodal")
        if name is not None and not _NAME.match(name):
            raise InvalidRequestError("name may only contain letters, digits, '.', '_' and '-'")
        if any(j.status == "running" and j.hf_id == hf_id for j in self.jobs.values()):
            raise InvalidRequestError(f"{hf_id} is already downloading")
        job = DownloadJob(id=uuid.uuid4().hex[:12], hf_id=hf_id, type=model_type, name=name)
        self.jobs[job.id] = job
        task = asyncio.create_task(self._run(job), name=f"download:{hf_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: DownloadJob) -> None:
        args = [sys.executable, str(SCRIPT), "--model", job.hf_id, "--type", job.type]
        if job.name:
            args += ["--name", job.name]
        if self._config_path:
            args += ["--config", self._config_path]
        env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            job.process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
            )
            assert job.process.stdout is not None
            buf = b""
            while chunk := await job.process.stdout.read(4096):
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    job.log.append(line[:300])
                    if m := _PERCENT.search(line):
                        job.progress = min(100.0, float(m.group(1)))
            code = await job.process.wait()
            job.status = "completed" if code == 0 else "failed"
            if code != 0:
                job.error = (job.log[-1] if job.log else f"exit code {code}")[:300]
        except asyncio.CancelledError:
            job.status = "cancelled"
            if job.process and job.process.returncode is None:
                job.process.kill()
            raise
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", str(exc)
        finally:
            job.finished_at = time.time()
            if job.status == "completed":
                job.progress = 100.0
                with contextlib.suppress(Exception):
                    self._on_complete()

    def list(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started_at)]

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
