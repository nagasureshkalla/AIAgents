"""AirLLM agent: answers with gpt-oss-120b, a model far larger than the GPU. Only used when asked for by name
("ask gpt-oss ...", "use airllm ...", "the 120b model").

Two engines (airllm.engine):
- "ollama" (default, fast): the model is a normal Ollama model (agents.airllm.model -> config `gpt_oss`), loaded by
  the ModelManager like any other; Ollama keeps the part that does not fit in VRAM in system RAM.
- "airllm" (fallback when the Ollama model is not pulled): the AirLLM service (~/Desktop/AI/AirLLM,
  github.com/lyogavin/airllm) streams the model through the GPU one layer at a time: a few GB of VRAM, but every
  token reads all layers again (~40 s per token on an RTX 3090).
  FREE GPU -> START the service -> LOAD (first time: split the checkpoint per layer) -> GENERATE (streamed) -> ANSWER
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agents.base_agent import BaseAgent
from app.agents.job_agent import _is_loopback
from app.core.config import AirLLMConfig, Settings
from app.core.exceptions import AgentSystemError
from app.schemas.agent import AgentTask

logger = logging.getLogger(__name__)

# "ask gpt-oss-120b:", "using airllm," ... at the start of the request is routing, not part of the question.
ROUTING_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:ask|use|using|with|via|let)\s+)?(?:the\s+)?(?:openai\s+)?"
    r"(?:gpt[- ]?oss(?:[- ]?120b)?|air ?llm|120\s?b(?:illion)?(?:[- ]parameters?)?\s+model|"
    r"(?:big|biggest|large|largest|huge)\s+(?:local\s+)?model)\s*(?:model\s*)?(?:to\s+)?[:,\-–]?\s*", re.I)


class AirLLMError(AgentSystemError):
    code = "airllm_failed"
    error_type = "airllm_failed"


def question_of(prompt: str) -> str:
    rest = ROUTING_PREFIX_RE.sub("", prompt, count=1).strip()
    return rest or prompt.strip()


class AirLLMClient:
    """HTTP client for the AirLLM service, plus starting/stopping it."""

    _proc: subprocess.Popen[bytes] | None = None  # the service this process started (shared by all tasks)

    def __init__(self, cfg: AirLLMConfig, settings: Settings) -> None:
        if not _is_loopback(cfg.url):
            raise AirLLMError(f"airllm.url must point at this computer, not {cfg.url}")
        self.cfg = cfg
        self.settings = settings
        self.url = cfg.url.rstrip("/")

    @property
    def project_dir(self) -> Path:
        return self.settings.resolve_path(self.cfg.project_dir)

    def _log_path(self) -> Path:
        return self.settings.output_dir / "airllm" / "service.log"

    async def health(self) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.url}/health")
                return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def ensure_running(self) -> dict[str, Any]:
        status = await self.health()
        if status:
            return status
        if not self.cfg.autostart:
            raise AirLLMError(f"The AirLLM service is not running at {self.url} (airllm.autostart is off)")
        self._start()
        log = self._log_path()
        deadline = time.monotonic() + self.cfg.startup_timeout
        while time.monotonic() < deadline:
            status = await self.health()
            if status:
                return status
            if AirLLMClient._proc.poll() is not None:
                break
            await asyncio.sleep(1)
        raise AirLLMError(f"The AirLLM service did not start; see {log}")

    def _start(self) -> None:
        python = self.project_dir / ".venv" / "bin" / "python"
        server = self.project_dir / "server.py"
        if not python.exists() or not server.exists():
            raise AirLLMError(f"AirLLM service not installed at {self.project_dir} (airllm.project_dir)")
        log = self._log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "AIRLLM_PORT": str(urlparse(self.url).port or 8770)}
        logger.info("starting the AirLLM service from %s", self.project_dir)
        with log.open("ab") as fh:
            AirLLMClient._proc = subprocess.Popen([str(python), str(server)], cwd=self.project_dir, env=env,
                                                  stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    async def load(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as c:
            return (await c.post(f"{self.url}/load")).json()

    async def generate(self, messages: list[dict[str, str]], max_new_tokens: int, reasoning: str):
        """Yields the service's events: token ... done | error."""
        body = {"messages": messages, "max_new_tokens": max_new_tokens, "reasoning": reasoning}
        timeout = httpx.Timeout(connect=10, read=None, write=30, pool=10)  # a token can take a long time
        async with httpx.AsyncClient(timeout=timeout) as c, c.stream("POST", f"{self.url}/generate", json=body) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode(errors="replace")
                with contextlib.suppress(ValueError):
                    detail = json.loads(detail).get("detail", detail)
                raise AirLLMError(f"AirLLM: {detail}")
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    @staticmethod
    def shutdown() -> None:
        proc = AirLLMClient._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, 15)
                proc.wait(timeout=20)
        AirLLMClient._proc = None


atexit.register(AirLLMClient.shutdown)


class AirLLMAgent(BaseAgent):
    name = "airllm"
    description = ("Answers with gpt-oss-120b (a 120B-parameter model) streamed layer by layer through the GPU by "
                   "AirLLM. Strongest local model, but slow: seconds per token. Ask for it by name.")
    capabilities = ["big_model_generation"]
    default_model = "gpt_oss"  # Ollama engine; the AirLLM service holds its own copy
    client_factory = AirLLMClient  # tests swap in a fake service

    def _ollama_model(self) -> str | None:
        """The Ollama model to use, or None when the engine is AirLLM or the model is not pulled."""
        if self.ctx.settings.airllm.engine != "ollama" or not self.model_key:
            return None
        try:
            spec = self.ctx.models.registry.get(self.model_key)
        except Exception:  # noqa: BLE001 - not configured: use AirLLM
            return None
        return self.model_key if spec.is_ollama and spec.installed else None

    def required_models(self) -> list[str]:
        model = self._ollama_model()
        return [model] if model else []  # the AirLLM service is not a ModelManager model

    async def run(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.ctx.settings.airllm
        if not cfg.enabled:
            raise AirLLMError("The AirLLM agent is disabled (airllm.enabled: false)")
        model = self._ollama_model()
        if model:
            return await self._run_ollama(task, model)
        if cfg.engine == "ollama":
            self.progress(task, f"gpt-oss is not in Ollama (ollama pull {self._tag()}): using AirLLM streaming instead")
        return await self._run_airllm(task)

    def _tag(self) -> str:
        with contextlib.suppress(Exception):
            return self.ctx.models.registry.get(self.model_key or "").ollama_model or "gpt-oss:120b"
        return "gpt-oss:120b"

    async def _run_ollama(self, task: AgentTask, model: str) -> dict[str, Any]:
        cfg = self.ctx.settings.airllm
        self.progress(task, f"Generating with {self._tag()} in Ollama (GPU + system RAM)")
        started = time.monotonic()
        out = await self.generate(task, question_of(task.prompt), model_key=model, max_tokens=cfg.ollama_max_new_tokens,
                                  temperature=0.2)
        seconds = time.monotonic() - started
        answer = out.text or "(no answer: the model used its whole budget reasoning; raise airllm.ollama_max_new_tokens)"
        tokens = out.completion_tokens
        speed = f" · {tokens / seconds:.1f} tokens/s" if tokens and seconds else ""
        footer = f"\n\n*{self._tag()} via Ollama · {tokens} tokens in {seconds:.0f}s{speed}*"
        return {"text": answer + footer, "answer": answer, "model": self._tag(), "engine": "ollama",
                "tokens": tokens, "seconds": round(seconds, 1)}

    async def _run_airllm(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.ctx.settings.airllm
        client = self.client_factory(cfg, self.ctx.settings)

        if cfg.free_gpu:
            self.progress(task, "Freeing GPU memory for the layer-by-layer model")
            freed = await self.ctx.models.prepare([], policy="exclusive")
            if freed.get("unloaded"):
                self.progress(task, f"Unloaded {', '.join(freed['unloaded'])}")

        self.progress(task, "Starting the AirLLM service")
        status = await client.ensure_running()
        if not status.get("installed"):
            raise AirLLMError(f"The model is not downloaded yet ({status.get('model_path')})")
        if not status.get("loaded"):
            first = not status.get("layers_split")
            self.progress(task, f"Loading {status.get('model')}" + (
                " — first run: splitting the checkpoint into per-layer files (one-time, can take 30+ minutes)"
                if first else ""))
            status = await client.load()
            deadline = time.monotonic() + cfg.load_timeout
            while not status.get("loaded"):
                if status.get("error") and not status.get("loading"):
                    raise AirLLMError(f"AirLLM could not load the model: {status['error']}")
                if time.monotonic() > deadline:
                    raise AirLLMError("Loading the model timed out (airllm.load_timeout)")
                await asyncio.sleep(5)
                status = await client.health() or {}

        question = question_of(task.prompt)
        self.progress(task, f"Generating with {status.get('model')} (reasoning: {cfg.reasoning})")
        done: dict[str, Any] | None = None
        count, last = 0, 0.0
        async for event in client.generate([{"role": "user", "content": question}], cfg.max_new_tokens, cfg.reasoning):
            if event["type"] == "token":
                count += 1
                if time.monotonic() - last > 3:  # one progress line every few seconds, not per token
                    last = time.monotonic()
                    phase = "thinking" if event.get("channel") == "analysis" else "writing the answer"
                    self.progress(task, f"{phase}… {count} tokens")
            elif event["type"] == "error":
                raise AirLLMError(f"AirLLM generation failed: {event['message']}")
            elif event["type"] == "done":
                done = event
        if done is None:
            raise AirLLMError("The AirLLM service closed the stream without an answer")
        if cfg.stop_after_task:
            AirLLMClient.shutdown()

        answer = done.get("answer") or "(the model used its whole token budget thinking; raise airllm.max_new_tokens)"
        tokens, seconds = done.get("tokens") or count, done.get("seconds") or 0
        speed = f"{tokens / seconds:.2f} tokens/s" if seconds else ""
        footer = f"\n\n*{status.get('model')} via AirLLM · {tokens} tokens in {seconds:.0f}s" + (f" · {speed}" if speed else "") + "*"
        return {"text": answer + footer, "answer": answer, "reasoning": done.get("reasoning"), "model": status.get("model"),
                "engine": "airllm", "tokens": tokens, "seconds": seconds}
