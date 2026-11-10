"""Control Panel agent, the Ollama engine, and requirement-driven model loading/unloading by the manager."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from app.agents.control_panel_agent import (
    PENDING_REQUESTS,
    ControlPanelAgent,
    ControlPanelClient,
    ControlPanelError,
)
from app.agents.manager_agent import ManagerAgent
from app.core.config import Settings
from app.models import model_registry
from app.models.base import GenerationRequest
from app.models.engines import ollama_engine
from app.models.model_manager import ModelState
from app.schemas.common import TaskStatus
from app.schemas.task import TaskRequest
from tests.fakes import (
    CUDA_DEVICE,
    FakeFactory,
    FakeGPU,
    build_container,
    make_settings,
)


# ---------------------------------------------------------------- routing
@pytest.mark.parametrize("prompt", [
    "Open Chrome, go to Google, search for NVIDIA RTX 3090, and open the first result.",
    "Open Calculator and calculate 25 * 89",
    "open vs code and create a python project called testapp",
    "In Chrome, go to youtube.com",
    "create main.py in VS Code",
    "click the Save button",
    "Close Spotify",
])
def test_desktop_requests_route_to_control_panel(prompt):
    assert ManagerAgent.classify_intent(prompt, False) == "computer_control"


@pytest.mark.parametrize(("prompt", "intent"), [
    ("search for linkedin posts of username", "research"),
    ("Explain memory leaks in Chrome", "explain"),
    ("Write a LinkedIn post about my promotion", "generate"),
    ("open a discussion about AI safety in an essay", "generate"),
    ("Open-source models are great, summarize why", "summarize"),
    ("Start a story about a dragon", "generate"),
])
def test_non_desktop_requests_do_not(prompt, intent):
    assert ManagerAgent.classify_intent(prompt, False) == intent


# ---------------------------------------------------------------- manager: load/unload by requirement
def _sized(container, sizes: dict[str, float]) -> None:
    for name, gb in sizes.items():
        container.models.registry.get(name).est_size_gb = gb


async def test_prepare_unloads_only_what_is_needed_for_room(tmp_path):
    gpu = FakeGPU(24.0)
    c = build_container(make_settings(tmp_path), factory=FakeFactory(gpu=gpu), device=CUDA_DEVICE)
    await c.start()
    try:
        _sized(c, {"base": 15.0, "fast_text": 1.0, "vision": 18.0})
        await c.models.load_model("fast_text")  # least recently used
        await c.models.load_model("base")
        report = await c.models.prepare(["vision"])
        # 1 + 15 + 18 > 24 * 0.85: unloading base alone makes room, so the small fast_text stays even though older
        assert report["unloaded"] == ["base"] and report["loaded"] == ["vision"]
        assert c.models.is_loaded("fast_text") and c.models.is_loaded("vision") and not c.models.is_loaded("base")
        # nothing to do when the requirement is already resident
        again = await c.models.prepare(["vision", "fast_text"])
        assert again["unloaded"] == [] and again["loaded"] == [] and set(again["resident"]) == {"vision", "fast_text"}
    finally:
        await c.stop()


async def test_prepare_exclusive_policy_and_active_models(tmp_path):
    gpu = FakeGPU(24.0)
    c = build_container(make_settings(tmp_path), factory=FakeFactory(gpu=gpu), device=CUDA_DEVICE)
    await c.start()
    try:
        await c.models.load_model("fast_text")
        key, _ = await c.models.acquire("base")  # running inference for another task
        report = await c.models.prepare(["vision"], policy="exclusive")
        assert "fast_text" in report["unloaded"] and "base" not in report["unloaded"]
        assert c.models.is_loaded("base")  # never unload a model that is in use
        c.models.release(key)
    finally:
        await c.stop()


async def test_required_models_are_protected_while_loading(tmp_path):
    gpu = FakeGPU(24.0)
    c = build_container(make_settings(tmp_path), factory=FakeFactory(gpu=gpu), device=CUDA_DEVICE)
    await c.start()
    try:
        _sized(c, {"base": 10.0, "vision": 10.0})
        report = await c.models.prepare(["base", "vision"])
        assert set(report["resident"]) == {"base", "vision"}  # loading vision did not evict base
    finally:
        await c.stop()


async def test_manager_decision_reports_model_preparation(container):
    record = await container.run_and_wait(TaskRequest(prompt="Summarize this: a short text about cats."), timeout=20)
    assert record.status == TaskStatus.COMPLETED
    assert "fast_text" in record.decision["models"]["required"]


# ---------------------------------------------------------------- Ollama engine (fake Ollama server)
class FakeOllama:
    def __init__(self) -> None:
        self.tags = {"qwen3.6:27b": {"name": "qwen3.6:27b", "size": 17 * 1024**3,
                                    "details": {"family": "qwen35", "quantization_level": "Q4_K_M"}}}
        self.loaded: dict[str, int] = {}
        self.chats: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": list(self.tags.values())})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n, "size_vram": v} for n, v in self.loaded.items()]})
        if request.url.path == "/api/generate":
            if body.get("keep_alive") == 0:
                self.loaded.pop(body["model"], None)
                return httpx.Response(200, json={"done_reason": "unload"})
            self.loaded[body["model"]] = int(17.4 * 1024**3)
            return httpx.Response(200, json={"done_reason": "load"})
        if request.url.path == "/api/chat":
            self.chats.append(body)
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}, "prompt_eval_count": 12,
                                             "eval_count": 4})
        return httpx.Response(404)


@pytest.fixture
def fake_ollama(monkeypatch):
    server = FakeOllama()
    transport = httpx.MockTransport(server.handler)
    real_client = httpx.Client
    monkeypatch.setattr(ollama_engine.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    monkeypatch.setattr(ollama_engine.httpx, "get",
                        lambda url, **kw: real_client(transport=transport).get(url, **kw))
    ollama_engine._tags_cache.clear()
    return server


def _ollama_settings(tmp_path) -> Settings:
    s = make_settings(tmp_path)
    data = s.model_dump()
    data["models"]["control_panel"] = {"engine": "ollama", "ollama_model": "qwen3.6:27b", "type": "multimodal",
                                       "est_size_gb": 18, "model_kwargs": {"num_ctx": 16384}}
    data["agents"]["control_panel"] = {"model": "control_panel"}
    return Settings.model_validate(data)


def test_ollama_engine_load_generate_unload(tmp_path, fake_ollama):
    settings = _ollama_settings(tmp_path)
    spec = model_registry.ModelRegistry.from_settings(settings).get("control_panel")
    assert spec.is_ollama and spec.installed and spec.format == "ollama" and spec.estimated_memory_gb() == 18
    model = ollama_engine.OllamaModel(spec, "cuda:0", settings)
    model.load()
    assert "qwen3.6:27b" in fake_ollama.loaded and model.memory_gb == pytest.approx(17.4)
    out = model.generate_sync(GenerationRequest(messages=[{"role": "user", "content": "hi"}],
                                                image=Image.new("RGB", (8, 8))))
    assert out.text == '{"ok": true}' and out.completion_tokens == 4
    sent = fake_ollama.chats[-1]
    assert sent["keep_alive"] == "-1m" and sent["options"]["num_ctx"] == 16384 and sent["messages"][-1]["images"]
    model.unload()
    assert "qwen3.6:27b" not in fake_ollama.loaded


def test_ollama_model_missing_gives_pull_hint(tmp_path, fake_ollama):
    fake_ollama.tags.clear()
    fake_ollama.tags["other:1b"] = {"name": "other:1b", "size": 1}
    settings = _ollama_settings(tmp_path)
    spec = model_registry.ModelRegistry.from_settings(settings).get("control_panel")
    assert not spec.installed
    with pytest.raises(Exception, match="ollama pull qwen3.6:27b"):
        ollama_engine.OllamaModel(spec, "cuda:0", settings).load()


# ---------------------------------------------------------------- Control Panel agent end to end
class FakeControlClient:
    """Stands in for the computer_ai_agent service: scripted events, records answers."""

    script: list[dict[str, Any]] = []
    answers: dict[str, Any] = {}
    seen_state: list[str] = []
    models = None
    stopped = False
    block = False
    service_stops = 0

    def __init__(self, cfg, settings) -> None:
        self.url = "http://127.0.0.1:8765"

    async def stop_service(self) -> None:
        FakeControlClient.service_stops += 1

    async def ensure_running(self, model_tag: str, num_ctx: int = 16384) -> dict:
        FakeControlClient.seen_state.append(f"{model_tag}:{num_ctx}")
        return {}

    async def run(self, command: str, on_event) -> dict:
        FakeControlClient.seen_state.append(FakeControlClient.models.state("control_panel").value)
        await on_event({"event": "task_accepted", "task_id": "cp1"})
        try:
            for e in FakeControlClient.script:
                e = {"task_id": "cp1", **e}
                await on_event(e)
                if e["event"] == "confirmation_required":
                    for _ in range(100):  # wait for the user's answer, like the real service
                        if e["request"]["id"] in FakeControlClient.answers:
                            break
                        await asyncio.sleep(0.02)
                if e["event"].startswith("task_"):
                    return e
            while FakeControlClient.block:
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            FakeControlClient.stopped = True
            raise
        return {"event": "task_completed", "task_id": "cp1", "result": "done", "actions": 0, "duration": 0.1}


@pytest.fixture
async def cp_container(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama_engine, "ollama_installed", lambda url, tag: True)
    monkeypatch.setattr(ControlPanelAgent, "client_factory", FakeControlClient)
    FakeControlClient.answers, FakeControlClient.seen_state = {}, []
    FakeControlClient.stopped, FakeControlClient.block, FakeControlClient.service_stops = False, False, 0
    c = build_container(_ollama_settings(tmp_path), factory=FakeFactory(), device=CUDA_DEVICE)
    FakeControlClient.models = c.models
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


async def test_control_panel_runs_desktop_task_with_its_model_held(cp_container):
    FakeControlClient.script = [
        {"event": "plan_created", "subtasks": [{"app": "calculator", "instruction": "calculate 25*89"}]},
        {"event": "application_opened", "application": "Calculator", "launched": False},
        {"event": "action_started", "step": 1, "label": "type_text \"25*89\n\""},
        {"event": "action_completed", "step": 1, "label": "type_text \"25*89\n\"", "ok": True,
         "message": "text is in the focused field", "agent": "CalculatorSubAgent", "duration_ms": 30},
        {"event": "task_completed", "result": "2225", "actions": 1, "duration": 9.5},
    ]
    record = await cp_container.run_and_wait(TaskRequest(prompt="Open Calculator and calculate 25 * 89"), timeout=20)
    assert record.status == TaskStatus.COMPLETED, record.error
    assert record.decision["intent"] == "computer_control"
    assert record.decision["models"]["required"] == ["control_panel"]
    assert [s.agent for s in record.plan.steps] == ["control_panel"]
    assert record.result["answer"] == "2225"
    res = record.agent_results["control"]
    assert res.models == ["control_panel"] and res.result["action_count"] == 1
    # the service was started for the managed model with the managed context, and the model was ACTIVE meanwhile
    assert FakeControlClient.seen_state == ["qwen3.6:27b:16384", ModelState.ACTIVE.value]
    assert cp_container.models.state("control_panel") == ModelState.IDLE  # released, evictable for other agents
    progress = [e for e in cp_container.events.history(record.task_id) if e.event.value == "agent_progress"]
    assert any("type_text" in (e.data.get("message") or "") for e in progress)


async def test_control_panel_forwards_confirmations(cp_container):
    from fastapi.testclient import (
        TestClient,  # noqa: F401 - API path tested via the route function below
    )

    req = {"id": "r1", "kind": "confirm", "message": "Clicking \"Delete\" may be irreversible. Continue?",
           "choices": ["Approve", "Deny"]}
    FakeControlClient.script = [{"event": "confirmation_required", "request": req},
                                {"event": "user_response", "request_id": "r1", "answer": "approve"},
                                {"event": "task_completed", "result": "deleted", "actions": 1, "duration": 1}]

    async def approve_when_asked() -> None:
        for _ in range(200):
            if "r1" in PENDING_REQUESTS:
                FakeControlClient.answers["r1"] = "approve"
                return
            await asyncio.sleep(0.02)

    helper = asyncio.create_task(approve_when_asked())
    record = await cp_container.run_and_wait(TaskRequest(prompt="Open Files and delete old.txt"), timeout=20)
    await helper
    assert record.status == TaskStatus.COMPLETED
    msgs = [e.data for e in cp_container.events.history(record.task_id) if e.event.value == "agent_progress"]
    ask = next(m for m in msgs if (m.get("control") or {}).get("request"))
    assert ask["control"]["request"]["id"] == "r1" and ask["message"].startswith("Waiting for you")
    assert "r1" not in PENDING_REQUESTS  # cleared once answered


async def test_failed_desktop_task_is_not_retried(cp_container):
    FakeControlClient.script = [{"event": "task_failed", "result": "Chrome did not open", "actions": 0, "duration": 2}]
    record = await cp_container.run_and_wait(TaskRequest(prompt="Open Chrome and go to example.com"), timeout=20)
    assert record.status == TaskStatus.FAILED and "Chrome did not open" in record.error
    assert record.agent_results["control"].attempts == 1
    assert record.agent_results["control"].error_type == "control_failed"


async def test_cancelling_the_task_stops_the_desktop_task(cp_container):
    FakeControlClient.script = []
    FakeControlClient.block = True
    record = cp_container.submit(TaskRequest(prompt="Open Chrome and go to example.com"))
    for _ in range(100):
        if ModelState.ACTIVE.value in FakeControlClient.seen_state:
            break
        await asyncio.sleep(0.05)
    cp_container.cancel(record.task_id)
    for _ in range(100):
        if FakeControlClient.stopped:
            break
        await asyncio.sleep(0.05)
    assert FakeControlClient.stopped
    assert cp_container.models.state("control_panel") != ModelState.ACTIVE


async def test_control_panel_refuses_non_ollama_model(cp_container):
    agent = cp_container.agents.get("control_panel")
    cp_container.settings.agents["control_panel"].model = "base"
    try:
        from app.schemas.agent import AgentTask

        with pytest.raises(ControlPanelError, match="needs an Ollama vision model"):
            await agent.run(AgentTask(task_id="t", step_id="s", agent="control_panel", instruction="x", prompt="x"))
    finally:
        cp_container.settings.agents["control_panel"].model = "control_panel"


async def test_screen_view_proxies_the_agents_screen(container, monkeypatch):
    from app.api import routes_control

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"\xff\xd8jpeg", headers={"content-type": "image/jpeg"})

    real = httpx.AsyncClient
    monkeypatch.setattr(routes_control.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    resp = await routes_control.control_panel_screen(container)
    assert resp.body == b"\xff\xd8jpeg" and resp.media_type == "image/jpeg"
    assert resp.headers["cache-control"] == "no-store"
    assert seen == [container.settings.control_panel.url.rstrip("/") + "/api/screen.jpg"]


async def test_screen_view_reports_an_unreachable_service(container, monkeypatch):
    from fastapi import HTTPException

    from app.api import routes_control

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(routes_control.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(HTTPException) as exc:
        await routes_control.control_panel_screen(container)
    assert exc.value.status_code == 503


# ---------------------------------------------------------------- the service is closed after every desktop task
@pytest.mark.parametrize("final", ["task_completed", "task_failed"])
async def test_service_is_stopped_after_each_desktop_task(cp_container, final):
    FakeControlClient.script = [{"event": final, "result": "r", "actions": 0, "duration": 1}]
    await cp_container.run_and_wait(TaskRequest(prompt="Open Calculator and calculate 2 * 3"), timeout=20)
    assert FakeControlClient.service_stops == 1


async def test_service_is_kept_when_stop_after_task_is_off(cp_container):
    cp_container.settings.control_panel.stop_after_task = False
    FakeControlClient.script = [{"event": "task_completed", "result": "6", "actions": 1, "duration": 1}]
    await cp_container.run_and_wait(TaskRequest(prompt="Open Calculator and calculate 2 * 3"), timeout=20)
    assert FakeControlClient.service_stops == 0


async def test_service_waits_for_the_last_overlapping_desktop_task(cp_container):
    ControlPanelAgent._active = 1  # another desktop task is still running in this process
    try:
        FakeControlClient.script = [{"event": "task_completed", "result": "6", "actions": 1, "duration": 1}]
        await cp_container.run_and_wait(TaskRequest(prompt="Open Calculator and calculate 2 * 3"), timeout=20)
        assert FakeControlClient.service_stops == 0
    finally:
        ControlPanelAgent._active = 0


def _listener(cwd, port: int, script: str = "app.py"):
    import subprocess
    import sys

    (cwd / script).write_text(
        "import socket, time\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"s.bind(('127.0.0.1', {port})); s.listen(); print('up', flush=True); time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, script], cwd=cwd, stdout=subprocess.PIPE)
    assert proc.stdout.readline().strip() == b"up"
    return proc


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(tmp_path, project, port: int) -> ControlPanelClient:
    settings = _ollama_settings(tmp_path)
    cfg = settings.control_panel.model_copy(update={"project_dir": str(project), "url": f"http://127.0.0.1:{port}"})
    return ControlPanelClient(cfg, settings)


def _sleeper():
    import subprocess
    import sys

    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


async def test_stop_service_terminates_the_service_this_process_started(tmp_path):
    proc = ControlPanelClient._proc = _sleeper()
    await _client(tmp_path, tmp_path, 8765).stop_service()
    assert proc.poll() is not None and ControlPanelClient._proc is None


async def test_stop_service_stops_a_hand_started_service_but_nothing_else(tmp_path):
    project, other = tmp_path / "computer_ai_agent", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    ControlPanelClient._proc = None

    port = _free_port()
    stranger = _listener(other, port, "server.py")  # some other program holding the port: left alone
    try:
        await _client(tmp_path, project, port).stop_service()
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()

    port = _free_port()
    service = _listener(project, port)  # computer_ai_agent started by hand from its project folder
    try:
        await _client(tmp_path, project, port).stop_service()
        assert service.wait(5) is not None
    finally:
        service.kill()
