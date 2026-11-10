from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from tests.fakes import FakeFactory, build_container, make_image, make_settings


def _client(settings) -> TestClient:
    factory = FakeFactory()
    app = create_app(settings, container_factory=lambda s: build_container(s, factory=factory))
    return TestClient(app)


@pytest.fixture
def client(tmp_path):
    with _client(make_settings(tmp_path)) as c:
        yield c


def wait_for(client: TestClient, task_id: str, timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/tasks/{task_id}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish")


def png_bytes(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- health / introspection
def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["device"] == "cpu"
    for key in ("agents_active", "tasks_running", "models_loaded", "ram_total"):
        assert key in body


def test_models_and_agents(client):
    models = {m["name"]: m for m in client.get("/api/models").json()}
    assert models["base"]["installed"] is True
    assert models["base"]["state"] == "NOT_LOADED"

    res = client.post("/api/models/fast_text/load")
    assert res.status_code == 200
    assert res.json()["state"] == "LOADED"
    assert client.post("/api/models/fast_text/unload").json()["unloaded"] is True
    assert client.post("/api/models/nope/load").status_code == 404

    agents = {a["name"]: a for a in client.get("/api/agents").json()}
    assert {"manager", "planner", "text", "vision", "multimodal", "reasoning", "validation"} <= set(agents)
    assert "image_analysis" in agents["vision"]["capabilities"]


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/tasks" in schema["paths"]
    assert "/api/image" in schema["paths"]


def test_dashboard_served(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (302, 307) and res.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "Agent Control Room" in page.text
    assert client.get("/ui/vendor/three.module.min.js").status_code == 200
    assert client.get("/api").json()["ui"] == "/ui/"


# ---------------------------------------------------------------- tasks
def test_text_task_lifecycle(client):
    res = client.post("/api/tasks", json={"prompt": "Summarize this document: tests are useful.", "priority": "high"})
    assert res.status_code == 202
    task_id = res.json()["task_id"]
    assert res.json()["status"] == "queued"

    body = wait_for(client, task_id)
    assert body["status"] == "completed", body["error"]
    assert body["answer"]
    assert body["plan"]["intent"] == "summarize"
    assert body["agent_results"]["summary"]["agent"] == "text"
    assert body["prompt"].startswith("Summarize this document")
    assert body["priority"] == "HIGH"


def test_chat_waits_for_answer(client):
    res = client.post("/api/chat", json={"message": "Write a haiku about GPUs"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed"
    assert body["answer"].startswith("Answer:")


def test_unknown_task_404(client):
    assert client.get("/api/tasks/task_missing").status_code == 404


def test_cancel_endpoint(client):
    task_id = client.post("/api/tasks", json={"prompt": "Explain recursion"}).json()["task_id"]
    res = client.delete(f"/api/tasks/{task_id}")
    assert res.status_code == 200
    assert res.json()["status"] in ("cancelled", "completed")


# ---------------------------------------------------------------- image input
def test_image_upload_runs_vision_pipeline(client):
    res = client.post(
        "/api/image",
        files={"file": ("photo.png", png_bytes(), "image/png")},
        data={"prompt": "Analyze this image and explain what is happening."},
    )
    assert res.status_code == 202, res.text
    body = wait_for(client, res.json()["task_id"])
    assert body["status"] == "completed", body["error"]
    assert "vision" in body["result"]["agents_used"]
    assert body["result"]["vision"]["objects"] == ["cat", "sofa", "lamp"]


def test_image_path_task(client, tmp_path):
    make_image(tmp_path / "data" / "input", "scene.png")
    res = client.post(
        "/api/tasks",
        json={"prompt": "What is wrong with this screenshot?", "input_type": "image", "image_path": "./data/input/scene.png"},
    )
    assert res.status_code == 202, res.text
    body = wait_for(client, res.json()["task_id"])
    assert body["status"] == "completed", body["error"]
    assert body["plan"]["intent"] == "screenshot_debug"
    assert body["agent_results"]["inspect"]["agent"] == "multimodal"


def test_invalid_image_rejected(client):
    res = client.post("/api/image", files={"file": ("x.png", b"definitely not an image", "image/png")})
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_image"


def test_path_traversal_rejected(client):
    res = client.post("/api/tasks", json={"prompt": "describe", "image_path": "../../../etc/passwd"})
    assert res.status_code == 400


def test_image_required_for_image_input_type(client):
    res = client.post("/api/tasks", json={"prompt": "describe", "input_type": "image"})
    assert res.status_code == 400


def test_request_size_limit(tmp_path):
    with _client(make_settings(tmp_path, security={"max_request_mb": 0.01})) as c:
        res = c.post("/api/tasks", content=b"x" * 50_000, headers={"content-type": "application/json"})
        assert res.status_code == 413


# ---------------------------------------------------------------- websocket
def test_websocket_streams_agent_events(client):
    task_id = client.post("/api/tasks", json={"prompt": "Summarize this: websockets stream events."}).json()["task_id"]
    events: list[dict] = []
    with client.websocket_connect(f"/ws/tasks/{task_id}") as ws:
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["event"] in ("task_completed", "task_failed", "task_cancelled"):
                break
    names = [e["event"] for e in events]
    assert names[0] == "task_queued"
    assert "agent_started" in names and "agent_completed" in names
    assert names[-1] == "task_completed"
    assert "answer" in events[-1]["result"]
    started_agents = {e["agent"] for e in events if e["event"] == "agent_started"}
    assert {"manager", "planner", "text", "reasoning"} <= started_agents


def test_websocket_unknown_task(client):
    with client.websocket_connect("/ws/tasks/task_missing") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "error"


def test_global_event_stream(client):
    with client.websocket_connect("/ws/events") as ws:
        task_id = client.post("/api/tasks", json={"prompt": "Summarize this: one stream for every task."}).json()["task_id"]
        events: list[dict] = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["event"] in ("task_completed", "task_failed", "task_cancelled"):
                break
    assert {e["task_id"] for e in events} == {task_id}
    names = [e["event"] for e in events]
    assert names[0] == "task_queued" and "plan_created" in names and names[-1] == "task_completed"


def test_routing_describes_intents(client):
    routes = {r["intent"]: r for r in client.get("/api/routing").json()}
    assert "summarize" in routes["summarize"]["phrases"]
    assert routes["summarize"]["agents"][0] == "text"
    assert routes["image_analysis"]["input"] == "image" and "vision" in routes["image_analysis"]["agents"]
    assert routes["generate"]["phrases"] == []


def test_image_upload_can_force_agents(client):
    res = client.post(
        "/api/image",
        files={"file": ("photo.png", png_bytes(), "image/png")},
        data={"prompt": "Describe this image.", "agents": "vision"},
    )
    assert res.status_code == 202, res.text
    body = wait_for(client, res.json()["task_id"])
    assert body["status"] == "completed", body["error"]
    assert {s["agent"] for s in body["plan"]["steps"]} <= {"vision", "validation"}
