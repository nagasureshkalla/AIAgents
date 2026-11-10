"""REST + WebSocket API on the simulated desktop."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.routes import create_app
from tests.fakes import make_runtime


def _client(tmp_path: Path, actions):  # noqa: ANN001, ANN202
    rt, app, _ = make_runtime(tmp_path, actions)
    return TestClient(create_app(rt.settings, runtime=rt)), rt, app


def test_task_api_runs_to_completion(tmp_path: Path):
    client, rt, app = _client(tmp_path, [{"action": "type_text", "text": "6*7\n"}, {"action": "done", "text": "42"}])
    with client:
        r = client.post("/api/task", json={"command": "Open Calculator and calculate 6*7"})
        assert r.status_code == 202
        tid = r.json()["task_id"]
        for _ in range(100):
            body = client.get(f"/api/tasks/{tid}").json()
            if body["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert body["status"] == "completed", body
        assert body["result"] == "42" and app.display == "42"
        assert [a["action"] for a in body["record"]["actions"]] == ["type_text"]
        assert any(e["event"] == "action_completed" for e in body["events"])
        status = client.get("/api/status").json()
        assert status["models"]["ok"] and status["desktop"]["backend"] == "fake"
        assert client.get("/api/history").json()[0]["task_id"] == tid
        assert client.get("/api/screen.jpg").headers["content-type"] == "image/jpeg"
        assert client.get("/").status_code == 200


def test_websocket_streams_events_and_accepts_commands(tmp_path: Path):
    client, rt, app = _client(tmp_path, [{"action": "type_text", "text": "1+1\n"}, {"action": "done", "text": "2"}])
    with client, client.websocket_connect("/ws/agent") as ws:
        assert ws.receive_json()["event"] == "hello"
        ws.send_json({"type": "command", "command": "Open Calculator and calculate 1+1"})
        seen = []
        for _ in range(400):
            e = ws.receive_json()
            seen.append(e["event"])
            if e["event"] in ("task_completed", "task_failed"):
                break
        assert "task_accepted" in seen and "action_started" in seen and seen[-1] == "task_completed"
