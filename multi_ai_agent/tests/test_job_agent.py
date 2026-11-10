"""Job Agent: routing, the Auto-Apply flow with per-application approval, and its safety limits."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agents.job_agent import (
    PENDING_APPROVALS,
    AutoApplyClient,
    JobAgent,
    JobAgentError,
)
from app.agents.manager_agent import ManagerAgent
from app.api.routes_job import Answer, answer_request
from app.core.config import Settings
from app.models.engines import ollama_engine
from app.schemas.common import TaskStatus
from app.schemas.task import TaskRequest
from tests.fakes import CUDA_DEVICE, FakeFactory, build_container, make_settings


@pytest.mark.parametrize("prompt", [
    "Find me Python developer jobs in Germany",
    "find remote React jobs",
    "search for data science internships",
    "Apply to the top 2 backend engineer roles",
    "start my job search",
    "auto-apply to jobs for me",
    "show jobs that match my resume",
])
def test_job_requests_route_to_job_agent(prompt):
    assert ManagerAgent.classify_intent(prompt, False) == "job_search"


@pytest.mark.parametrize(("prompt", "intent"), [
    ("Open Chrome and search for Python jobs", "computer_control"),
    ("Explain what a job scheduler does", "explain"),
    ("search the web for the latest news", "research"),
    ("Write a cover letter template", "generate"),
])
def test_other_requests_do_not(prompt, intent):
    assert ManagerAgent.classify_intent(prompt, False) == intent


def test_backend_must_be_local(tmp_path):
    s = make_settings(tmp_path)
    with pytest.raises(JobAgentError, match="this computer"):
        AutoApplyClient(s.job_agent.model_copy(update={"url": "http://8.8.8.8:8010"}), s)


class FakeAutoApply:
    """Stands in for the Auto-Apply backend's /api/v1/agent API."""

    profile: dict[str, Any] = {}
    sent: list[dict[str, Any]] = []
    calls: list[Any] = []

    def __init__(self, cfg, settings) -> None:
        self.url = cfg.url

    async def ensure_running(self, model_tag: str, num_ctx: int) -> None:
        FakeAutoApply.calls.append(("start", model_tag, num_ctx))

    async def summary(self) -> dict[str, Any]:
        return FakeAutoApply.profile

    async def hunt(self, **params: Any) -> dict[str, Any]:
        FakeAutoApply.calls.append(("hunt", params))
        jobs = [
            {"id": 1, "title": "Python Developer", "company": "Acme", "location": "Berlin", "url": "https://j/1",
             "score": 0.91, "can_email_apply": True, "reasons": ["Strong Python fit"]},
            {"id": 2, "title": "Backend Engineer", "company": "Globex", "location": "Remote", "url": "https://j/2",
             "score": 0.84, "can_email_apply": True, "reasons": []},
            {"id": 3, "title": "Python Dev", "company": "Initech", "location": "Munich", "url": "https://j/3",
             "score": 0.7, "can_email_apply": False, "reasons": []},
        ]
        return {"jobs": jobs, "scored": 3, "new_jobs_found": 2}

    async def drafts(self, job_ids: list[int]) -> list[dict[str, Any]]:
        FakeAutoApply.calls.append(("drafts", job_ids))
        return [{"job_id": i, "ok": True, "title": f"Job {i}", "company": f"Co {i}", "url": f"https://j/{i}",
                 "recipient_email": f"hr{i}@co.io", "sender_email": "me@gmail.com", "subject": "Application",
                 "cover_letter": "Dear team, ...", "has_resume_attachment": True} for i in job_ids]

    async def upload_resume(self, path) -> dict[str, Any]:
        FakeAutoApply.calls.append(("resume", path.name))
        FakeAutoApply.profile["resume_uploaded"] = True
        return {"name": "Ada Lovelace", "email": "ada@example.com", "target_roles": ["Backend Engineer"],
                "experience_level": "Senior (5+ Yrs)", "skills": ["Python"], "links": [], "ats_score": 80}

    async def send(self, job_id: int, cover_letter: str, recipient_email: str) -> dict[str, Any]:
        FakeAutoApply.sent.append({"job_id": job_id, "to": recipient_email})
        return {"success": True, "recipient_email": recipient_email}


def _settings(tmp_path) -> Settings:
    data = make_settings(tmp_path).model_dump()
    data["models"]["control_panel"] = {"engine": "ollama", "ollama_model": "qwen3.6:27b", "type": "multimodal",
                                       "est_size_gb": 18, "model_kwargs": {"num_ctx": 16384}}
    data["agents"]["job_agent"] = {"model": "control_panel"}
    data["job_agent"] = {"max_drafts": 2, "approval_timeout": 5, "browser_apply": False, "linkedin": False}
    return Settings.model_validate(data)


@pytest.fixture
async def container(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama_engine, "ollama_installed", lambda url, tag: True)
    monkeypatch.setattr(JobAgent, "client_factory", FakeAutoApply)
    FakeAutoApply.sent, FakeAutoApply.calls = [], []
    FakeAutoApply.profile = {"resume_uploaded": True, "gmail_connected": True, "daily_limit": 20, "sent_today": 0,
                             "model": "qwen3.6:27b", "target_roles": ["Python Developer"]}
    c = build_container(_settings(tmp_path), factory=FakeFactory(), device=CUDA_DEVICE)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


async def test_lists_ranked_jobs_without_sending(container):
    record = await container.run_and_wait(TaskRequest(prompt="Find me Python developer jobs in Germany"), timeout=20)
    assert record.status == TaskStatus.COMPLETED, record.error
    assert [s.agent for s in record.plan.steps] == ["job_agent"]
    assert record.decision["models"]["required"] == ["control_panel"]
    hunt = next(c for c in FakeAutoApply.calls if c[0] == "hunt")[1]
    assert hunt["country"] == "de" and "python developer" in (hunt["query"] or "").lower()
    assert ("start", "qwen3.6:27b", 16384) in FakeAutoApply.calls  # backend shares the managed model + num_ctx
    answer = record.result["answer"]
    assert "1. **91%** [Python Developer](https://j/1) — Acme · Berlin · email apply" in answer
    assert not FakeAutoApply.sent and not any(c[0] == "drafts" for c in FakeAutoApply.calls)


async def test_sends_only_approved_applications(container):
    async def answer_prompts() -> None:
        answers = iter(["approve", "deny"])
        handled: set[str] = set()
        for _ in range(400):
            for rid in list(PENDING_APPROVALS):
                if rid not in handled:
                    handled.add(rid)
                    await answer_request(rid, Answer(answer=next(answers)))
            if len(handled) == 2:
                return
            await asyncio.sleep(0.02)

    helper = asyncio.create_task(answer_prompts())
    record = await container.run_and_wait(TaskRequest(prompt="Apply to the top 5 Python developer jobs"), timeout=20)
    await helper
    assert record.status == TaskStatus.COMPLETED, record.error
    # max_drafts caps it at 2, and the link-only posting is never drafted
    assert ("drafts", [1, 2]) in FakeAutoApply.calls
    assert FakeAutoApply.sent == [{"job_id": 1, "to": "hr1@co.io"}]
    apps = record.agent_results["jobs"].result["applications"]
    assert [a["status"] for a in apps] == ["sent", "skipped"]
    asks = [e.data["control"]["request"] for e in container.events.history(record.task_id)
            if e.event.value == "agent_progress" and (e.data.get("control") or {}).get("request")]
    assert len(asks) == 2 and asks[0]["endpoint"].startswith("/api/job-agent/requests/")
    assert "To: hr1@co.io" in asks[0]["message"] and "Dear team" in asks[0]["message"]


async def test_no_gmail_means_nothing_is_drafted(container):
    FakeAutoApply.profile["gmail_connected"] = False
    record = await container.run_and_wait(TaskRequest(prompt="apply to python jobs"), timeout=20)
    assert record.status == TaskStatus.COMPLETED
    assert "Connect Gmail" in record.result["answer"]
    assert not FakeAutoApply.sent and not any(c[0] == "drafts" for c in FakeAutoApply.calls)


async def test_without_a_resume_it_asks_where_it_is(container):
    FakeAutoApply.profile["resume_uploaded"] = False
    record = await container.run_and_wait(TaskRequest(prompt="find react jobs"), timeout=20)
    assert record.status == TaskStatus.COMPLETED
    assert "Tell me where your resume is" in record.result["answer"]
    assert not any(c[0] in ("hunt", "resume") for c in FakeAutoApply.calls)


async def test_account_is_set_up_from_a_resume_named_in_chat(container, tmp_path, monkeypatch):
    from app.agents import job_agent

    monkeypatch.setattr(job_agent.Path, "home", lambda: tmp_path)
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "My CV.pdf").write_bytes(b"%PDF-1.4 resume")
    FakeAutoApply.profile["resume_uploaded"] = False
    prompt = "set up my job profile from my resume '~/Docs/My CV.pdf' and find matching jobs"
    assert ManagerAgent.classify_intent(prompt, False) == "job_search"
    record = await container.run_and_wait(TaskRequest(prompt=prompt), timeout=20)
    assert record.status == TaskStatus.COMPLETED, record.error
    assert ("resume", "My CV.pdf") in FakeAutoApply.calls
    answer = record.result["answer"]
    assert "Account set up from your resume" in answer and "Ada Lovelace" in answer and "Backend Engineer" in answer
    hunt = next(c[1] for c in FakeAutoApply.calls if c[0] == "hunt")
    assert hunt["query"] is None  # "matching" is not a role: the backend searches the inferred target roles


def test_resume_outside_home_is_refused(tmp_path, monkeypatch):
    from app.agents import job_agent

    monkeypatch.setattr(job_agent.Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()
    path, problem = JobAgent.resume_file("use /etc/passwd.txt", None)
    assert path is None and "home folder" in problem


async def test_unknown_approval_id_is_rejected():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await answer_request("job-nope", Answer(answer="approve"))
    assert exc.value.status_code == 404


def test_web_app_and_backend_are_served_by_the_control_room(tmp_path, monkeypatch):
    """The Auto-Apply web app runs at /job/ and reaches its backend through /job-api on the Control Room's port."""
    import httpx
    from fastapi.testclient import TestClient

    from app.api import routes_job
    from app.main import create_app

    project = tmp_path / "Auto-Apply-AI"
    (project / "frontend" / "out" / "opportunities").mkdir(parents=True)
    (project / "frontend" / "out" / "opportunities" / "index.html").write_text("<p>jobs</p>")
    data = _settings(tmp_path).model_dump()
    data["job_agent"]["project_dir"] = str(project)
    settings = Settings.model_validate(data)

    seen: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"ok": True})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(routes_job.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(backend), **kw))
    monkeypatch.setattr(AutoApplyClient, "healthy", lambda self: asyncio.sleep(0, True))
    monkeypatch.setattr(ollama_engine, "ollama_installed", lambda url, tag: True)

    app = create_app(settings, container_factory=lambda s: build_container(s, factory=FakeFactory()))
    with TestClient(app) as client:
        assert client.get("/job", follow_redirects=False).headers["location"] == "/job/opportunities/"
        assert client.get("/job/opportunities/").text == "<p>jobs</p>"
        r = client.post("/job-api/api/v1/search/trigger?limit=5", json={"q": 1},
                        headers={"Authorization": "Bearer t", "Origin": "http://evil.example"})
        assert r.status_code == 201 and r.json() == {"ok": True}
    sent = seen[-1]
    assert str(sent.url) == "http://127.0.0.1:8010/api/v1/search/trigger?limit=5"
    assert sent.headers["authorization"] == "Bearer t" and "origin" not in sent.headers
    assert sent.content == b'{"q":1}'


class FakeBrowser:
    """Stands in for JobBrowser: LinkedIn finds two jobs; applying asks for approval like the real one."""

    log: list[Any] = []

    def __init__(self, cfg, settings, *, ask, llm, progress) -> None:
        self.ask = ask

    async def __aenter__(self):
        FakeBrowser.log.append("open")
        return self

    async def __aexit__(self, *exc):
        FakeBrowser.log.append("close")

    async def linkedin_login(self) -> bool:
        return True

    async def linkedin_search(self, query, location, limit):
        FakeBrowser.log.append(("search", query, location))
        return [{"title": "Python Developer", "company": "Hooli", "url": "https://www.linkedin.com/jobs/view/7/",
                 "easy_apply": True, "description": "Python", "source": "LinkedIn", "location": "Remote"},
                {"title": "Python Engineer", "company": "Pied Piper", "url": "https://www.linkedin.com/jobs/view/8/",
                 "easy_apply": False, "description": "Python", "source": "LinkedIn", "location": "Remote"}]

    async def apply(self, job, applicant, letter):
        from app.agents.job_browser import Outcome

        FakeBrowser.log.append(("apply", job["url"], applicant["email"]))
        if job.get("easy_apply") is False:
            await letter()  # the company form asks for a cover letter
        answer = await self.ask(f"Submit this application to {job['company']} — {job['title']}?",
                                choices=["Submit", "Skip this job"], kind="choice")
        return Outcome("submitted", "confirmed") if answer == "Submit" else Outcome("skipped", "not approved")


async def test_linkedin_and_website_applications(container, monkeypatch):
    monkeypatch.setattr(JobAgent, "browser_factory", FakeBrowser)
    FakeBrowser.log = []
    cfg = container.settings.job_agent
    cfg.browser_apply, cfg.linkedin, cfg.max_drafts = True, True, 1
    recorded: list[Any] = []

    async def import_jobs(self, jobs):
        FakeAutoApply.calls.append(("import", [j["url"] for j in jobs]))
        return {"jobs": [{**j, "id": 70 + i, "score": 0.95 - i * 0.1, "can_email_apply": False}
                         for i, j in enumerate(jobs)]}

    async def applicant(self):
        return {"name": "Ada Lovelace", "email": "ada@example.com", "resume_path": "/home/ada/cv.pdf"}

    async def can_apply(self, job_id):
        return {"ok": job_id != 3, "reason": "Already applied to Python Dev at Initech." if job_id == 3 else None}

    async def cover_letter(self, job_id):
        return f"Letter for {job_id}"

    async def applied(self, job_id, via, submitted, notes=None, cover_letter=None):
        recorded.append((job_id, via, submitted, cover_letter))
        return {"ok": True}

    for name, fn in [("import_jobs", import_jobs), ("applicant", applicant), ("can_apply", can_apply),
                     ("cover_letter", cover_letter), ("applied", applied)]:
        monkeypatch.setattr(FakeAutoApply, name, fn, raising=False)

    async def answer_prompts() -> None:
        handled: set[str] = set()
        for _ in range(600):
            for rid in list(PENDING_APPROVALS):
                if rid not in handled:
                    handled.add(rid)
                    events = [e for evs in container.events._history.values() for e in evs]
                    msg = next(e.data["control"]["request"]["message"] for e in events
                               if (e.data.get("control") or {}).get("request", {}).get("id") == rid)
                    # approve the email + Easy Apply, skip the second website application
                    await answer_request(rid, Answer(answer="deny" if "Pied Piper" in msg else
                                                     "approve" if msg.startswith("Send") else "Submit"))
            await asyncio.sleep(0.02)

    helper = asyncio.create_task(answer_prompts())
    record = await container.run_and_wait(TaskRequest(prompt="Apply to the top 3 Python developer jobs in Germany"),
                                          timeout=20)
    helper.cancel()
    assert record.status == TaskStatus.COMPLETED, record.error
    assert ("search", "Python developer", "Germany") in FakeBrowser.log and FakeBrowser.log[-1] == "close"
    apps = record.agent_results["jobs"].result["applications"]
    by_company = {a["company"]: a for a in apps}
    assert FakeAutoApply.sent == [{"job_id": 1, "to": "hr1@co.io"}]          # email application (max_drafts 1)
    assert by_company["Hooli"]["status"] == "applied" and by_company["Hooli"]["via"] == "LinkedIn Easy Apply"
    assert by_company["Pied Piper"]["status"] == "skipped"                   # not approved: nothing submitted
    assert by_company["Initech"]["status"] == "skipped" and "Already applied" in by_company["Initech"]["detail"]
    assert recorded == [(70, "LinkedIn Easy Apply", True, None)]
    answer = record.result["answer"]
    assert "LinkedIn Easy Apply" in answer and "[Python Developer](https://www.linkedin.com/jobs/view/7/)" in answer
