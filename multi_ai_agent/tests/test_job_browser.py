"""Browser applications against local copies of a LinkedIn Easy Apply flow and a company careers site.

Runs the real JobBrowser in headless Chrome. The local model and your dashboard answers are scripted."""

from __future__ import annotations

import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.agents.job_browser import JobBrowser, _norm
from tests.fakes import make_settings

SITES = Path(__file__).parent / "job_sites"
ROUTES = {
    r"^/feed/?$": "search.html",
    r"^/jobs/search/": "search.html",
    r"^/jobs/view/111/?$": "easy_apply.html",
    r"^/jobs/view/222/?$": "external_job.html",
    r"^/ats/job/222$": "ats_job.html",
    r"^/ats/login": "ats_login.html",
    r"^/ats/signup$": "ats_signup.html",
    r"^/ats/verify$": "ats_verify.html",
    r"^/ats/apply/222$": "ats_apply.html",
}

pytestmark = pytest.mark.skipif(
    shutil.which("google-chrome") is None or __import__("importlib").util.find_spec("playwright") is None,
    reason="needs Playwright and Google Chrome",
)


@pytest.fixture(scope="module")
def site():
    records: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            for pattern, name in ROUTES.items():
                if re.search(pattern, self.path):
                    body = (SITES / name).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            records.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"port": server.server_port, "records": records}
    server.shutdown()


class Script:
    """Your dashboard answers and the local model, scripted."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, list[str], str]] = []

    async def ask(self, message: str, *, choices: list[str], kind: str, screenshot: str | None = None) -> str:
        self.asked.append((message, choices, kind))
        if message.startswith("Submit this application"):
            return "Submit"
        if "Create" in message and "account" in message:
            return "Create account"
        if "verification code" in message:
            return "123456"
        if "Expected salary" in message:
            return "120000"
        if "notice period" in message:
            return "2 weeks"
        return "Skip this job"

    async def llm(self, user: str, system: str) -> str:
        if "FIELDS:" not in user:
            return '{"ref": null}'
        out = {}
        for line in user.split("FIELDS:\n", 1)[1].splitlines():
            f = json.loads(line)
            label = f["label"].lower()
            out[f["ref"]] = ("5" if "years" in label else "Yes" if "authorized" in label else
                             "LinkedIn" if "hear about" in label else "United States (+1)" if "country code" in label
                             else None)
        return json.dumps(out)


@pytest.fixture
def browser_cfg(tmp_path, site):
    settings = make_settings(tmp_path)
    cfg = settings.job_agent
    cfg.browser_headless = True
    cfg.browser_allow_hosts = ["127.0.0.1", "localhost"]
    cfg.linkedin_url = f"http://127.0.0.1:{site['port']}"
    cfg.browser_profile_dir = str(tmp_path / "profile")
    cfg.accounts_file = str(tmp_path / "accounts.json")
    cfg.answers_file = str(tmp_path / "answers.json")
    resume = tmp_path / "Ada_Lovelace_Resume.pdf"
    resume.write_bytes(b"%PDF-1.4 resume")
    applicant = {"name": "Ada Lovelace", "email": "ada@example.com", "phone": "5551234567",
                 "skills": ["Python", "FastAPI"], "resume_path": str(resume)}
    return settings, cfg, applicant


async def letter() -> str:
    return "Dear Globex team, I would love to build your Python backend."


async def test_linkedin_search_and_easy_apply(browser_cfg, site):
    settings, cfg, applicant = browser_cfg
    script = Script()
    async with JobBrowser(cfg, settings, ask=script.ask, llm=script.llm, progress=lambda m: None) as b:
        assert await b.linkedin_login()
        jobs = await b.linkedin_search("Python Developer", "United States", 5)
        assert [(j["title"], j["company"], j["easy_apply"]) for j in jobs] == [
            ("Python Developer", "Initech", True), ("Backend Engineer", "Globex", False)]
        assert "FastAPI" in jobs[0]["description"]
        outcome = await b.apply(jobs[0], applicant, letter)

    assert outcome.status == "submitted", outcome
    sent = next(r for r in site["records"] if r["site"] == "linkedin")
    assert sent == {"site": "linkedin", "em": "ada@example.com", "cc": "United States (+1)", "ph": "5551234567",
                    "cv": "Ada_Lovelace_Resume.pdf", "yrs": "5", "auth": "Yes", "sal": "120000"}
    # the unknown salary was asked once, and saved for next time
    assert [m for m, _, _ in script.asked if "salary" in m.lower()]
    saved = json.loads(Path(cfg.answers_file).read_text())
    assert saved[_norm("Expected salary (USD)")] == "120000"
    # the approval prompt listed what was entered before submitting
    approval = next(m for m, _, _ in script.asked if m.startswith("Submit this application"))
    assert "Mobile phone number: 5551234567" in approval and "Resume: Ada_Lovelace_Resume.pdf" in approval


async def test_external_site_signup_verify_and_apply(browser_cfg, site):
    settings, cfg, applicant = browser_cfg
    script = Script()
    job = {"title": "Backend Engineer", "company": "Globex", "url": f"http://127.0.0.1:{site['port']}/jobs/view/222/"}
    async with JobBrowser(cfg, settings, ask=script.ask, llm=script.llm, progress=lambda m: None) as b:
        outcome = await b.apply(job, applicant, letter)
        assert await b.page.evaluate("localStorage.getItem('cookies')") == "rejected"  # optional cookies declined

    assert outcome.status == "submitted", outcome
    signup = next(r for r in site["records"] if r["site"] == "signup")
    assert signup["email"] == "ada@example.com" and signup["password_len"] >= 16 and signup["marketing"] is False
    accounts = json.loads(Path(cfg.accounts_file).read_text())
    assert accounts["localhost"]["email"] == "ada@example.com"
    assert oct(Path(cfg.accounts_file).stat().st_mode & 0o777) == "0o600"
    sent = next(r for r in site["records"] if r["site"] == "ats")
    assert sent["first_name"] == "Ada" and sent["last_name"] == "Lovelace" and sent["email"] == "ada@example.com"
    assert sent["resume"] == "Ada_Lovelace_Resume.pdf" and sent["cover_letter"].startswith("Dear Globex")
    assert sent["source"] == "LinkedIn" and sent["gender"] == "Decline to self-identify"
    assert sent["ssn"] == "" and sent["notice"] == "2 weeks"
    messages = [m for m, _, _ in script.asked]
    assert any("Create" in m and "localhost" in m for m in messages)
    assert not any("Social Security" in m for m in messages)  # optional sensitive field: left empty, not asked
    # the password never reaches the model or the approval text
    assert accounts["localhost"]["password"] not in " ".join(messages)


async def test_nothing_is_submitted_without_approval(browser_cfg, site):
    settings, cfg, applicant = browser_cfg
    script = Script()

    async def deny_submit(message, **kw):
        return "Skip this job" if message.startswith("Submit this application") else await script.ask(message, **kw)

    before = len([r for r in site["records"] if r["site"] == "linkedin"])
    job = {"title": "Python Developer", "company": "Initech", "url": f"http://127.0.0.1:{site['port']}/jobs/view/111/"}
    async with JobBrowser(cfg, settings, ask=deny_submit, llm=script.llm, progress=lambda m: None) as b:
        outcome = await b.apply(job, applicant, letter)
    assert outcome.status == "skipped" and outcome.detail == "not approved"
    assert len([r for r in site["records"] if r["site"] == "linkedin"]) == before


async def test_refuses_private_hosts(tmp_path):
    from app.agents.job_browser import _host_allowed

    assert not _host_allowed("http://127.0.0.1:8000/", [])
    assert not _host_allowed("https://192.168.1.1/admin", [])
    assert not _host_allowed("file:///etc/passwd", [])
    assert _host_allowed("http://127.0.0.1:9/", ["127.0.0.1"])
