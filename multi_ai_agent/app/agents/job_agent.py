"""Job Agent: finds jobs, scores them against your resume and prepares applications through Auto-Apply AI.

UNDERSTAND (local model -> query/country/apply) -> START the Auto-Apply backend -> HUNT (job boards + local-model
scoring) + LINKEDIN (search in a visible Chrome, imported and scored by the backend) -> DRAFT (cover letters)
-> ASK (one approval per application) -> SEND email applications / APPLY IN THE BROWSER (LinkedIn Easy Apply or the
company's own site, see job_browser.py) only for what you approved -> REPORT

The backend lives in ~/Desktop/AI/JobAgent/Auto-Apply-AI (job_agent.project_dir). It is started with the same
Ollama model tag and num_ctx as this agent's model, with keep_alive=-1m, so both share one resident model and the
ModelManager stays in charge of unloading it.
Safety: the backend must be on this computer; it only mails addresses published in the posting, enforces the daily
limit and never applies twice; this agent never sends without an explicit Approve in the dashboard, and the
recipient and letter sent are exactly the ones shown in the approval prompt. Browser applications are submitted
only after you approved the filled form (answers + screenshot), and are recorded with the same duplicate / daily
limit checks.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import ipaddress
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agents.base_agent import BaseAgent, _models_used
from app.agents.job_browser import JobBrowser, Outcome
from app.core.config import JobAgentConfig, Settings
from app.core.exceptions import AgentSystemError
from app.models.engines.ollama_engine import ollama_num_ctx
from app.schemas.agent import AgentTask
from app.utils.text import extract_json, truncate

logger = logging.getLogger(__name__)

# Approval prompts waiting for the dashboard: request id -> future resolved by POST /api/job-agent/requests/{id}.
PENDING_APPROVALS: dict[str, asyncio.Future[Any]] = {}

APPLY_RE = re.compile(r"\b(apply|send (?:my |the )?applications?|submit (?:my |the )?applications?|auto[- ]?apply)\b", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(\d{1,2})\s+(?:\w+\s+){0,3}?(?:jobs?|roles?|positions?|openings?|internships?|applications?)\b", re.IGNORECASE)
COUNTRIES = {
    "united states": "us", "usa": "us", "america": "us", "canada": "ca", "united kingdom": "gb", "uk": "gb",
    "england": "gb", "germany": "de", "france": "fr", "netherlands": "nl", "india": "in", "australia": "au",
    "singapore": "sg", "ireland": "ie", "spain": "es", "italy": "it", "poland": "pl", "brazil": "br",
    "mexico": "mx", "new zealand": "nz", "south africa": "za", "austria": "at", "belgium": "be", "switzerland": "ch",
}

# A resume named in the request: ~/Documents/cv.pdf, /home/me/CV.docx, or quoted when it contains spaces.
RESUME_PATH_RE = re.compile(r"""["']((?:~|/)[^"'\n]+?\.(?:pdf|docx|txt))["']|((?:~|/)\S+?\.(?:pdf|docx|txt))\b""", re.I)
MAX_RESUME_BYTES = 10 * 1024 * 1024
# Words that describe the request, not a role: "find matching jobs" means "use my target roles".
GENERIC_QUERY_WORDS = {
    "matching", "match", "some", "any", "all", "new", "good", "best", "relevant", "suitable", "fitting", "open",
    "available", "my", "me", "the", "for", "top", "latest", "recent", "more", "few", "great", "right", "a", "an",
}

UNDERSTAND = """You turn a job-hunting request into search parameters. Reply with ONE JSON object only:
{"query": "<job title / keywords to search, e.g. 'Python Developer'; null if none given>",
 "country": "<ISO 3166 alpha-2 code in lower case if a country is named, else null>",
 "internship": <true if they want internships>}"""


class JobAgentError(AgentSystemError):
    code = "job_agent_failed"
    error_type = "job_agent_failed"


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    try:
        return all(ipaddress.ip_address(i[4][0].split("%")[0]).is_loopback for i in socket.getaddrinfo(host, None))
    except (socket.gaierror, ValueError):
        return False


class AutoApplyClient:
    """HTTP client for the Auto-Apply backend's /api/v1/agent API, plus starting/stopping that backend."""

    _proc: subprocess.Popen[bytes] | None = None  # the backend this process started (shared by all tasks)

    def __init__(self, cfg: JobAgentConfig, settings: Settings) -> None:
        if not _is_loopback(cfg.url):
            raise JobAgentError(f"job_agent.url must point at this computer, not {cfg.url}")
        self.cfg = cfg
        self.settings = settings
        self.url = cfg.url.rstrip("/")

    @property
    def project_dir(self) -> Path:
        return self.settings.resolve_path(self.cfg.project_dir)

    def _key(self) -> str:
        path = self.project_dir / "backend" / "data" / "agent.key"
        try:
            return path.read_text().strip()
        except OSError as exc:
            raise JobAgentError(f"Cannot read the Auto-Apply agent key at {path}: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"X-Agent-Key": self._key()}
        if self.cfg.user_email:
            headers["X-Agent-User"] = self.cfg.user_email
        return headers

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                return (await c.get(f"{self.url}/api/v1/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_running(self, model_tag: str, num_ctx: int) -> None:
        if await self.healthy():
            return
        if not self.cfg.autostart:
            raise JobAgentError(f"The Auto-Apply backend is not running at {self.url} (job_agent.autostart is off)")
        self._start(model_tag, num_ctx)
        deadline = time.monotonic() + self.cfg.startup_timeout
        while time.monotonic() < deadline:
            if await self.healthy():
                return
            if AutoApplyClient._proc is not None and AutoApplyClient._proc.poll() is not None:
                break
            await asyncio.sleep(0.5)
        raise JobAgentError(f"The Auto-Apply backend did not start; see {self._log_path()}")

    def _log_path(self) -> Path:
        return self.settings.output_dir / "job_agent" / "backend.log"

    def _start(self, model_tag: str, num_ctx: int) -> None:
        script = self.project_dir / "run.sh"
        if not script.exists():
            raise JobAgentError(f"Auto-Apply AI not found: {script} (job_agent.project_dir)")
        port = urlparse(self.url).port or 8010
        env = {
            **os.environ,
            "BACKEND_PORT": str(port),
            "LOCAL_LLM_URL": self.settings.inference.ollama_url,
            "LOCAL_LLM_MODEL": model_tag,
            "LOCAL_LLM_NUM_CTX": str(num_ctx),
            "LOCAL_LLM_KEEP_ALIVE": "-1m",  # the ModelManager unloads the model, not Ollama
            "FRONTEND_BASE_URL": self.cfg.web_url,
        }
        log = self._log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        logger.info("starting Auto-Apply backend on port %s with model %s", port, model_tag)
        with log.open("ab") as fh:
            AutoApplyClient._proc = subprocess.Popen(
                ["bash", str(script), "backend"], cwd=self.project_dir, env=env, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    @staticmethod
    def shutdown() -> None:
        proc = AutoApplyClient._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, 15)
                proc.wait(timeout=10)
        AutoApplyClient._proc = None

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.cfg.request_timeout) as c:
                r = await c.request(method, f"{self.url}/api/v1/agent{path}", headers=self._headers(), **kwargs)
        except httpx.HTTPError as exc:
            raise JobAgentError(f"Auto-Apply backend request failed: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:300]
            raise JobAgentError(str(detail or f"HTTP {r.status_code}"), details={"status": r.status_code})
        return r.json()

    async def summary(self) -> dict[str, Any]:
        return await self._call("GET", "/summary")

    async def hunt(self, **params: Any) -> dict[str, Any]:
        return await self._call("POST", "/hunt", json=params)

    async def drafts(self, job_ids: list[int]) -> list[dict[str, Any]]:
        return (await self._call("POST", "/drafts", json={"job_ids": job_ids}))["drafts"]

    async def upload_resume(self, path: Path) -> dict[str, Any]:
        files = {"file": (path.name, path.read_bytes())}
        return await self._call("POST", "/resume", files=files)

    async def send(self, job_id: int, cover_letter: str, recipient_email: str) -> dict[str, Any]:
        return await self._call("POST", "/send", json={"job_id": job_id, "cover_letter": cover_letter,
                                                       "recipient_email": recipient_email})

    async def import_jobs(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call("POST", "/jobs/import", json={"jobs": jobs})

    async def applicant(self) -> dict[str, Any]:
        return await self._call("GET", "/applicant")

    async def cover_letter(self, job_id: int) -> str:
        return (await self._call("POST", "/cover-letter", json={"job_id": job_id}))["cover_letter"]

    async def can_apply(self, job_id: int) -> dict[str, Any]:
        return await self._call("GET", f"/jobs/{job_id}/can-apply")

    async def applied(self, job_id: int, via: str, submitted: bool, notes: str | None = None,
                      cover_letter: str | None = None) -> dict[str, Any]:
        return await self._call("POST", "/applied", json={"job_id": job_id, "via": via, "submitted": submitted,
                                                          "notes": notes, "cover_letter": cover_letter})


atexit.register(AutoApplyClient.shutdown)


class JobAgent(BaseAgent):
    name = "job_agent"
    description = ("Finds job and internship openings, scores them against your resume with the local model, drafts "
                   "cover letters and sends applications only after you approve each one (Auto-Apply AI).")
    capabilities = ["job_search"]
    default_model = "control_panel"
    client_factory = AutoApplyClient  # tests swap in a fake backend
    browser_factory = JobBrowser      # ... and a fake browser

    def _country(self, prompt: str, guess: Any) -> str | None:
        low = prompt.lower()
        for name, code in COUNTRIES.items():
            if re.search(rf"\b{re.escape(name)}\b", low):
                return code
        if isinstance(guess, str) and re.fullmatch(r"[a-z]{2}", guess.strip().lower()):
            return guess.strip().lower()
        return None

    @staticmethod
    def resume_file(prompt: str, configured: str | None) -> tuple[Path | None, str | None]:
        """(path, problem): a resume named in the request, else job_agent.resume_path. Only files in your home folder."""
        m = RESUME_PATH_RE.search(prompt)
        raw = (m.group(1) or m.group(2)) if m else configured
        if not raw:
            return None, None
        home = Path.home().resolve()
        path = (home / raw[2:] if raw.startswith("~/") else Path(raw)).resolve()
        if not path.is_relative_to(home):
            return None, f"For safety I only read resumes inside your home folder, not {path}."
        if not path.is_file():
            return None, f"I could not find a resume at {path}."
        if path.stat().st_size > MAX_RESUME_BYTES:
            return None, f"{path.name} is larger than 10 MB."
        return path, None

    async def _understand(self, task: AgentTask) -> dict[str, Any]:
        """Search parameters from the request. The model only extracts keywords; 'apply' is decided by rules."""
        prompt = RESUME_PATH_RE.sub(" ", task.prompt)
        data: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            out = await self.generate(task, prompt[:1500], system=self.system_prompt("jobs", UNDERSTAND),
                                      max_tokens=120, temperature=0.0)
            parsed = extract_json(out.text)
            data = parsed if isinstance(parsed, dict) else {}
        query = data.get("query") if isinstance(data.get("query"), str) else None
        if not query:
            m = re.search(r"\b(?:find|search(?:\s+for)?|look\s+for|apply\s+(?:to|for))\s+(?:me\s+)?(?:the\s+)?(?:(?:top|best|first)\s+)?(?:\d+\s+)?"
                          r"(?:some\s+|new\s+)?(.+?)\s+(?:jobs?|roles?|positions?|openings?|internships?)\b", prompt, re.IGNORECASE)
            query = m.group(1) if m else None
        query = re.sub(r"[^\w+#./ -]", " ", query or "").strip()[:120] or None
        if query and all(w.lower() in GENERIC_QUERY_WORDS or w.isdigit() for w in query.split()):
            query = None
        internship = bool(data.get("internship")) or bool(re.search(r"\binternships?\b", prompt, re.IGNORECASE))
        if query and internship and "intern" not in query.lower():
            query += " Internship"
        count = COUNT_RE.search(prompt)
        return {
            "query": query,
            "country": self._country(prompt, data.get("country")),
            "apply": bool(APPLY_RE.search(prompt)),
            "count": int(count.group(1)) if count else None,
        }

    async def _prompt(self, task: AgentTask, message: str, *, kind: str = "confirm", choices: list[str] | None = None,
                      details: dict[str, Any] | None = None, screenshot: str | None = None) -> str:
        """Show a prompt in the dashboard (confirm = Approve/Deny, choice = buttons, input = text) and wait for it."""
        cfg = self.ctx.settings.job_agent
        rid = f"job-{uuid.uuid4().hex[:10]}"
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        PENDING_APPROVALS[rid] = fut
        request = {"id": rid, "kind": kind, "message": message, "endpoint": f"/api/job-agent/requests/{rid}",
                   "details": details or {}, "choices": choices or [],
                   **({"screenshot": screenshot} if screenshot else {})}
        self.progress(task, "Waiting for you", control={"event": "confirmation_required", "request": request})
        try:
            answer = str(await asyncio.wait_for(fut, cfg.approval_timeout))
        except TimeoutError:
            answer = "timeout"
        finally:
            PENDING_APPROVALS.pop(rid, None)
        self.progress(task, "Answered" if answer != "timeout" else "No answer (timed out)",
                      control={"event": "user_response", "request_id": rid})
        return answer

    async def _ask(self, task: AgentTask, message: str, details: dict[str, Any]) -> bool:
        """Approve/Skip prompt in the dashboard."""
        return await self._prompt(task, message, details=details) == "approve"

    def _browser(self, task: AgentTask) -> JobBrowser:
        cfg = self.ctx.settings.job_agent

        async def ask(message: str, *, choices: list[str], kind: str, screenshot: str | None = None) -> str:
            return await self._prompt(task, message, kind=kind, choices=choices, screenshot=screenshot)

        async def llm(user: str, system: str) -> str:
            out = await self.generate(task, user, system=self.system_prompt("forms", system), max_tokens=700,
                                      temperature=0.0)
            return out.text

        return self.browser_factory(cfg, self.ctx.settings, ask=ask, llm=llm, progress=lambda m: self.progress(task, m))

    def _shared_model(self) -> tuple[str, Any]:
        """The Ollama model this agent shares with the Auto-Apply backend: (registry name, spec)."""
        key = self.model_key
        if not key:
            raise JobAgentError("No model configured for the Job Agent (agents.job_agent.model)")
        resolved = self.ctx.models.resolve_required(key)
        spec = self.ctx.models.registry.get(resolved)
        if not spec.is_ollama:
            raise JobAgentError(f"The Job Agent needs an Ollama model (engine: ollama) shared with Auto-Apply AI; "
                                f"'{resolved}' uses {spec.engine}")
        return resolved, spec

    async def ensure_backend(self) -> AutoApplyClient:
        """Start the Auto-Apply backend (if needed) for the web app the Control Room serves at /job/."""
        cfg = self.ctx.settings.job_agent
        if not cfg.enabled:
            raise JobAgentError("The Job Agent is disabled (job_agent.enabled: false)")
        _, spec = self._shared_model()
        client = self.client_factory(cfg, self.ctx.settings)
        await client.ensure_running(spec.ollama_model or "", ollama_num_ctx(spec, self.ctx.settings))
        return client

    async def run(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.ctx.settings.job_agent
        if not cfg.enabled:
            raise JobAgentError("The Job Agent is disabled (job_agent.enabled: false)")
        resolved, spec = self._shared_model()
        models = self.ctx.models

        self.progress(task, "Reading your request")
        params = await self._understand(task)
        client = self.client_factory(cfg, self.ctx.settings)
        async with models.use(resolved):  # keep the shared model resident while the backend uses it
            used = _models_used.get()
            if used is not None and resolved not in used:
                used.append(resolved)
            self.progress(task, "Connecting to Auto-Apply AI", app_url=cfg.web_url)
            await client.ensure_running(spec.ollama_model or "", ollama_num_ctx(spec, self.ctx.settings))
            profile = await client.summary()
            setup: dict[str, Any] | None = None
            named, problem = self.resume_file(task.prompt, None)
            path = named or (None if profile.get("resume_uploaded") else self.resume_file("", cfg.resume_path)[0])
            if problem:
                return self._result(cfg, params, profile, [], [], notes=[problem])
            if path is not None:
                self.progress(task, f"Setting up your account from {path.name} with the local model")
                setup = await client.upload_resume(path)
                profile = await client.summary()
            if not profile.get("resume_uploaded"):
                return self._result(cfg, params, profile, [], [], notes=[(
                    'Tell me where your resume is, e.g. "find jobs with my resume ~/Documents/resume.pdf" '
                    f"(or set job_agent.resume_path, or drop it at {cfg.web_url}). Your account is set up from it.")])

            what = params["query"] or ", ".join(profile.get("target_roles") or []) or "your target roles"
            self.progress(task, f"Searching job boards for {what} and scoring them against your resume")
            hunt = await client.hunt(query=params["query"], country=params["country"], search=True,
                                     limit=cfg.max_results, min_score=cfg.min_score)
            jobs = hunt.get("jobs", [])
            self.progress(task, f"Scored {hunt.get('scored', 0)} jobs ({hunt.get('new_jobs_found', 0)} new)")

            notes: list[str] = []
            applications: list[dict[str, Any]] = []
            browser_on = cfg.browser_apply and (cfg.linkedin or params["apply"])
            if not browser_on:
                if params["apply"]:
                    applications, notes = await self._apply(task, client, cfg, profile, jobs, params["count"])
                elif any(j.get("can_email_apply") for j in jobs):
                    notes.append('Say "apply to the top N" to draft cover letters and approve each application here.')
            else:
                self.progress(task, "Opening Chrome for LinkedIn and website applications")
                try:
                    async with self._browser(task) as browser:
                        linkedin: list[dict[str, Any]] = []
                        if cfg.linkedin:
                            linkedin, li_notes = await self._linkedin(task, browser, client, params, profile)
                            notes += li_notes
                        jobs = sorted(jobs + linkedin, key=lambda j: j.get("score") or 0, reverse=True)
                        if params["apply"]:
                            applications, more = await self._apply(task, client, cfg, profile, jobs, params["count"])
                            notes += more
                            web, more = await self._apply_in_browser(task, browser, client, cfg, profile, jobs,
                                                                     params["count"], len(applications))
                            applications += web
                            notes += more
                        elif jobs:
                            notes.append('Say "apply to the top N" to apply: email, LinkedIn Easy Apply or the company '
                                         "website. You approve every application before it is submitted.")
                except ImportError:
                    notes.append("Browser applications need Playwright: pip install playwright (in the Control Room venv).")
                    if params["apply"]:
                        applications, more = await self._apply(task, client, cfg, profile, jobs, params["count"])
                        notes += more
        return self._result(cfg, params, profile, jobs, applications, notes=notes, new=hunt.get("new_jobs_found", 0),
                            setup=setup)

    async def _linkedin(self, task: AgentTask, browser: JobBrowser, client: AutoApplyClient, params: dict[str, Any],
                        profile: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        """Search LinkedIn (logged in by you, in the agent's window), then let the backend dedupe and score the jobs."""
        cfg = self.ctx.settings.job_agent
        query = params["query"] or next(iter(profile.get("target_roles") or []), None)
        if not query:
            return [], ["LinkedIn: tell me a job title to search for."]
        self.progress(task, "Checking that LinkedIn is logged in")
        if not await browser.linkedin_login():
            return [], ["LinkedIn was skipped (not logged in)."]
        country = params["country"]
        location = next((name.title() for name, code in COUNTRIES.items() if code == country), None) if country else None
        self.progress(task, f"Searching LinkedIn for {query}" + (f" in {location}" if location else ""))
        found = await browser.linkedin_search(query, location, cfg.max_results)
        if not found:
            return [], ["LinkedIn returned no jobs for that search."]
        self.progress(task, f"Scoring {len(found)} LinkedIn jobs against your resume")
        imported = await client.import_jobs(found)
        return imported.get("jobs", []), []

    def _resume_copy(self, path: Path, name: str | None) -> Path:
        """Recruiters see the uploaded file name: "Ada_Lovelace_Resume.pdf", not the backend's storage id."""
        stem = re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_")
        if not stem or not path.is_file():
            return path
        target = self.ctx.settings.resolve_path(self.ctx.settings.job_agent.browser_profile_dir).parent / \
            f"{stem}_Resume{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
            target.write_bytes(path.read_bytes())
        return target

    async def _apply_in_browser(self, task: AgentTask, browser: JobBrowser, client: AutoApplyClient,
                                cfg: JobAgentConfig, profile: dict[str, Any], jobs: list[dict[str, Any]],
                                count: int | None, done: int) -> tuple[list[dict[str, Any]], list[str]]:
        """LinkedIn Easy Apply first, then company websites (LinkedIn's external Apply, job board links)."""
        left = max(0, int(profile.get("daily_limit") or 0) - int(profile.get("sent_today") or 0) - done)
        asked = (count - done) if count else cfg.max_browser_applications  # "apply to the top 3" counts email ones too
        want = min(asked, cfg.max_browser_applications, left)
        candidates = [j for j in jobs if not j.get("can_email_apply")]
        candidates.sort(key=lambda j: (not j.get("easy_apply"), -(j.get("score") or 0)))
        if left == 0:
            return [], ["Daily application limit reached."]
        if want <= 0 or not candidates:
            return [], []
        applicant = await client.applicant()
        if not applicant.get("resume_path"):
            return [], ["Upload your resume first: the website forms need the file."]
        applicant["resume_path"] = str(self._resume_copy(Path(applicant["resume_path"]), applicant.get("name")))
        out: list[dict[str, Any]] = []
        for job in candidates:
            if len([a for a in out if a["status"] == "applied"]) >= want:
                break
            via = "LinkedIn Easy Apply" if job.get("easy_apply") else "Company website"
            entry = {"job_id": job["id"], "title": job.get("title"), "company": job.get("company"), "url": job.get("url"),
                     "via": via}
            check = await client.can_apply(job["id"])
            if not check.get("ok"):
                out.append({**entry, "status": "skipped", "detail": check.get("reason")})
                continue
            self.progress(task, f"Applying to {job.get('title')} at {job.get('company')} ({via})")
            letters: list[str] = []

            async def letter(job_id: int = job["id"], letters: list[str] = letters) -> str:
                letters.append(await client.cover_letter(job_id))
                return letters[-1]

            try:
                outcome = await browser.apply(job, applicant, letter)
            except Exception as exc:  # noqa: BLE001 - a broken site must not stop the other applications
                logger.warning("browser application to %s failed: %s", job.get("url"), exc)
                outcome = Outcome("failed", str(exc)[:200])
            if outcome.status == "submitted":
                try:
                    await client.applied(job["id"], via, True, outcome.detail, letters[-1] if letters else None)
                except JobAgentError as exc:
                    outcome = Outcome("submitted", f"{outcome.detail}; not recorded: {exc.message}")
                out.append({**entry, "status": "applied", "detail": f"{via}: {outcome.detail}"})
            else:
                out.append({**entry, "status": outcome.status.replace("_", " "), "detail": outcome.detail})
        return out, []

    async def _apply(self, task: AgentTask, client: AutoApplyClient, cfg: JobAgentConfig, profile: dict[str, Any],
                     jobs: list[dict[str, Any]], count: int | None) -> tuple[list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        if not profile.get("gmail_connected"):
            if not any(j.get("can_email_apply") for j in jobs):
                return [], []
            return [], [f"Connect Gmail at {cfg.web_url}/settings to send email applications (none were sent)."]
        left = max(0, int(profile.get("daily_limit") or 0) - int(profile.get("sent_today") or 0))
        want = min(count or cfg.max_drafts, cfg.max_drafts, left)
        emailable = [j for j in jobs if j.get("can_email_apply")][:want]
        if left == 0:
            return [], ["Daily application limit reached; nothing was sent."]
        if not emailable:
            return [], [] if cfg.browser_apply else ["None of these postings publish a contact email; apply through their links."]
        self.progress(task, f"Writing {len(emailable)} cover letter(s) with the local model")
        out: list[dict[str, Any]] = []
        for draft in await client.drafts([j["id"] for j in emailable]):
            entry = {"job_id": draft["job_id"], "title": draft.get("title"), "company": draft.get("company"),
                     "url": draft.get("url")}
            if not draft.get("ok"):
                out.append({**entry, "status": "not drafted", "detail": draft.get("error")})
                continue
            message = (f"Send this application to {draft['company']} ({draft['title']})?\n"
                       f"To: {draft['recipient_email']}  ·  From: {draft['sender_email']}  ·  "
                       f"Resume attached: {'yes' if draft.get('has_resume_attachment') else 'no'}\n"
                       f"Subject: {draft['subject']}\n\n{draft['cover_letter']}")
            if not await self._ask(task, message, {"job_id": draft["job_id"], "recipient": draft["recipient_email"]}):
                out.append({**entry, "status": "skipped", "detail": "not approved"})
                continue
            self.progress(task, f"Sending to {draft['company']}")
            try:
                sent = await client.send(draft["job_id"], draft["cover_letter"], draft["recipient_email"])
            except JobAgentError as exc:
                out.append({**entry, "status": "failed", "detail": exc.message})
                continue
            out.append({**entry, "status": "sent" if sent.get("success") else "failed",
                        "detail": sent.get("recipient_email") if sent.get("success") else sent.get("error")})
        if want < (count or 0):
            notes.append(f"Limited to {want} application(s) (job_agent.max_drafts / daily limit).")
        return out, notes

    @staticmethod
    def _result(cfg: JobAgentConfig, params: dict[str, Any], profile: dict[str, Any], jobs: list[dict[str, Any]],
                applications: list[dict[str, Any]], *, notes: list[str], new: int = 0,
                setup: dict[str, Any] | None = None) -> dict[str, Any]:
        lines: list[str] = []
        if setup:
            lines += ["## Account set up from your resume", "",
                      f"- **Name:** {setup.get('name') or '—'} · **Email:** {setup.get('email') or '—'}"
                      + (f" · **Phone:** {setup['phone']}" if setup.get("phone") else ""),
                      f"- **Target roles:** {', '.join(setup.get('target_roles') or []) or '—'}"
                      + (f" · **Level:** {setup['experience_level']}" if setup.get("experience_level") else ""),
                      f"- **Skills:** {', '.join((setup.get('skills') or [])[:12]) or '—'}",
                      *([f"- **Links:** {' · '.join(setup['links'])}"] if setup.get("links") else []),
                      f"- **ATS score:** {setup.get('ats_score') if setup.get('ats_score') is not None else '—'}", ""]
        lines += [f"## Jobs for {params['query'] or 'your target roles'}"
                 + (f" ({params['country'].upper()})" if params["country"] else "")]
        if jobs:
            # A list, not a table: the dashboard's markdown renderer has no table support.
            lines += ["", f"Scored against your resume with **{profile.get('model')}** ({new} new postings found).", ""]
            for i, j in enumerate(jobs, 1):
                where = " · ".join(str(v) for v in (j.get("company"), j.get("location")) if v)
                how = ("email apply" if j.get("can_email_apply") else "LinkedIn Easy Apply" if j.get("easy_apply")
                       else "website apply")
                lines.append(f"{i}. **{round((j.get('score') or 0) * 100)}%** [{truncate(j['title'], 80)}]({j['url']}) — "
                             f"{truncate(where, 80)} · {how}")
            best = jobs[0]
            if best.get("reasons"):
                lines += ["", f"Top match — {best['title']} at {best['company']}: {best['reasons'][0]}"]
        elif profile.get("resume_uploaded"):
            lines += ["", "No matching jobs found. Try another title or country."]
        if applications:
            lines += ["", "## Applications", ""]
            lines += [f"- **{a['status']}** — {a['title']} at {a['company']}" + (f": {a['detail']}" if a.get("detail") else "")
                      for a in applications]
        if notes:
            lines += ["", *[f"*{n}*" for n in notes]]
        lines += ["", f"Manage your profile, history and Gmail in Auto-Apply AI: {cfg.web_url}"]
        return {"text": "\n".join(lines), "jobs": jobs, "applications": applications, "params": params, "setup": setup,
                "app_url": cfg.web_url, "model": profile.get("model")}
