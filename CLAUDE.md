# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The repo holds two Python projects that work together. Each has its own venv; run commands from the project folder.

- `multi_ai_agent/`: **Agent Control Room**. A FastAPI multi-agent pipeline with local Hugging Face/Ollama inference
  and a 3D web dashboard. Serves http://127.0.0.1:8000.
- `computer_ai_agent/`: **Computer Agent**. Controls desktop apps (X11) by observing the screen, planning one step,
  acting, then verifying. Serves http://127.0.0.1:8765. It runs standalone, and it also backs the pipeline's
  `control_panel` agent.

All AI inference must stay local. `app/core/local_guard.py` (pipeline) and `models/llm.py` (computer agent) refuse
non-local Ollama URLs and cloud AI keys; don't add cloud providers.

## multi_ai_agent

```bash
cd multi_ai_agent
.venv/bin/python run.py                         # server + dashboard on :8000 (python -m app.main is equivalent)
.venv/bin/python -m pytest -q                   # ~134 tests, fake models, no GPU needed, ~7 s
.venv/bin/python -m pytest -q tests/test_control_panel.py::test_failed_desktop_task_is_not_retried
.venv/bin/ruff check app tests                  # style rules FURB167/RUF012 are pre-existing repo-wide; don't churn them
.venv/bin/python scripts/download_models.py --model <hf_id> --type text|vision|multimodal
```
Configuration lives in `config.yaml`. `.env` and `MAA_*` env vars override it (see `_ENV_OVERRIDES` in
`app/core/config.py`). Models load only from `models/` (`offline_mode: true`); nothing is downloaded at startup.

### Request flow (`app/orchestration/orchestrator.py`)
1. **`manager` agent:** rule-based intent detection, plus optional LLM routing (off by default).
   - Intents map to capabilities through `INTENT_CAPABILITIES`; workers are chosen with `registry.first_for(cap)`.
   - Rule order matters: `CONTROL_RULE` (desktop) runs before `RESEARCH_RULE` (web), which runs before the text
     intents. `classify_intent` is pure and unit-tested.
   - The manager then calls `ModelManager.prepare()` with every chosen agent's `required_models()`.
2. **`planner` agent:** turns the intent into a DAG using `PLAN_TEMPLATES`.
3. **Execution:** steps run concurrently as their dependencies allow. Retries are decided by `error_type`; anything in
   `_NOT_RECOVERABLE` (including `control_failed`) is never retried.
4. **`ResultAggregator`:** a single content step is returned as-is, with no model call; several steps are synthesized
   with the `base` model.

**Agents** are auto-discovered: any `BaseAgent` subclass with a `name` in `app/agents/*.py`. Each agent's model comes
from `agents.<name>.model` in config, and inference goes through `BaseAgent.generate()`, which uses the scheduler.

**Model lifecycle** (`app/models/model_manager.py`):
- Loading is LRU and VRAM-threshold based. GPU free memory is read device-wide (`torch.cuda.mem_get_info`), so
  Ollama's VRAM counts too.
- `prepare(required)` unloads idle, non-required models, choosing the smallest set that makes room; the
  `runtime.model_policy: exclusive` setting unloads them all instead. It then loads the required models, protected
  from evicting each other.
- Models in use (`active > 0`) are never evicted.

**Engines:** transformers (default), llama_cpp (GGUF), vllm, and **ollama** (`app/models/engines/ollama_engine.py`).
- Ollama models are config entries with `engine: ollama` and `ollama_model: <tag>`; they have no `path`. The config
  loader keeps them even though path-less models are normally dropped.
- The engine loads with `keep_alive=-1m` so the ModelManager, not Ollama, decides when the model is unloaded, and
  unloads with `keep_alive=0`.
- Every caller must send the same `num_ctx` (`ollama_num_ctx`), or Ollama silently reloads the model.

**Control Panel agent** (`app/agents/control_panel_agent.py`):
- It holds its model with `models.use()` for the whole desktop task, auto-starts `computer_ai_agent` as a subprocess
  with matching `MODEL_NAME`, `MAX_CONTEXT`, `MODEL_KEEP_ALIVE=-1m` and `MODEL_WARMUP=false`, and drives it over
  `/ws/agent`.
- Progress events are re-emitted as `agent_progress` events carrying a `control` payload.
- Approvals are forwarded through `POST /api/control-panel/requests/{id}`, using the `PENDING_REQUESTS` map.
- `control_panel.stop_after_task` (default true) stops the service after every desktop task, once no other desktop
  task is in flight (`ControlPanelAgent._active`): the subprocess it started, or a hand-started one found on the port
  whose cwd is `project_dir`. The next task autostarts it again. The dashboard calls it **AI DYOT**; the id stays
  `control_panel`.
- Tests swap `client_factory` for a fake service.

**Web Tester agent** (`app/agents/web_tester_agent.py`, cube "Web Tester"): tests websites with the
[agent-browser](https://github.com/vercel-labs/agent-browser) CLI (`npm i -g agent-browser@0.38.2`; drives the system
Chrome via `web_tester.browser_path`).
- Routed by the `web_test` intent (`WEB_TEST_PATTERNS`, checked before desktop and research rules): an imperative
  test/QA/dogfood/smoke-test verb aimed at a URL, host or "the/my website/page/app".
- Flow: open → axe audit (`a11y --json`) → loop of snapshot (`-i -c -u`) → model picks ONE command as JSON →
  run → report (`data/output/web_tests/<task>/report.md` + screenshots).
- Safety: one named session per task with `--allowed-domains <target host>`, closed at the end; the model may only
  use `ALLOWED_COMMANDS` (no eval/cookies/state/close); `open` must stay on the target host.
- The agent adds facts the snapshot lacks: new JS errors (`errors --json`; `--clear` does not clear, so they are
  de-duplicated), same-site links fetched once for 4xx/5xx, and after clicks/presses whether the URL changed plus
  native validation messages (`INVALID_FIELDS_JS`), so blocked submits are not reported as missing validation.
- Tool skill: `harness/web_tester/skills/agent-browser/SKILL.md`. When present, the agent follows it and loads
  `agent-browser skills get core|dogfood` (sections in `SKILL_SECTIONS`) into the system prompt.
- TestingTree export (`app/agents/testing_tree_export.py`, `web_tester.export_to_testing_tree`): the model names
  each flow (`"scenario": {"suite", "name"}`); `FlowRecorder` turns successful commands into TestingTree steps,
  resolving `@eN` refs from the snapshot into Playwright `role=<role>[name="<name>"s]` selectors (id/name attribute
  fallback). Flows are cleaned (trimmed after the last real check, input-only and link-only flows dropped, exact
  repeats removed) and every case ends with a check. Navigation (working links) and Accessibility suites are added.
  They go to project "Web Tester · <host>" via `POST /api/projects/{id}/testcase-definitions` (skipping cases already
  saved), are replayed by TestingTree's runner and the pass/fail lands in report.md. TestingTree being down never
  fails the test run.
- Shares the `control_panel` Ollama model (no double VRAM accounting). Tests: `tests/test_web_tester.py`
  (fake browser + scripted model).

**Job Agent** (`app/agents/job_agent.py`, cube "Job Agent"): job search and applications through Auto-Apply AI
(`~/Desktop/AI/JobAgent/Auto-Apply-AI`, branch `local-hardened`; config `job_agent:`).
- Routed by the `job_search` intent (`JOB_PATTERNS`, checked after the desktop rule and before research, whose
  "find ... jobs" pattern would otherwise catch it).
- Flow: the model extracts query/country (whether to apply is decided by `APPLY_RE`, never the model) → autostart
  `run.sh backend` on :8010 with the same Ollama tag/`num_ctx` and `LOCAL_LLM_KEEP_ALIVE=-1m` → `/api/v1/agent/summary`
  → `/hunt` (job boards + scoring) → `/drafts` → one approval prompt per application → `/send` only on Approve.
- Auth to the backend: `backend/data/agent.key` (mode 600, loopback clients only). Auto-Apply runs in LOCAL_MODE:
  no sign-up/login, one owner account (`services/account_setup.py`) set up from the resume. A resume path in the
  prompt (`RESUME_PATH_RE`, home folder only) or `job_agent.resume_path` is uploaded via `/agent/resume`.
- Queries made only of generic words ("find matching jobs") become None, so the backend searches the inferred
  target roles.
- Approvals: progress events carry `control.request` with an `endpoint` (`/api/job-agent/requests/{id}`,
  `routes_job.py`, `PENDING_APPROVALS` futures). The UI's `controlAsk` posts to `req.endpoint`; `ASKING_AGENTS` in
  `app.js` lists agents whose prompts it renders.
- The dashboard markdown renderer has no tables or blockquotes, so results are a numbered list.
- **Browser applications** (`app/agents/job_browser.py`, `JobBrowser`; `JobAgent.browser_factory`): LinkedIn search
  (`linkedin_search`, logged in by the user in the agent's own Chrome profile) → backend `/jobs/import` (dedupe +
  scoring) → `apply()`: an observe → fill → click loop over `OBSERVE_JS` (fields with labels/options/required/
  invalid, buttons, captcha/confirmation signals). Button meaning comes from regexes (`START_RE`, `NEXT_RE`,
  `SUBMIT_RE`, `AVOID_RE`); the model (`FILL_SYSTEM`, `CHOOSE_SYSTEM`) only proposes field values / one button ref.
  Unknown required answers are asked via `_prompt` (kind `choice`/`input`; `routes_job` passes text through) and
  saved in `answers_file`. Submit always goes through an approval with a screenshot; then backend `/applied`.
  Site accounts: `_auth` (approval, generated password in `accounts_file`, codes asked in chat). Never fills
  `SENSITIVE_RE` fields, never solves CAPTCHAs. Requests are guarded to public https hosts (`browser_allow_hosts`
  for tests). Tests: `tests/test_job_browser.py` runs headless Chrome against `tests/job_sites/*.html` (a LinkedIn
  Easy Apply modal and a careers site with sign-up + verification).
- Tests: `tests/test_job_agent.py` (fake backend). The backend's own tests: `cd ../../JobAgent/Auto-Apply-AI/backend &&
  .venv/bin/python -m pytest -q`.

**Speech agent** (`app/agents/speech_agent.py`, cube "Speech · Hey CC", role `control` so it is never routed to):
the voice front door. Services live in `app/models/speech_model.py` (`SpeechToText`: Whisper turbo via transformers,
ffmpeg decoding, a silence gate (`speech.min_rms`) because Whisper invents "Thank you." on silence; `TextToSpeech`:
Piper, CPU) and are `container.speech` / `container.tts` (`ctx.tts` for agents). Routes: `routes_speech.py`.
- The browser does the listening (`app/ui/voice.js`: AudioWorklet `pcm-worklet.js` → energy VAD → 16 kHz WAV per
  utterance → `/api/speech/listen`). `match_wake` decides; fused forms ("ACC", "A C C") count only at the start.
- Do NOT bias Whisper with a "Hey CC" prompt: tested, it strips the wake word and hallucinates "CC, CC, …" on noise.
- Bluetooth headsets (16 kHz HFP; A2DP has no mic and Linux falls back to a speaker *monitor*, which the UI now
  warns about) make Whisper drop or misspell "CC" ("Hey, ...", "Hey Sissi", "Hey Casey", "ACC"). Spellings are in
  `_NAME`; when a transcript only starts with a greeting, `SpeechToText._wake_score` scores " Hey CC" against the
  first 3 s (said ≈ -0.5, not said ≈ -8.5; `speech.wake_score_threshold` -3.5). The VAD ends speech relative to the
  speaking level (`peak * 0.3`, slow decay) and the pre-speech noise floor (`noise * 1.6`), after
  `speech.end_of_speech_seconds` (4 s: pauses to think stay in one request). `speech.require_wake_word: false`
  (conversation mode, the user's choice) sends every non-filler utterance (`is_request`). The log keeps only non-wake attempts' first three words.
- Replies to voice turns (`chat.voice`) are spoken through `/api/speech/speak` chunk by chunk; the listener is muted
  while speaking. Tests: `tests/test_speech.py` (fake Whisper/Piper).

**AirLLM agent** (`app/agents/airllm_agent.py`, cube "AirLLM · 120B", capability `big_model_generation`): answers
with openai/gpt-oss-120b. `airllm.engine: ollama` (default) uses the ModelManager model `gpt_oss` (Ollama
`gpt-oss:120b`, ~21 GB VRAM + RAM; `model_kwargs.think: low` is passed as Ollama's `think`): measured 11 tokens/s
warm. If that model is not pulled it falls back to the AirLLM service in `~/Desktop/AI/AirLLM` (own venv,
`server.py` on :8770, autostarted; see its README). Routed only when named (`BIG_MODEL_RE`: gpt-oss / airllm / 120b model / "ask the
big model"). The AirLLM path frees the GPU first (`models.prepare([], policy="exclusive")`); it measured ~40 s per
token, so `airllm.max_new_tokens` (AirLLM only) defaults to 96. gpt-oss needed `gpt_oss_airllm.py` (MXFP4 experts expanded per layer on
the GPU with transformers' `convert_moe_packed_tensors`; no Hub kernels) and `fix_split.py` (AirLLM's splitter
left layers 25-27 incomplete). Tests: `tests/test_airllm_agent.py` (fake service).

**Harness** (`app/core/harness.py`): the text, reasoning, validation and aggregator agents overlay a prompt harness
from `harness/<agent>/` (google-research/harness2 layout) on their built-in prompts.
- `skills/<mode>/SKILL.md` replaces the built-in prompt for that mode (`BaseAgent.prompts` holds the built-ins);
  `systemprompt.md`, `guardrails.md` (minus skill trigger lines), `tool_descriptions/tool_guidance.md` and
  `LongTermMEMORY.md` are appended. Agents get their system prompt via `self.system_prompt(mode)`.
- Files are re-read when they change. `harness.enabled` / `harness.root` / `harness.dirs` in config
  (`MAA_HARNESS_ENABLED`, `MAA_HARNESS_DIR`); tests point the root at tmp_path (`tests/fakes.make_settings`).
- `scripts/harness.py init|show|rollout`; task sets for harness2 runs are in `harness/tasks/<agent>.json`.
  Improving a harness happens in `../GoogleHrness` (`python -m backend.agents improve multi_ai_agent/<agent>`).

**UI** (`app/ui/`, static, no build step): `app.js` handles the chat and panels; `scene.js` and `cubes.js` draw the
three.js pipeline view.
- Colours are CSS tokens in `styles.css`: light by default, dark via `prefers-color-scheme` or
  `<html data-theme>`. `scene.js` reads `--scene-fog`, `--scene-edge` and `--scene-floor`.
- `?debug=1` exposes `window.__scene` (call `__scene.setState(agent, 'running')`).
- All agent cubes share one colour (`AGENT_TINT` in `scene.js`); state colours (running/done/failed) go on top.
- UI style the user wants: flat sections (no card panels, no div borders or outlines, no `panel` class),
  ChatGPT/Claude-style chat, and a light/dark toggle.

## computer_ai_agent

```bash
cd computer_ai_agent
.venv/bin/python app.py                                    # web control panel on :8765
.venv/bin/python app.py --command "Open Calculator and calculate 25 * 89"   # one task in the terminal
.venv/bin/python app.py --display :7 --debug --command "..."               # drive a test X display, save step artifacts
.venv/bin/python -m pytest -q                              # simulated desktop + scripted model, ~2 s
Xephyr :7 -screen 1280x800 -ac & DISPLAY=:7 xfwm4 &        # throwaway screen for real-desktop tests
AGENT_TEST_DISPLAY=:7 .venv/bin/python -m pytest -q        # + tests/test_desktop.py against :7
```
- **Venv:** it was created with `--system-site-packages` because AT-SPI (`gi`) comes from the distro. Packages are
  top-level (`agents`, `core`, `models`, ...), not namespaced.
- **Configuration:** `.env` / `config/settings.py`. The default model is Ollama `qwen3.6:27b`.
- **The agent really moves the mouse and types.** Test on a nested Xephyr display, never on the user's session,
  unless asked.
  - Chrome on another display needs its own profile (the plugin does this automatically).
  - GNOME single-instance apps (Terminal, Files, gedit when already running, the VS Code snap) open on the user's
    main screen even when launched for `:7`.

### Architecture
- **`MasterAgent`** (`agents/master_agent.py`):
  - `TaskPlanner` splits the command into per-app subtasks. "Open X" and single-app commands skip the model.
  - `ApplicationManager` finds apps (`.desktop` files plus aliases) and launches or focuses them.
  - `AgentManager` returns a plugin sub-agent (`applications/plugins/<app>/agent.py` + `knowledge.json`), or builds
    an `XyzSubAgent` class dynamically.
  - `TaskManager` runs one task at a time.
  - "Close X" (`TaskPlanner.close_plan`, and model-planned subtasks matching `CLOSE_INSTRUCTION`) never goes to a
    sub-agent: `MasterAgent._close_subtask` sends `_NET_CLOSE_WINDOW` and then checks the app has no windows left.
    Clicking a CSD "Close" button had moved focus to Chrome and been reported as success while the app kept running.
- **`GenericApplicationAgent.run`** (`agents/application_agent.py`) is the core loop:
  - observe → `ActionPlanner` returns one action, plus up to 3 keyboard-only `then` steps;
  - each action goes through `PermissionManager` (SAFE / CONFIRM / BLOCK) → `ActionEngine` → `_settle`, which waits
    only on the target window, with nav and settle timeouts → re-observe → `Verifier.check_action`;
  - recovery happens after failures; the independent `Verifier.check_goal` runs before reporting done.
  - The state machine (`core/state_machine.py`) validates every transition, so new flows may need new edges.
- **Perception** (`vision/ui_detector.py`):
  - AT-SPI first. GTK4 reports window-relative extents only, and `accessibility.py` converts them.
  - OCR (RapidOCR) only when accessibility is thin; then OpenCV box detection. Observations are cached by frame diff.
  - `Observation.summary()` is what the model reads. The focused (or largest) text field is shown verbatim in a block:
    one-line or truncated renderings made the model "repair" correct text.
- **Input:**
  - Keys go through `X11Keyboard`, which sends real XTest events (GTK4 ignores pynput's synthetic events) and remaps
    a spare keycode for characters not on the layout.
  - Long or structured text is pasted via `actions/clipboard.py` (an X11 selection owner) to avoid editor
    auto-closing, and the user's clipboard is restored afterwards.
- **Speed** is dominated by model calls, about 2 s text-only or 3.5 s with a screenshot.
  - Keep the `SYSTEM` prompt in `planner/action_planner.py` constant per app, so Ollama can reuse its prefix cache;
    goal, history and observation go in the user message.
  - Screenshots are only attached on the first step, after a problem, or when there are fewer than 6 elements.
- **Harness** (`planner/harness.py`): `harness/` (harness2 layout) is appended to the action planner's `SYSTEM`,
  never replacing it, so the JSON action protocol stays in code. `skills/<app>/SKILL.md` applies only to that app.
  `HARNESS_DIR` overrides the directory. `scripts/harness_rollout.py` runs a task from `harness/tasks/*.json` on the
  simulated calculator desktop (tests/fakes.py) with the real model, for harness2 (`../GoogleHrness`).
- **Logs:**
  - `logs/tasks/<task_id>.jsonl` holds every decision (raw model output, `model_ms`) and action (`settle_ms`,
    `observe_ms`). Replay it to diagnose slow or wrong runs.
  - `logs/applications/<app>.log` is the per-action log.
  - `data/applications/<app>/` holds the learned, user-editable memory.
