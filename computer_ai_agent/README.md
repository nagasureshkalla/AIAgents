# Computer Master Agent

A local AI computer-use multi-agent system for Linux (Ubuntu 24.04, X11). Give it a natural-language command
and it controls real applications on your desktop. A **Master Agent** works out which app is needed, opens it, and
hands the work to an **application sub-agent**. The sub-agent then runs an **observe → plan → act → observe →
verify** loop until the goal is visibly done.

All inference is local. The default model server is Ollama on your NVIDIA GPU, or you can use Hugging Face
`transformers` in-process. No cloud AI API is used anywhere.

```
                 USER (web panel / CLI)
                          │
                   MASTER AGENT ── TaskPlanner (command → subtasks per app)
                          │
        ┌─────────────────┼──────────────────┐
 ApplicationManager   AgentManager        Safety (permission manager + confirmations)
 (discover/launch/    (plugin sub-agents or
  detect/switch)       dynamic XyzSubAgent)
                          │
              Application Sub-Agent (GenericApplicationAgent)
                          │
        ┌─────────────────┴──────────────────┐
   Perception (UIDetector)              ActionEngine
   AT-SPI → OCR → CV boxes → VLM        mouse / XTest keyboard / windows
        └─────────────────┬──────────────────┘
                    Verifier (screen diff, title/focus change, goal check)
                          │
                 success / recover / ask the user
```

## Verified results on this machine

These runs used an RTX 3090 with `qwen3.6:27b` through Ollama, on a nested test X display:

| Command | Result |
|---|---|
| *Open Calculator and calculate 137 * 46* | **6302**: 1 action, goal check passed, 21 s |
| *Open Chrome, go to Google, search for NVIDIA RTX 3090, and open the first result.* | NVIDIA's *GeForce RTX 3090 Family* page opened (first organic result, ads skipped, cookie banner rejected): 7 actions, 72 s |
| *Open Chrome and search Google for local AI agents on Linux* (through the web panel) | results page shown: 3 actions, 45 s |

Most of the time goes into model calls, at about 5–8 s per planning step for a 27B model. The GUI side is fast:
- a screenshot takes about 20 ms;
- an accessibility snapshot takes about 150–200 ms;
- clicks, keys and typing take 40–120 ms.

---

## Install (Ubuntu 24.04)

```bash
cd computer_ai_agent
python3.12 -m venv --system-site-packages .venv      # system site-packages = distro python3-gi for AT-SPI
.venv/bin/pip install -r requirements.txt
cp .env.example .env                                  # then edit if needed

# local model server (once):
ollama pull qwen3.6:27b        # or any vision-capable model; set MODEL_NAME in .env
```

No `sudo` is needed. Screenshots, input and window control use pure-Python X11 (python-xlib, XTest, mss), and
OCR uses RapidOCR via pip. Optional extras are in `requirements-gpu.txt`: GPU OCR, and the `transformers` backend.

**Session type:** log in with **"Ubuntu on Xorg"** for full support. The Wayland backend is best-effort; see
Limitations below.

## Run

```bash
.venv/bin/python app.py                                    # web control panel → http://127.0.0.1:8765
.venv/bin/python app.py --command "Open Chrome and search for Python"   # one task in the terminal, no web UI
.venv/bin/python app.py --debug ...                        # also save screenshots/OCR/decisions per step
.venv/bin/python app.py --display :7 ...                   # drive another X display (see "Safe testing")
```

In CLI mode, approvals and questions are asked in the terminal. When stdin is not a terminal, the safe default is
chosen automatically: deny, or stop.

### Safe testing on a throwaway screen

The agent really moves the mouse and types. To try it without touching your own desktop, use a nested X server:

```bash
Xephyr :7 -screen 1280x800 -ac & sleep 1; DISPLAY=:7 xfwm4 &
.venv/bin/python app.py --display :7 --command "Open Calculator and calculate 25 * 89"
.venv/bin/python app.py --display :7                       # web panel showing display :7
```

Chrome is single-instance per profile. When the target display differs from your session's display, the Chrome
plugin automatically uses a separate profile in `data/chrome-profiles/`, so the window opens on the test screen.
GNOME's own single-instance apps open on your main desktop even when started from `:7`, so avoid them on the test
screen. That includes Terminal, Files and the Text Editor.

## Test

```bash
.venv/bin/python -m pytest -q                              # 18 tests: simulated desktop + scripted model, ~2 s
Xephyr :7 -screen 1280x800 -ac & DISPLAY=:7 xfwm4 &
AGENT_TEST_DISPLAY=:7 .venv/bin/python -m pytest -q        # + 3 real-desktop tests (capture, a11y, XTest, click)
```

---

## Phases

Each phase lives in its own modules. "Test" is the quickest command that exercises that phase.

### Phase 1: Basic computer control
`actions/mouse.py`, `actions/keyboard.py`, `actions/window_manager.py`, `actions/platform.py`, `vision/screen_capture.py`
- **Mouse:** pynput, with gliding moves so apps get hover events; click, double/right click, drag, scroll.
- **Keyboard:** `X11Keyboard` sends **real XTest key events**. pynput's synthetic events are silently ignored by
  GTK4 apps such as GNOME Calculator 46. Characters missing from the keyboard layout (é, →, ✓…) are typed by
  temporarily remapping a spare keycode. Key names from the model (CTRL, Enter, PageDown, F5, super…) are normalized.
- **Windows:** EWMH lists, activates (switching virtual desktop if needed) and closes windows, and reads PID,
  WM_CLASS and geometry. When no window manager is running, it falls back to the raw window tree.
- **Capture:** `mss`, about 20 ms per frame.
- Test: `AGENT_TEST_DISPLAY=:7 pytest -q tests/test_desktop.py::test_capture_and_windows`

### Phase 2: Master Agent
`agents/master_agent.py`, `planner/task_planner.py`, `applications/*`
- The **TaskPlanner** turns the command into ordered subtasks, one per app, each with checkpoints. It uses the LLM,
  with a no-model fast path for "Open X" and a rule-based fallback.
- The **ApplicationManager**:
  - discovers installed apps from `.desktop` entries (user, system, snap, flatpak) plus `$PATH`;
  - understands aliases ("chrome", "vs code", "files", "browser" → your default browser);
  - launches apps detached and waits for their window;
  - tracks PIDs and window IDs; switches between and closes apps.
- **Recovery:**
  - If the app isn't found, the Master asks you which installed app to use.
  - If a launch fails, it retries once, then asks you to open the app yourself.
- The **TaskManager** runs one desktop task at a time, since there is one mouse and one keyboard.
- Test: `pytest -q tests/test_agents.py::test_calculator_task_completes_with_verification`

### Phase 3: Generic Application Agent
`agents/application_agent.py`, `planner/action_planner.py`, `planner/verifier.py`, `actions/action_engine.py`
- **One action at a time.** The model gets the goal, checkpoints, app knowledge, the last 10 actions with their
  verified outcomes, and the current element list. It returns exactly one JSON action. A list is cut to its first
  item, so blind sequences never run.
- **Targets:** element id (accessibility → OCR → CV) first, raw coordinates only as a last resort. Element ids are
  validated against the current observation.
- **Waiting:** after each action, the agent waits until the screen stops changing. After likely navigations (link
  click, Enter, typed line ending in ⏎) it first waits for the title change or repaint. This fixed a real
  failure, where Chrome still showed the old page while the new one loaded.
- **Verification:**
  - Per action: pixel change, window title change, focus change, and typed text appearing in the focused field.
  - When the agent says "done": an independent, strict goal check that also sees which actions were taken. It
    rejected landing on a site's cookie-policy page as "first result opened".
- **Self-correction:**
  - "No visible change" or an engine error puts the agent in RECOVERING. It forces OCR and sends the screenshot,
    and the model is told to try a different approach.
  - After `MAX_ACTION_RETRIES` (3) failures of the same action, it asks you: try another approach, or stop.
  - Invalid model replies are re-asked, and a step limit leads to a question instead of a silent failure.
- Test: `pytest -q tests/test_agents.py` (recovery, premature-done rejection, stop/pause)

### Phase 4: Vision and OCR
`vision/accessibility.py`, `vision/ocr.py`, `vision/ui_detector.py`, `models/vision.py`
- **AT-SPI accessibility first:** real roles, names, states, text and exact boxes. GTK4 can't report screen
  coordinates, so window-relative extents are converted. Password fields are never read.
- **OCR (RapidOCR)** is added only when the accessibility tree is thin, for example web pages without
  accessibility, Electron, Blender or games. It processes only the active window region.
- **CV:** OpenCV rectangle detection finds likely fields and buttons when there is no accessibility at all.
- **Vision-language model:** gets a *set-of-marks* screenshot, with numbered boxes matching the element ids,
  downscaled to `MODEL_IMAGE_WIDTH`. It is used only when useful (`VISION_MODE=auto`): no accessibility, after a
  failure, few elements, or a first visit. `LocalVisionModel.locate()` and `.describe()` are available for plugins.
- **Change detection:** perceptual thumbnail diffs. Unchanged screens hit an observation cache.
- Test: `AGENT_TEST_DISPLAY=:7 pytest -q tests/test_desktop.py::test_launch_type_and_read_back_through_accessibility`

### Phase 5: Local LLM
`models/llm.py`, `models/model_manager.py`
- `LocalLLMProvider` sits over two backends:
  - **Ollama** (default): GPU, `keep_alive`, `think: false` for speed, JSON mode. It refuses any non-local URL.
  - **TransformersBackend:** loads `MODEL_PATH` on `cuda:GPU_DEVICE` with `GPU_MEMORY_LIMIT`, for text or
    image-text-to-text models.
- Configure with `MODEL_PROVIDER`, `MODEL_NAME`, `MODEL_PATH`, `GPU_DEVICE`, `GPU_MEMORY_LIMIT`, `MAX_CONTEXT`
  and `TEMPERATURE`. Optional extras: `FAST_MODEL_NAME` and a separate `VISION_MODEL_NAME`.
- GPU generations are serialized, and timings are reported in `/api/status`.
- Model choice: on an RTX 3090, `qwen3.6:27b` produced valid, well-grounded actions. `gemma4:e2b` rambled and
  produced truncated JSON.
- Test: `curl -s localhost:8765/api/status | jq .models`

### Phase 6: Application-specific sub-agents
`applications/plugins/<app>/agent.py` + `knowledge.json`, `agents/agent_manager.py`

Built-in plugins: **chrome, firefox, vscode, terminal, files, calculator, blender, libreoffice**.
- Plugins add launch flags (for example `--force-renderer-accessibility` for Chrome and VS Code), app-state
  readers (browser URL, terminal output tail, calculator display) and seed knowledge (shortcuts, tips).
- Any other app gets a `XyzSubAgent` class created at runtime from `GenericApplicationAgent`. It inspects the UI
  first and builds the spec §6 JSON UI model.

To add a plugin, create `applications/plugins/myapp/agent.py`:
```python
from agents.application_agent import GenericApplicationAgent
APP_KEYS = ["myapp", "org.example.MyApp"]            # desktop ids / names / WM_CLASS it serves
class MyAppSubAgent(GenericApplicationAgent):
    def launch_args(self): return ["--some-flag"]
    def describe_state(self, obs): return "extra facts for the planner"
AGENT = MyAppSubAgent
```
…plus an optional `knowledge.json` (`{"tips": [...], "shortcuts": {...}}`).
- Test: `pytest -q tests/test_agents.py::test_dynamic_subagent_for_unknown_app`

### Phase 7: Persistent memory
`memory/application_memory.py`, `memory/task_memory.py`

`data/applications/<app>/` holds:
- `profile.json`
- `knowledge.json`: seeded from the plugin, then **yours to edit**; learned shortcuts are added to it.
- `ui_memory.json`: window-relative positions of controls seen before.
- `successful_actions.json`
- `failed_patterns.json`

`data/tasks/<task_id>.json` holds full task records: command, subtasks, every action with verification, messages,
result and duration. Similar earlier successes are fed back as hints to both planners.
- Test: `pytest -q tests/test_units.py -k memory`

### Phase 8: Multi-agent orchestration
- **Several live sub-agents:** one per app, kept and reused. The Master switches between them per subtask and
  passes each result to later subtasks (`uses_previous`).
- **Message protocol:** every message uses the standard `AgentMessage` envelope
  (`{"from","to","task","context","priority"}`, and `{"status","result"}` for replies). Messages are logged to the
  task journal and streamed to the UI, which has a tab per agent.
- Example: *"Open Chrome and download X, then open VS Code and create a project using that file"* → two subtasks,
  Chrome then VS Code.

### Phase 9: Safety and human approval
`safety/permission_manager.py`, `safety/confirmation.py`

Every action is classified before it runs:
- **BLOCK:** typing into password or secret fields. The agent must ask you to do it yourself, and it never tries
  to get around logins or CAPTCHAs.
- **CONFIRM:**
  - clicking controls labelled delete/remove/send/post/publish/buy/pay/checkout/transfer/uninstall/format/…;
  - dangerous shell commands (`rm`, `sudo`, `mkfs`, `dd`, `shutdown`, `git push --force`, `curl | sh`…);
  - deleting files in a file manager;
  - closing or quitting apps you didn't ask to close.
- **Approval prompt:** shows the message plus a screenshot in the UI or CLI. A deny is fed back to the model,
  which won't repeat that action.
- **Controls:** Pause, Resume and Stop at any time. Multiple-choice questions come up when the agent is unsure
  (`ask_user`).
- Test: `pytest -q tests/test_units.py::test_safety_rules tests/test_agents.py -k destructive`

### Phase 10: Performance
- Accessibility is preferred over pixels: about 150–200 ms for a full window snapshot, with no OCR at all for
  GTK/Qt apps.
- Observations are cached by thumbnail diff, and OCR runs only on the active window and only when needed.
- Screenshots are downscaled before the model sees them, and images are sent only when useful (`VISION_MODE=auto`).
- Keyboard-first planning hints: one `type_text "25*89⏎"` instead of seven clicks.
- Screen settling is adaptive: static UIs return after about 250 ms, and page loads are waited for (max `SETTLE_TIMEOUT`).
- Async throughout; blocking X11/AT-SPI calls run in threads; the model stays in VRAM (`MODEL_KEEP_ALIVE`) and is
  warmed up when the server starts.
- Optional GPU OCR: `pip uninstall onnxruntime && pip install onnxruntime-gpu`.

---

## Web panel

The panel follows your OS light/dark setting, with a toggle in the top bar:
- the **live screen**, with an action overlay: click point, target box, action label, detected elements, and
  optionally OCR text;
- the **status:** application, agent, state machine state, current action, and expected result;
- **agent tabs:** Master plus one per sub-agent;
- the **conversation:** plan, launches, every action with its reason, ✓/✗ and timing, the goal check, the result,
  and a collapsible action history;
- **inline Approve/Deny prompts and choices**, and Pause/Resume/Stop.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/task` | `{"command": "..."}` → 202 + task id |
| GET | `/api/tasks`, `/api/tasks/{id}` | status, full record, events |
| POST | `/api/tasks/{id}/pause` · `/resume` · `/stop` | human in the loop |
| GET/POST | `/api/confirmations`, `/api/confirmations/{request_id}` | `{"answer": "approve"}` / a choice |
| GET | `/api/status`, `/api/applications`, `/api/agents`, `/api/agents/{desktop_id}/memory`, `/api/history` | |
| GET | `/api/screen.jpg` · POST `/api/observe` | current frame / current UI elements |
| WS | `/ws/agent` | events below; send `{"type":"command"|"respond"|"pause"|"resume"|"stop", ...}` |

The WebSocket sends these events: `task_started`, `plan_created`, `application_detected`, `application_opened`,
`agent_created`, `agent_message`, `state_changed`, `observation`, `screen_updated`, `action_started`,
`action_completed`, `verification_started`, `verification_completed`, `recovery`, `confirmation_required`,
`user_input_required`, `task_paused`/`resumed`, `task_completed`, `task_failed` and `task_stopped`.

## Logs

- `logs/master/master.log`: everything.
- `logs/applications/<app>.log`: every action in the spec's format (time, app, agent, action, target,
  coordinates, result).
- `logs/tasks/<task_id>.jsonl`: decisions (raw model output and timing), actions, messages and results.
- With `DEBUG=true` / `--debug`, `logs/tasks/<task_id>/` also gets a screenshot and the observation (OCR and
  elements) for each step.

## Limitations (honest list)

- **Wayland:** input works through `ydotool` if it's installed. Window listing and activation only see XWayland
  windows, and screenshots need `grim` or `gnome-screenshot`. Use an Xorg session for reliable control.
- **Windows and macOS** are not implemented yet. The backend seams are there: implement `WindowManager` and
  `ScreenCapture` (mouse and keyboard already work through pynput) and add a branch in `actions/platform.py`.
- **Accessibility inside apps:**
  - An already-running Chrome or VS Code started without `--force-renderer-accessibility` exposes little through
    accessibility, so the agent falls back to OCR plus the screenshot (slower, less precise).
  - Apps the agent launches itself get the flag.
- **Speed is bounded by the model.** A 27B vision model takes about 5–8 s per step on an RTX 3090. Smaller models
  are faster but make worse decisions.
