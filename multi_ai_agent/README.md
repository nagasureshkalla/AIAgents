# Local Multi-AI Agent

A local-first, asynchronous multi-agent framework built on FastAPI, asyncio and Hugging Face models that run on your
own NVIDIA GPU, with automatic CPU fallback. Inference never calls cloud APIs.

```
USER ─► MANAGER ─► PLANNER ─┬─► TEXT ──────┐
                            ├─► VISION ────┼─► REASONING ─► VALIDATION ─► RESULT AGGREGATOR ─► USER
                            └─► MULTIMODAL ┘
```

* **Manager agent** reads the request (text, image, or both), works out the intent, and picks agents by their declared capabilities.
* **Planner agent** turns that decision into a dependency graph (DAG) and marks which steps can run in parallel.
* **Specialists**:
  * **Text**: summarize, classify, extract, format, generate.
  * **Vision**: scene description and object listing.
  * **Multimodal**: visual questions and screenshot debugging.
  * **Reasoning**: step-by-step interpretation.
  * **Research**: autonomous web research. It searches the internet, reads pages, cross-checks sources and answers with
    verified citations, while every LLM call runs on your GPU (see [§13](#13-web-research-agent)).
* **Validation agent** runs rule checks plus a short model review of the answer.
* **Result aggregator** uses the *base* model to combine everything into one final answer.
* **Resource scheduler** has a priority pool for each device. GPU inference is admitted one job at a time by default, and models are swapped by least-recent use, so VRAM doesn't run out (no OOM).

---

## 1. Project tree

```text
multi_ai_agent/
├── app/
│   ├── main.py                 # FastAPI app factory (python -m app.main / uvicorn app.main:app)
│   ├── cli.py                  # run.py CLI + startup banner
│   ├── api/                    # routes_tasks (tasks, chat, analyze, image, WS), routes_models, routes_agents, routes_health
│   ├── agents/                 # base_agent, registry (auto-discovery), manager, planner, text, vision, multimodal, reasoning, validation
│   ├── core/                   # config, device, logging, events, exceptions, container, local_guard (no cloud AI)
│   ├── models/                 # model_registry, model_manager (lifecycle/LRU/VRAM), text_model, vision_model, base,
│   │                           # engines/ (llama_cpp, vllm), downloader (HF Hub downloads in a subprocess)
│   ├── research/               # web research: search/, browser/, retrieval/, pipeline/, citations, memory
│   ├── orchestration/          # orchestrator (DAG), scheduler (priority + VRAM aware), task_queue, result_aggregator
│   ├── schemas/                # Pydantic: agent, task, message, result, common
│   └── utils/                  # image validation, safe paths, performance monitor, text parsing
├── models/{text,vision,multimodal}/   # downloaded Hugging Face models live here
├── data/{input,output,cache,tasks}/   # uploads, outputs, HF cache, task snapshots (JSON)
├── tests/                      # pytest suite (fake models, no GPU/downloads needed)
├── scripts/download_models.py  # fetch models into models/
├── scripts/benchmark.py        # in-process latency / tokens-per-second benchmark
├── config.yaml  .env  .env.example  run.py  pytest.ini
└── requirements.txt  requirements-cuda.txt  requirements-vision.txt  requirements-dev.txt
```

## 2. Installation (Linux + NVIDIA GPU)

Prerequisites: Python 3.12+, an NVIDIA driver recent enough for CUDA 12 (`nvidia-smi` works).

```bash
cd multi_ai_agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt            # core (PyPI torch on Linux includes CUDA 12)
pip install -r requirements-cuda.txt       # GPU utilization metrics (nvidia-ml-py)
pip install -r requirements-vision.txt     # optional processor extras (num2words)
pip install -r requirements-dev.txt        # pytest, ruff

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If your driver needs a different CUDA build of torch:
`pip install torch --index-url https://download.pytorch.org/whl/cu124` (pick the right `cuXXX`).

Run the tests. They use fake models, so no GPU or downloads are needed:

```bash
pytest -q
```

## 3. Model download

The default models are small and fast:

| key | Hugging Face id | folder | approx. VRAM (bf16) |
|---|---|---|---|
| `research` | `Qwen/Qwen2.5-7B-Instruct` | `models/text/research` | ~16 GB |
| `base` | `Qwen/Qwen2.5-1.5B-Instruct` | `models/text/base` | ~3.5 GB |
| `fast_text` | `Qwen/Qwen2.5-0.5B-Instruct` | `models/text/fast_text` | ~1.2 GB |
| `vision` | `HuggingFaceTB/SmolVLM-256M-Instruct` | `models/vision/fast_vision` | ~1 GB |
| `multimodal` | `HuggingFaceTB/SmolVLM-500M-Instruct` | `models/multimodal/fast_multimodal` | ~1.5 GB |

```bash
python scripts/download_models.py --all          # everything in config.yaml
python scripts/download_models.py --list         # see what is installed

# individual models
python scripts/download_models.py --model Qwen/Qwen2.5-0.5B-Instruct --type text
python scripts/download_models.py --model HuggingFaceTB/SmolVLM-256M-Instruct --type vision
python scripts/download_models.py --model HuggingFaceTB/SmolVLM-500M-Instruct --type multimodal
```

Only the download script goes online. If a model is already on disk it is skipped (use `--force` to download it
again). The server never downloads anything at startup. With `offline_mode: true` it sets `HF_HUB_OFFLINE=1`, loads only from
`models/`, and gives a clear error, including the download command, when a model is missing.
Gated models need `HF_TOKEN=...` set in your environment.

The minimum working setup is a single text model. `model_fallbacks` in `config.yaml` lets `base` and `fast_text`
stand in for each other, and `vision` and `multimodal` do the same.

## 4. Startup

```bash
python run.py                        # 127.0.0.1:8000, device auto (CUDA -> MPS -> CPU)
python run.py --device cuda
python run.py --device cpu
python run.py --port 9000
python -m app.main                   # same as run.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```text
========================================
          LOCAL MULTI AI AGENT
========================================

Server: http://127.0.0.1:8000
API Docs: http://127.0.0.1:8000/docs

Device: CUDA
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB (used 0.4 GB)
...
Models:
  ✓ base         text       Qwen/Qwen2.5-1.5B-Instruct
  ✓ fast_text    text       Qwen/Qwen2.5-0.5B-Instruct
  ✓ vision       vision     HuggingFaceTB/SmolVLM-256M-Instruct
  ✓ multimodal   multimodal HuggingFaceTB/SmolVLM-500M-Instruct

Agents:
  ✓ Manager
  ✓ Multimodal
  ✓ Planner
  ...
Status: READY
========================================
```

Interactive docs are at `http://127.0.0.1:8000/docs` (Swagger) and `/redoc`.

### Dashboard

Open `http://127.0.0.1:8000/` (it redirects to `/ui/`) for the live control room:

* **Agent pipeline (three.js)**: every agent is a 3D node. Running agents pulse amber with particles flowing in
  from their upstream agents, and turn green or red when they finish. Drag to orbit, scroll to zoom.
* **Running now**: each active agent, its step, and how long it has been running.
* **Resources**: CPU, RAM, GPU utilization and VRAM with 90-second sparklines, plus task, agent and GPU-slot counters.
* **Chat**: send a message (Enter sends, Shift+Enter adds a new line), optionally with an image (attach, drag and drop, or paste).
  Each reply shows the agents' progress as live step chips, then the final answer as formatted Markdown. Running replies have a Stop button.
* **Speech agent (voice)**: two ways to talk to the agents, both transcribed by local Whisper
  (`speech.path`, default `models/speech/whisper` = `openai/whisper-large-v3-turbo`, ~1.6 GB VRAM, loaded on first use
  and unloaded after 5 idle minutes).
  * **Hey CC** (hands-free): switch on the **Hey CC** button and the page keeps listening. Voice activity detection
    cuts what you say into utterances; only the ones that contain “Hey CC” are acted on (“Hey CC, what is the capital
    of France?”). Say “Hey CC” alone and it answers “Yes?” and takes your next sentence (within 8 s) as the request.
    Everything else is dropped after transcription and never stored. The setting is remembered per browser.
  * **🎤** (push-to-talk): one message, sent when you stop talking.
  * Replies to voice requests are read aloud by a local **Piper** voice (`speech.voice`, default
    `models/speech/piper`, en_US-lessac-medium, CPU), with the browser's on-device voices as a fallback. The mic is
    muted while it speaks. Click 🎤 or press Esc to stop; **🔊 Listen** replays any reply. Say “Hey CC, stop” to
    cancel. Models: `python scripts/download_models.py --model openai/whisper-large-v3-turbo --type speech --name
    whisper`; Piper voices come from huggingface.co/rhasspy/piper-voices (the `.onnx` plus its `.onnx.json`).
* **Task detail**: hidden until you click **Details →** on a reply. It slides in with the plan grouped by parallel stage,
  each agent's output, the answer and an event timeline. Close it with ✕, Esc, or by clicking outside it.
* **Help (ⓘ next to API docs)**: three tabs. *Agents* shows what each agent does, its model, the phrases that select it,
  and an example, with **Try example** and **Use only this agent** buttons. "Use only this agent" sends your chat messages
  straight to that agent until you clear the chip. *Models* shows each model's state, size and which agents use it, and how to add one.
  *Routing* shows the live intent → trigger phrases → agents table.
* **Models / Agent performance**: load and unload models; calls, errors, average latency and tokens/sec per agent.

The UI is plain HTML and ES modules in `app/ui/`, with three.js vendored in `app/ui/vendor/`. It needs no build
step and makes no internet requests. Live updates come from `WS /ws/events`, which streams every task's events.

## 5. API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/tasks` | Queue a text/image task → `{"task_id", "status": "queued"}` |
| GET | `/api/tasks/{id}` | Status, plan, per-agent results, final answer |
| POST | `/api/speech/transcribe` | Multipart `file` (webm/ogg/mp4/wav) → `{"text", "seconds"}` with local Whisper |
| GET | `/api/speech` | Voice input status (enabled, installed, loaded) |
| POST | `/api/speech/listen` | One utterance → `{"text", "woke", "command"}` (wake word “Hey CC”; `follow_up=true` after a bare wake word) |
| POST | `/api/speech/speak` | `{"text"}` → `audio/wav` spoken by the local Piper voice |
| GET | `/api/tasks/{id}/events` | Event history |
| DELETE | `/api/tasks/{id}` | Cancel a queued/running task |
| GET | `/api/tasks` | Recent tasks (`?status=completed`) |
| POST | `/api/chat` | Send a message and wait for the answer |
| POST | `/api/analyze` | Analyze `text` and/or `image_path` and wait for the result |
| POST | `/api/image` | Multipart image upload + prompt (`wait=true` to block, `agents=vision` to force agents) |
| GET | `/api/models` | Configured/discovered models, installed?, lifecycle state, device |
| POST | `/api/models/{name}/load` / `unload` | Manual model management (`?device=cpu`) |
| GET | `/api/agents` | Agents and declared capabilities |
| GET | `/api/routing` | Intents, their trigger phrases, and the agents each one runs |
| GET | `/api/health` | Device, GPU, VRAM/RAM, active agents, running tasks |
| GET | `/api/metrics` | Latency, tokens/sec, CPU/RAM/GPU/VRAM per agent and model |
| POST | `/api/research` | Start web research `{"query", "depth": "quick\|normal\|deep"}` → `{"research_id", "stream"}` |
| GET | `/api/research/{id}` | Research status, cited answer, confidence, verified claims, sources |
| GET | `/api/research/{id}/sources` | Every page visited (cited, read, failed) and the extracted evidence |
| GET | `/api/stream/{id}` | Server-Sent Events: `status`, `source`, `answer`, `error` |
| GET | `/api/gpu` | NVIDIA GPU(s), CUDA version, VRAM (driver + torch allocator), utilization |
| GET | `/api/status` | Local-only inference status: AI mode, model, GPU, web mode |
| POST | `/api/models/load` / `unload` | Body `{"name": "..."}` (same as the path-based routes) |
| POST | `/api/models/download` | Download `{"hf_id", "type"}` from the HF Hub into `models/` (progress: `GET /api/models/downloads`) |
| WS | `/ws/tasks/{id}` | Real-time events |
| WS | `/ws/events` | Real-time events for all tasks (used by the dashboard) |

Priorities are `CRITICAL`, `HIGH`, `NORMAL` and `LOW`. They order both the task queue and GPU admission.

## 6. Example text request

```bash
curl -s -X POST http://127.0.0.1:8000/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Summarize this document: Local inference keeps data private ...", "priority": "HIGH"}'
# {"task_id":"task_9f82c1a0b3d4","status":"queued"}

curl -s http://127.0.0.1:8000/api/tasks/task_9f82c1a0b3d4 | python -m json.tool

# Synchronous
curl -s -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message": "Explain why running two 7B models on one 8 GB GPU fails."}'
```

## 7. Example image request

```bash
# Upload (saved into data/input/ after validation)
curl -s -X POST http://127.0.0.1:8000/api/image \
  -F file=@./photo.jpg \
  -F prompt="Analyze this image and explain what is happening." \
  -F priority=HIGH

# Or reference a file already in data/input/
cp photo.png data/input/image.png
curl -s -X POST http://127.0.0.1:8000/api/tasks -H 'Content-Type: application/json' -d '{
  "prompt": "Analyze this image and explain what is happening.",
  "input_type": "image",
  "image_path": "./data/input/image.png"
}'

# Screenshot debugging (routes to the multimodal agent)
curl -s -X POST http://127.0.0.1:8000/api/image -F file=@screen.png \
  -F prompt="What is wrong with this screenshot?" -F wait=true
```

`image_path` must point inside `data/input/`. Other paths, traversal (`../`), non-images, files over `max_image_mb`, and
images over `max_image_pixels` are rejected with a 400.

## 8. Multi-agent execution example

For "Analyze this image and explain what is happening.", the manager picks
`intent=image_analysis → [vision, text, reasoning, validation]` and the planner produces:

```json
{
  "intent": "image_analysis",
  "steps": [
    {"id": "see",      "agent": "vision",     "depends_on": []},
    {"id": "intent",   "agent": "text",       "depends_on": []},
    {"id": "reason",   "agent": "reasoning",  "depends_on": ["see", "intent"]},
    {"id": "validate", "agent": "validation", "depends_on": ["reason", "see"], "optional": true}
  ],
  "parallel_groups": [["intent", "see"], ["reason"], ["validate"]]
}
```

`see` and `intent` start together as concurrent asyncio tasks. When both finish, `reason` receives their outputs as
`AgentMessage`s, then `validate` runs, then the aggregator writes the final answer with the base model.

Stream the events live with any WebSocket client:

```bash
pip install websockets
python - <<'EOF'
import asyncio, json, sys, websockets
async def main(task_id):
    async with websockets.connect(f"ws://127.0.0.1:8000/ws/tasks/{task_id}") as ws:
        async for msg in ws:
            e = json.loads(msg); print(e["event"], e.get("agent", ""), e.get("step_id", ""))
asyncio.run(main(sys.argv[1]))
EOF
```

```text
task_queued / task_started
agent_started manager      / agent_completed manager
agent_started planner      / agent_completed planner / plan_created
agent_started vision see   / agent_started text intent       <- parallel
agent_completed text intent / agent_completed vision see
agent_started reasoning reason / agent_completed reasoning reason
agent_started validation validate / agent_completed validation validate
aggregation_started
task_completed   {"result": {"answer": "...", "vision": {"objects": [...]}, "validation": {...}, ...}}
```

How failures are handled: each agent returns
`{"agent", "status", "result", "execution_time", "error"}` and never crashes the system. The orchestrator then picks one of these:

* **retry**: timeouts and transient errors, up to `max_retries` times.
* **fallback to CPU**: CUDA out-of-memory or other CUDA errors, when `allow_cpu_fallback` is on.
* **skip**: an optional step failed. The validation agent is told which step failed.
* **cancel**: a required step failed, so the task fails with a clear error.

## 9. Adding a new agent

1. Create `app/agents/code_agent.py`:

```python
from typing import Any
from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentTask

class CodeAgent(BaseAgent):
    name = "code"
    description = "Writes and explains code."
    capabilities = ["code_generation", "code_explanation"]
    keywords = ["python", "code", "function", "bug"]   # manager adds this agent when these appear
    default_model = "fast_text"                         # or set agents.code.model in config.yaml

    async def run(self, task: AgentTask) -> dict[str, Any]:
        context = self.render_upstream(task)            # results from upstream agents, if any
        out = await self.generate(task, task.prompt, system="You are an expert programmer.")
        return {"text": out.text, "language": "python"}
```

2. Restart. The registry finds every `BaseAgent` subclass in `app/agents/`, so you don't need to register it anywhere. Agents defined outside
   that package can use the `@register_agent` decorator.
3. Capabilities are what routing uses. Prompts that match `keywords` add the agent as a parallel step that the
   validator reviews. To make it part of an intent's core flow, add a `StepTemplate` that references its capability
   in `app/agents/planner_agent.py`. To force it for one request, use `"options": {"agents": ["code"]}`.
4. To disable an agent, set `agents: { code: { enabled: false } }`.

## 10. Adding a new Hugging Face model

```bash
python scripts/download_models.py --model Qwen/Qwen2.5-3B-Instruct --type text --name qwen3b
```

The folder `models/text/qwen3b` is auto-discovered as model `qwen3b`. To give it a stable key or tuning, add it to
`config.yaml`, then point an agent at it:

```yaml
models:
  qwen3b:
    path: ./models/text/qwen3b
    type: text                # text | vision | multimodal
    hf_id: Qwen/Qwen2.5-3B-Instruct
    max_tokens: 768
    dtype: auto               # auto | bfloat16 | float16 | float32
    est_size_gb: 7.0          # optional; otherwise estimated from weight files
    model_kwargs: {}          # passed to from_pretrained (e.g. {attn_implementation: sdpa})
agents:
  reasoning: { model: qwen3b }
```

Supported model types:

* **Text**: any causal LM that `AutoModelForCausalLM` can load.
* **Vision and multimodal**: image-text-to-text models such as SmolVLM, Idefics3, Qwen2-VL / Qwen2.5-VL and LLaVA, loaded through
  `AutoModelForImageTextToText` (or `AutoModelForVision2Seq` as a fallback) with `AutoProcessor`.
* Models that need custom code require `trust_remote_code: true` and `--allow-code` when downloading.

## 11. GPU / CPU configuration

| Setting | Where | Meaning |
|---|---|---|
| `runtime.device` / `--device` / `MAA_DEVICE` | auto, cuda, mps, cpu | Priority CUDA → MPS → CPU; unavailable choices fall back to CPU |
| `CUDA_VISIBLE_DEVICES` | env | Restrict visible GPUs; among visible GPUs the one with most free memory is used |
| `runtime.gpu_concurrency` | default 1 | Simultaneous GPU inferences. Raise only with spare VRAM |
| `runtime.vram_threshold` | default 0.85 | Before loading, evict LRU idle models until projected VRAM use is under this fraction |
| `runtime.max_loaded_models` | default 3 | Cap on resident models |
| `runtime.idle_unload_seconds` | default 0 | Unload models idle longer than this (0 = off) |
| `runtime.allow_cpu_fallback` | default true | On CUDA OOM (load or inference), retry on CPU |
| `runtime.preload` | `[fast_text]` | Models loaded at startup |
| `models.*.dtype` | auto | bf16 on GPUs that support it, else fp16; fp32 on CPU |

To run CPU-only, use `python run.py --device cpu`, or set `MAA_DEVICE=cpu`. Expect slower responses, and keep to the 0.5B–1.5B models.

To expose the server on your network, set `server.host: 0.0.0.0` **and** `server.allow_public: true` (or use
`--host 0.0.0.0 --allow-public`). The API has no authentication, so put it behind a reverse proxy with auth.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Model 'base' is not installed at ...` | Run the download command from the error, or `python scripts/download_models.py --all` |
| Banner shows `Device: CPU` on a GPU box | `python -c "import torch; print(torch.cuda.is_available())"` → if False, install a torch build matching your driver (see §2), check `nvidia-smi`, `CUDA_VISIBLE_DEVICES` |
| `CUDA out of memory` | Lower `gpu_concurrency` to 1, lower `vram_threshold`/`max_loaded_models`, use smaller models, set `est_size_gb` for big models; OOM steps automatically retry on CPU |
| Vision model fails to load (`Unrecognized configuration class`) | Upgrade within the pinned range: `pip install -U "transformers>=4.49,<5"`; check the model is image-text-to-text |
| `ImportError: num2words` | `pip install -r requirements-vision.txt` |
| Model tries to reach huggingface.co | Keep `offline_mode: true`; ensure the folder contains `config.json` + `*.safetensors` |
| `Refusing to bind to 0.0.0.0` | Intentional; add `allow_public: true` |
| HTTP 413 | Body larger than `security.max_request_mb` |
| HTTP 400 `invalid_image` | Not a valid PNG/JPEG/WEBP/BMP/GIF, too large, or too many pixels |
| Slow first request | First use loads the model; add it to `runtime.preload` |
| Answers are low quality | Use larger models for `base` (e.g. Qwen2.5-3B/7B-Instruct) if VRAM allows; set `use_llm_routing: true` |
| Logs as JSON | `server.json_logs: true` or `MAA_JSON_LOGS=true` |
| Benchmark | `python scripts/benchmark.py --runs 5 --image ./data/input/image.png` |

## 13. Web research agent

The internet supplies the information and your NVIDIA GPU supplies the intelligence. The research agent never
calls a cloud LLM: the only traffic leaving the machine is web search queries and page downloads.

```
question ─► PLAN (fast model) ─► SEARCH (DuckDuckGo) ─► READ (httpx / Playwright) ─► EXTRACT evidence + quotes
              ▲                                                                          │
              └──── not enough? new queries (fast model) ◄── ANALYZE ◄───────────────────┘
                                                              │ enough / budget used
                                                              ▼
                                   VERIFY across sources (SUPPORTED / CONFLICTING / INSUFFICIENT)
                                                              ▼
                                   ANSWER (reasoning model) + code-built Confidence and Sources
```

**How to use it**

* In the dashboard, switch on **🌐 Web** next to the message box and pick a depth (quick, normal or deep). Questions
  about recent events ("latest …", "news …", "search the web …") are routed to the research agent automatically.
* Through the API:

```bash
curl -X POST http://127.0.0.1:8000/api/research -H 'Content-Type: application/json' \
  -d '{"query": "What are the latest open source AI video models?", "depth": "deep"}'
curl -N http://127.0.0.1:8000/api/stream/<research_id>          # live status / source / answer events
curl http://127.0.0.1:8000/api/research/<research_id>/sources   # every visited page + evidence
```

**What every answer contains:** `## Answer`, `## Key Findings` (each finding with `[n]` citations), a
`## Disagreements Between Sources` section when sources conflict, `## Confidence` (High, Medium or Low, computed from the
verification results), and `## Sources`.

**Citation guarantees**

* A source gets a number only after the browser has fetched it and extracted readable text.
* Each extracted claim must come with a quote that actually appears on the page; claims that can't be traced to a quote are dropped.
* Any `[n]` the model writes that doesn't point to a visited source is removed, and so is any URL the model writes.
  The Sources list is built from the fetch records, never from model output.

**Safety while browsing:** the agent opens http(s) only and refuses private, loopback and link-local addresses, also when a redirect
points there. It honours `robots.txt`, caps page size (`MAX_PAGE_SIZE_MB`) and times out slow pages (`REQUEST_TIMEOUT`).
CAPTCHAs, login walls and paywalls are reported as *blocked* and never bypassed.

**Models.** `research.fast_model` (default `fast_text`) writes search queries and checks for gaps. `research.reasoning_model`
(default `research`, Qwen2.5-7B-Instruct) extracts evidence, verifies claims and writes the answer. If only one
model is installed it does everything, through `model_fallbacks`. To point at any folder you download:
`MODEL_PATH=./models/text/Qwen3-8B` and `FAST_MODEL_PATH=./models/text/Qwen3-4B`. A folder that contains a `.gguf` file
uses the llama.cpp engine; set `engine: vllm` on a model to use vLLM. Both engines are optional installs, see
`requirements-cuda.txt`.

**Modes**

| Mode | Setting | LLM | Web |
|---|---|---|---|
| LOCAL AI + WEB | `WEB_ENABLED=true` (default) | NVIDIA GPU | search + public pages |
| LOCAL AI + OFFLINE | `WEB_ENABLED=false` | NVIDIA GPU | disabled; answers come from the model with an explicit notice and no citations |

**GPU requirement.** Research inference needs CUDA by default. Without it the agent fails with *"NVIDIA CUDA GPU was not
detected. Local AI inference cannot start."* For debugging only, `research.allow_cpu: true` permits CPU. `CUDA_DEVICE=N`
pins a GPU, and `CUDA_VISIBLE_DEVICES` is respected. `GPU_MODE=multi` is reserved for future multi-GPU support.

**Privacy.** `AI_MODE` must be `local`, otherwise the server refuses to start. At startup, cloud LLM credentials
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) are removed from the process environment and the server
logs *"Cloud AI providers are disabled. All inference is running locally."* `GET /api/status` reports the same.

**Optional extras:** `pip install -r requirements-browser.txt && playwright install chromium` enables headless
Chromium for JavaScript-heavy pages (used automatically when a static fetch returns almost no text). If you run
your own SearXNG instance, set `SEARCH_PROVIDER=searxng` and `SEARXNG_URL=...`.


---

## 14. Control Panel agent (desktop control)

The **Control Panel** agent (`control_panel`) operates the applications on your computer. It opens apps, clicks,
types and browses, and verifies every step on screen. It delegates the screen work to the sibling
[`computer_ai_agent`](../computer_ai_agent/README.md) service, which has its own venv and a live-screen UI at
http://127.0.0.1:8765. The model lifecycle stays in this pipeline.

```text
"Open Chrome, go to Google, search for NVIDIA RTX 3090 and open the first result"
  manager  → intent computer_control → agents [control_panel] → requires model control_panel
           → unloads idle models that are not needed (to make room) → loads control_panel (Ollama qwen3.6:27b)
  planner  → one step: control_panel
  control_panel → holds its model while working → starts computer_ai_agent if needed → streams each action,
                  asks you before destructive actions → returns the verified result
```

**Routing.** These go to the Control Panel:
- "open / launch / close / switch to <app>…"
- "start/run/use <app>"
- click / scroll / drag
- "on my computer / desktop"
- "<do something> in Chrome / VS Code / Terminal / Files…"

These checks run before web research, so *"Open Chrome and search Google for X"* acts on the desktop. Questions
and writing, such as *"Explain memory leaks in Chrome"* or *"Write a LinkedIn post"*, are not routed there.

**Model.** Configured in `config.yaml` as an `engine: ollama` model, so it's managed like the Hugging Face models:
```yaml
models:
  control_panel: { engine: ollama, ollama_model: qwen3.6:27b, type: multimodal, est_size_gb: 18,
                   model_kwargs: { num_ctx: 16384 } }
agents:
  control_panel: { model: control_panel, timeout_seconds: 1800 }
```
- Run `ollama pull qwen3.6:27b` first.
- **Loading** asks Ollama to keep the model in VRAM indefinitely (`keep_alive=-1m`), so this ModelManager, not
  Ollama's idle timer, decides when it leaves.
- **Unloading** uses `keep_alive=0` and waits until the VRAM is actually free.
- The service is started with the same model and `num_ctx`, so Ollama never reloads behind the manager's back.
- `inference.ollama_url` must point at this computer; a remote URL is refused at startup.

**Requirement-driven loading.** After choosing agents, the manager asks each one for its `required_models()` and
calls `ModelManager.prepare()`:
1. Idle models that aren't required are unloaded.
   - `runtime.model_policy: on_demand` (the default) unloads only until the required models fit. It picks a single
     model that frees enough room when there is one; otherwise it goes least-recently-used first.
   - `exclusive` always unloads every model that isn't required.
2. Required models that aren't resident are loaded, protected from evicting each other.
3. Models running inference for another task are never unloaded.

The decision reports `models: {required, loaded, unloaded, resident, skipped}`, and the chat shows
"Unloading model … · Loading model …" while it happens. Verified on an RTX 3090:

| Task | Before | Manager did | After |
|---|---|---|---|
| *Open Calculator and calculate 23 * 19* → **437** | research 15 GB + fast_text | unloaded research (only), loaded control_panel | fast_text + control_panel (19.6 GB) |
| *What is the latest stable Python release?* | fast_text + control_panel | unloaded control_panel (Ollama) + fast_text, loaded research | research (16.2 GB) |

**Chat UI.**
- While the agent works, the turn shows its live steps (▶ / ✓ / ✗ / ↻).
- **Approve / Deny** prompts appear inline, with a screenshot (answered through `POST /api/control-panel/requests/{id}`).
- A link opens the Control Panel's live screen.
- After the answer, a collapsible list shows every desktop action.

**Service settings** (`control_panel:` section):
- `url`
- `autostart` (on first use)
- `display` (e.g. `":7"` for a nested Xephyr test screen; also `MAA_CONTROL_PANEL_DISPLAY`)
- `project_dir`
- `require_confirmation`

When this server shuts down, it unloads the model and stops a service it started.

**Safety.** A failed desktop task is never retried automatically (`error_type: control_failed`), because repeating
GUI actions isn't safe. Cancelling the pipeline task stops the desktop task.

Tests: `pytest -q tests/test_control_panel.py`. They cover routing, prepare() eviction choices, the Ollama engine
against a fake Ollama server, and the agent end to end with a scripted service (held model, approval forwarding, no
retry, cancellation).

## 15. Job Agent: LinkedIn and website applications

"Apply to the top 3 Python developer jobs" searches the job boards (email applications through Auto-Apply AI)
**and LinkedIn**, scores everything against your resume with the local model, then applies:

1. **Email**: postings that publish a contact address. You approve each letter (Gmail must be connected).
2. **LinkedIn Easy Apply**: in a visible Chrome window with its own profile (`job_agent.browser_profile_dir`).
   The first time, it asks you to log in to LinkedIn in that window; the agent never handles your LinkedIn password.
3. **Company websites**: LinkedIn's external **Apply** and job-board links open the employer's site.

The browser engine (`app/agents/job_browser.py`) reads the page's HTML at every step: form controls with their
labels, types, options and required/invalid state, the buttons, and signals (login wall, CAPTCHA, confirmation). It
fills contact facts and the resume by rule, reuses your saved answers, asks the local model for screening questions
(it answers only from your resume facts and says "null" otherwise), and asks you for the rest in the chat. Your
answers are saved in `job_agent.answers_file` and reused. Then it clicks Next / Review until Submit.

* **Before Submit** it always stops and shows every answer it entered plus a screenshot. Nothing is submitted
  without your **Submit**, and every submitted application is recorded in Auto-Apply AI, with the same duplicate
  and daily-limit checks as email applications.
* **Sign-in / sign-up**: when a site needs an account, it asks first, then creates one with your email and a
  generated password saved in `job_agent.accounts_file` (mode 600, on this computer only). Verification codes are
  asked in the chat; for a verification link you click it and press Continue. Passwords are typed by code and never
  shown to the model.
* **Never automated**: CAPTCHAs (you solve them in the window), government IDs, bank/card numbers and dates of birth
  (left empty, or you type them yourself). EEO questions get "decline to self-identify". Optional cookies are declined.
* Limits: `job_agent.max_browser_applications` per request (default 5) and the Auto-Apply daily limit.
* LinkedIn's terms do not allow automated tools, and the automated Chrome identifies itself as automated, so your
  LinkedIn account may be restricted. Keep requests small. Turn LinkedIn off with `job_agent.linkedin: false`,
  or all browser applying with `job_agent.browser_apply: false`.

Needs Playwright in the venv (`pip install -r requirements-browser.txt`) and Google Chrome (`browser_channel: chrome`).
