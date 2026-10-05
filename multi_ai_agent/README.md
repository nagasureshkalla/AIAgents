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
│   ├── core/                   # config, device, logging, events, exceptions, container (dependency injection)
│   ├── models/                 # model_registry, model_manager (lifecycle/LRU/VRAM), text_model, vision_model, base
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

## 5. API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/tasks` | Queue a text/image task → `{"task_id", "status": "queued"}` |
| GET | `/api/tasks/{id}` | Status, plan, per-agent results, final answer |
| GET | `/api/tasks/{id}/events` | Event history |
| DELETE | `/api/tasks/{id}` | Cancel a queued/running task |
| GET | `/api/tasks` | Recent tasks (`?status=completed`) |
| POST | `/api/chat` | Send a message and wait for the answer |
| POST | `/api/analyze` | Analyze `text` and/or `image_path` and wait for the result |
| POST | `/api/image` | Multipart image upload + prompt (`wait=true` to block) |
| GET | `/api/models` | Configured/discovered models, installed?, lifecycle state, device |
| POST | `/api/models/{name}/load` / `unload` | Manual model management (`?device=cpu`) |
| GET | `/api/agents` | Agents and declared capabilities |
| GET | `/api/health` | Device, GPU, VRAM/RAM, active agents, running tasks |
| GET | `/api/metrics` | Latency, tokens/sec, CPU/RAM/GPU/VRAM per agent and model |
| WS | `/ws/tasks/{id}` | Real-time events |

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
