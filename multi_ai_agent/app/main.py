"""FastAPI application factory and entry point (`python -m app.main` or `uvicorn app.main:app`)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import routes_agents, routes_health, routes_models, routes_tasks
from app.api.middleware import RequestSizeLimitMiddleware
from app.core.config import Settings, load_settings
from app.core.container import Container
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging

DESCRIPTION = """
Local, asynchronous **multi-agent** system running Hugging Face models on your own GPU/CPU.

**Flow:** Manager agent → Planner agent → specialist agents (Text, Vision, Multimodal, Reasoning) running
concurrently where dependencies allow → Validation agent → Result aggregator.

* **Tasks** – `POST /api/tasks` queues text or image work and returns a `task_id`; poll `GET /api/tasks/{id}`.
* **Image input** – `POST /api/image` (multipart upload) or `image_path` inside `data/input/`.
* **Text input** – `POST /api/chat` / `POST /api/analyze` wait for the final answer.
* **WebSocket** – `ws://127.0.0.1:8000/ws/tasks/{task_id}` streams `agent_started`, `agent_completed`, …,
  `task_completed` events.
* **Models** – list / load / unload local models. **Agents** – list agents and their capabilities.
* **Health** – device, VRAM/RAM, active agents, running tasks, and performance metrics.
"""

OPENAPI_TAGS = [
    {"name": "tasks", "description": "Submit, inspect and cancel multi-agent tasks (text and image input)."},
    {"name": "websocket", "description": "Real-time task events at `/ws/tasks/{task_id}`."},
    {"name": "agents", "description": "Registered agents and their declared capabilities."},
    {"name": "models", "description": "Locally installed Hugging Face models and their lifecycle state."},
    {"name": "health", "description": "Device, memory, task and performance monitoring."},
]


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.server.log_level, settings.server.json_logs)
    factory = container_factory or Container.build

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = factory(settings)
        app.state.container = container
        await container.start()
        try:
            yield
        finally:
            await container.stop()

    app = FastAPI(
        title="Local Multi-AI Agent",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=int(settings.security.max_request_mb * 1024 * 1024))
    register_exception_handlers(app)
    app.include_router(routes_tasks.router)
    app.include_router(routes_tasks.ws_router)
    app.include_router(routes_agents.router)
    app.include_router(routes_models.router)
    app.include_router(routes_health.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"name": "Local Multi-AI Agent", "version": __version__, "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()


def main() -> None:
    from app.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
