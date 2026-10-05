"""Task, chat, analysis, image-upload and WebSocket endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, WebSocket, WebSocketDisconnect, status

from app.api.deps import ContainerDep
from app.core.container import Container
from app.core.events import EventType, TaskEvent
from app.core.exceptions import AgentTimeoutError, InvalidImageError, InvalidRequestError, TaskNotFoundError
from app.schemas.common import Priority, TaskStatus
from app.schemas.result import ChatResponse, TaskResult
from app.schemas.task import AnalyzeRequest, ChatRequest, TaskCreatedResponse, TaskRecord, TaskRequest
from app.utils.files import safe_join, sanitize_filename
from app.utils.image import extension_for, validate_image_bytes

router = APIRouter(prefix="/api", tags=["tasks"])
ws_router = APIRouter(tags=["websocket"])


async def _wait(container: Container, request: TaskRequest, timeout: float) -> TaskRecord:
    try:
        return await container.run_and_wait(request, timeout)
    except TimeoutError:
        raise AgentTimeoutError(f"Task did not finish within {timeout:.0f}s and was cancelled") from None


# ---------------------------------------------------------------- async tasks
@router.post(
    "/tasks",
    response_model=TaskCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a task (text or image) for multi-agent processing",
    description=(
        "Returns immediately with a task id. Poll `GET /api/tasks/{task_id}` or subscribe to "
        "`ws://<host>/ws/tasks/{task_id}` for real-time agent events. `image_path` must point inside data/input/ "
        "(upload with `POST /api/image`)."
    ),
)
async def create_task(request: TaskRequest, container: ContainerDep) -> TaskCreatedResponse:
    record = container.submit(request)
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/tasks", response_model=list[TaskResult], summary="List recent tasks")
async def list_tasks(
    container: ContainerDep,
    status_filter: TaskStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[TaskResult]:
    return [TaskResult.from_record(r) for r in container.store.list(status_filter, limit)]


@router.get("/tasks/{task_id}", response_model=TaskResult, summary="Get task status, plan, agent results and answer")
async def get_task(task_id: str, container: ContainerDep) -> TaskResult:
    return TaskResult.from_record(container.store.get(task_id))


@router.get("/tasks/{task_id}/events", summary="Event history of a task")
async def task_events(task_id: str, container: ContainerDep) -> list[dict]:
    container.store.get(task_id)
    return [e.to_message() for e in container.events.history(task_id)]


@router.delete("/tasks/{task_id}", summary="Cancel a queued or running task")
async def cancel_task(task_id: str, container: ContainerDep) -> dict:
    cancelled = container.cancel(task_id)
    if cancelled:
        # Give the running coroutine a moment to observe the cancellation and record it.
        with contextlib.suppress(TimeoutError):
            await container.store.wait(task_id, timeout=2.0)
    record = container.store.get(task_id)
    return {"task_id": task_id, "cancelled": cancelled, "status": record.status}


# ---------------------------------------------------------------- synchronous helpers
@router.post("/chat", response_model=ChatResponse, summary="Send a message and wait for the final answer")
async def chat(request: ChatRequest, container: ContainerDep) -> ChatResponse:
    task_request = TaskRequest(
        prompt=request.message, image_path=request.image_path, priority=request.priority, options=request.options
    )
    record = await _wait(container, task_request, request.wait_timeout)
    answer = record.result.get("answer") if record.result else None
    return ChatResponse(
        task_id=record.task_id, status=record.status, answer=answer, error=record.error, duration=record.duration
    )


@router.post("/analyze", response_model=TaskResult, summary="Analyze text and/or an image and wait for the result")
async def analyze(request: AnalyzeRequest, container: ContainerDep) -> TaskResult:
    if not request.text and not request.image_path:
        raise InvalidRequestError("Provide `text`, `image_path`, or both")
    prompt = request.instruction if not request.text else f"{request.instruction}\n\n{request.text}"
    task_request = TaskRequest(prompt=prompt, image_path=request.image_path, priority=request.priority)
    record = await _wait(container, task_request, request.wait_timeout)
    return TaskResult.from_record(record)


@router.post(
    "/image",
    summary="Upload an image with a prompt and queue (or wait for) a vision/multimodal task",
    responses={202: {"model": TaskCreatedResponse}, 200: {"model": TaskResult}},
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_image(
    container: ContainerDep,
    response: Response,
    file: UploadFile = File(..., description="PNG/JPEG/WEBP/BMP/GIF image"),
    prompt: str = Form(default="Analyze this image and explain what is happening."),
    priority: str = Form(default="NORMAL"),
    wait: bool = Form(default=False, description="Block until the task finishes and return the result"),
    wait_timeout: float = Form(default=300, gt=0, le=3600),
):
    security = container.settings.security
    max_bytes = int(security.max_image_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise InvalidImageError(f"Image exceeds {security.max_image_mb} MB")
    fmt = validate_image_bytes(data, security)
    try:
        prio = Priority.parse(priority)
    except ValueError:
        raise InvalidRequestError(f"Invalid priority '{priority}'") from None

    stem = sanitize_filename(Path(file.filename or "upload").stem)
    target = safe_join(container.settings.input_dir, f"{uuid.uuid4().hex[:10]}_{stem}.{extension_for(fmt)}")
    await asyncio.to_thread(target.write_bytes, data)

    request = TaskRequest(prompt=prompt, image_path=str(target), priority=prio)
    if wait:
        record = await _wait(container, request, wait_timeout)
        response.status_code = status.HTTP_200_OK
        return TaskResult.from_record(record)
    record = container.submit(request)
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


# ---------------------------------------------------------------- websocket
@ws_router.websocket("/ws/tasks/{task_id}")
async def task_stream(websocket: WebSocket, task_id: str) -> None:
    """Stream task events (agent_started, agent_completed, ..., task_completed) as JSON messages.

    Past events are replayed first, so subscribing after the task has started loses nothing.
    The socket closes after a terminal event (task_completed / task_failed / task_cancelled).
    """
    container: Container = websocket.app.state.container
    await websocket.accept()
    try:
        record = container.store.get(task_id)
    except TaskNotFoundError:
        await websocket.send_json({"task_id": task_id, "event": "error", "error": "task not found"})
        await websocket.close(code=4404)
        return

    queue = container.events.subscribe(task_id, replay=True)
    try:
        if record.status.is_terminal and not any(e.is_terminal for e in container.events.history(task_id)):
            terminal = {
                TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
                TaskStatus.FAILED: EventType.TASK_FAILED,
                TaskStatus.CANCELLED: EventType.TASK_CANCELLED,
            }[record.status]
            await queue.put(TaskEvent(task_id=task_id, event=terminal, data={"result": record.result, "error": record.error}))
        while True:
            event = await queue.get()
            await websocket.send_json(event.to_message())
            if event.is_terminal:
                break
        await websocket.close()
    except WebSocketDisconnect:
        pass
    finally:
        container.events.unsubscribe(task_id, queue)
