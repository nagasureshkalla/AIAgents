"""Domain exceptions and their HTTP mapping."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AgentSystemError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotFoundError(AgentSystemError):
    status_code = 404
    code = "model_not_found"


class ModelLoadError(AgentSystemError):
    status_code = 500
    code = "model_load_error"


class GPUOutOfMemoryError(AgentSystemError):
    status_code = 503
    code = "gpu_out_of_memory"


class InsufficientResourcesError(AgentSystemError):
    status_code = 503
    code = "insufficient_resources"


class ModelInUseError(AgentSystemError):
    status_code = 409
    code = "model_in_use"


class InvalidImageError(AgentSystemError):
    status_code = 400
    code = "invalid_image"


class InvalidRequestError(AgentSystemError):
    status_code = 400
    code = "invalid_request"


class AgentNotFoundError(AgentSystemError):
    status_code = 404
    code = "agent_not_found"


class AgentTimeoutError(AgentSystemError):
    status_code = 504
    code = "agent_timeout"


class TaskNotFoundError(AgentSystemError):
    status_code = 404
    code = "task_not_found"


class TaskExecutionError(AgentSystemError):
    status_code = 500
    code = "task_failed"


class TaskCancelledError(AgentSystemError):
    status_code = 409
    code = "task_cancelled"


def classify_error(exc: BaseException) -> str:
    """Map an exception to a coarse error type used by the orchestrator's recovery policy."""
    from app.core.device import is_cuda_error, is_oom_error

    if isinstance(exc, GPUOutOfMemoryError) or is_oom_error(exc):
        return "oom"
    if isinstance(exc, ModelNotFoundError):
        return "missing_model"
    if isinstance(exc, ModelLoadError):
        return "model_load"
    if isinstance(exc, (InvalidImageError, InvalidRequestError)):
        return "invalid_input"
    if isinstance(exc, (TimeoutError, AgentTimeoutError)):
        return "timeout"
    if isinstance(exc, InsufficientResourcesError):
        return "resources"
    if is_cuda_error(exc):
        return "cuda"
    return "agent_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentSystemError)
    async def _handle_agent_error(_: Request, exc: AgentSystemError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("request failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "details": exc.details},
        )
