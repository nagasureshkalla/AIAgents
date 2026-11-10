"""Research-specific errors."""

from __future__ import annotations

from app.core.exceptions import AgentSystemError


class WebDisabledError(AgentSystemError):
    status_code = 409
    code = "web_disabled"
    error_type = "web_disabled"


class SearchError(AgentSystemError):
    status_code = 502
    code = "search_failed"


class UnsafeURLError(AgentSystemError):
    status_code = 400
    code = "unsafe_url"


class GPURequiredError(AgentSystemError):
    status_code = 503
    code = "gpu_required"
    error_type = "gpu_unavailable"
