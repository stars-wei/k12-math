"""Small, safe error types for the local web interface."""

from __future__ import annotations


class UserVisibleError(RuntimeError):
    """An error whose message is safe and useful to show to a learner."""


class ConfigurationError(UserVisibleError):
    """A required local setting is missing."""


class UpstreamServiceError(UserVisibleError):
    """An external AI/OCR service could not complete a request."""


class GraphServiceError(UserVisibleError):
    """Neo4j could not be reached or did not accept a query."""


class ExecutionError(UserVisibleError):
    """A graph-backed calculation could not be completed safely."""


def friendly_message(error: Exception) -> str:
    """Return a short public message without exposing HTTP payloads or secrets."""
    if isinstance(error, UserVisibleError):
        return str(error)
    if isinstance(error, ValueError):
        return str(error)
    return "求解过程中发生未预期错误，请检查输入后重试。"
