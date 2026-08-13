"""Registry that maps trusted Neo4j operation IDs to Python handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


OperationHandler = Callable[[dict[str, Any]], None]


class OperationRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, OperationHandler] = {}

    def register(self, operation_id: str) -> Callable[[OperationHandler], OperationHandler]:
        def decorator(handler: OperationHandler) -> OperationHandler:
            if operation_id in self._handlers:
                raise ValueError(f"操作已注册：{operation_id}")
            self._handlers[operation_id] = handler
            return handler

        return decorator

    def execute(self, operation_id: str, state: dict[str, Any]) -> None:
        try:
            handler = self._handlers[operation_id]
        except KeyError as exc:
            raise LookupError(f"未注册的 Neo4j 操作：{operation_id}") from exc
        handler(state)

    def contains(self, operation_id: str) -> bool:
        """Return whether a graph operation has a trusted local handler."""
        return operation_id in self._handlers
