"""ASGI middleware: request ids, JSON access logs, request metrics and API key auth.

Plain ASGI callables rather than ``BaseHTTPMiddleware`` keep the per-request overhead in microseconds, which
matters when the scoring path itself takes about a millisecond.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fraudnet.logging_utils import request_id_var
from fraudnet.serving.metrics import Metrics

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID = b"x-request-id"
API_KEY = b"x-api-key"
OPEN_PATHS = frozenset({"/health", "/metrics", "/", "/favicon.ico"})
logger = logging.getLogger("fraudnet.access")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _route(scope: Scope) -> str:
    path = getattr(scope.get("route"), "path", None)
    return path if isinstance(path, str) else "unmatched"


class RequestContextMiddleware:
    """Assign or propagate a request id, time the request, log one JSON line and count it."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap HTTP requests; other scope types pass through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, REQUEST_ID)
        request_id = incoming if incoming and len(incoming) <= 128 else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        status = {"code": 500}
        started = time.perf_counter()

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID, request_id.encode("latin-1")))
                headers.append(
                    (b"x-process-time-ms", f"{(time.perf_counter() - started) * 1000:.3f}".encode())
                )
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route, method = _route(scope), scope.get("method", "GET")
            self.metrics.requests_total.labels(method, route, str(status["code"])).inc()
            self.metrics.request_duration.labels(method, route).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "method": method,
                    "path": scope.get("path"),
                    "status": status["code"],
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )
            request_id_var.reset(token)


class ApiKeyMiddleware:
    """When a key is configured, require ``X-API-Key`` on every route except health, metrics and the page."""

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject with a JSON 401 when the key is missing or wrong (constant-time comparison)."""
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or self.api_key is None
            or path in OPEN_PATHS
            or path.startswith("/static/")
        ):
            await self.app(scope, receive, send)
            return
        supplied = _header(scope, API_KEY) or ""
        if hmac.compare_digest(supplied.encode(), self.api_key.encode()):
            await self.app(scope, receive, send)
            return
        body = json.dumps(
            {"detail": "invalid or missing API key", "request_id": scope.get("state", {}).get("request_id")}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
