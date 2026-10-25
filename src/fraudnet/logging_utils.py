"""Structured JSON logging with the request id carried in a context variable.

One JSON object per line can be shipped to any log pipeline without a parser, and reading the request id
from a context variable keeps the handlers free of logging plumbing.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any, TextIO

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON line, including any ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record; values that are not JSON types are converted with ``str``."""
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Install the JSON formatter on the root logger and route uvicorn's loggers through it."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True
