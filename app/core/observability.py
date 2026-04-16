"""Structured logging helpers for runtime observability."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    """Format logs as one JSON object per line."""

    def __init__(self, service: str, version: str):
        super().__init__()
        self.service = service
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "version": self.version,
            "message": record.getMessage(),
            "event": getattr(record, "event", record.getMessage()),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(service: str, version: str):
    """Configure root logging once for structured output."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter(service=service, version=version))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any):
    logger.log(level, event, extra={"event": event, **fields})


__all__ = [
    "JsonLogFormatter",
    "configure_logging",
    "get_logger",
    "log_event",
]
