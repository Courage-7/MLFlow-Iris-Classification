# logging.py — Centralized logging setup

import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(
    name: str, level: int = logging.INFO, log_format: str = "text"
) -> logging.Logger:
    """Setup and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_format: 'text' for human-readable, 'json' for structured JSON.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        if log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
