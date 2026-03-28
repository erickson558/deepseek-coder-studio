"""Central logging configuration for CLI, API and desktop GUI modes."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.core.config import get_settings
from app.core.runtime import get_log_file

_LOGGING_CONFIGURED = False


def configure_logging(include_console: bool = True, force: bool = False) -> None:
    """Configure application logging and write logs to the runtime log file."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    settings = get_settings()
    log_file = get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: dict[str, dict[str, object]] = {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(log_file),
            "encoding": "utf-8",
        }
    }
    root_handlers = ["file"]
    if include_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
        root_handlers.append("console")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": handlers,
            "root": {
                "handlers": root_handlers,
                "level": settings.log_level.upper(),
            },
        }
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the shared project configuration."""
    return logging.getLogger(name)
