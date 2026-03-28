"""Runtime path helpers for source execution and packaged executables."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen_binary() -> bool:
    """Return True when the app is running from a packaged executable."""
    return bool(getattr(sys, "frozen", False))


def get_runtime_dir() -> Path:
    """Return the directory where runtime artifacts should live."""
    if is_frozen_binary():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_runtime_file(filename: str) -> Path:
    """Return an absolute runtime file path located next to the app entrypoint."""
    return get_runtime_dir() / filename


def get_config_file() -> Path:
    """Return the GUI configuration file path."""
    return get_runtime_file("config.json")


def get_log_file() -> Path:
    """Return the application log file path."""
    return get_runtime_file("log.txt")
