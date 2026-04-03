"""Validation helpers for guided training workflows."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from app.models.training import TrainingJobConfig
from app.training.trainer import (
    build_missing_training_dependencies_message,
    get_missing_training_dependencies,
)


def validate_training_job_config(config: TrainingJobConfig) -> dict[str, Any]:
    """Validate the minimum requirements for a training job before execution."""
    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, message: str) -> None:
        checks.append({"name": name, "status": status, "message": message})

    if config.base_model.strip():
        add_check("base_model", "ok", f"Base model configured: {config.base_model}")
    else:
        add_check("base_model", "error", "A base model is required.")

    train_file = Path(config.train_file)
    if train_file.exists():
        add_check("train_file", "ok", f"Train split found: {train_file}")
    else:
        add_check("train_file", "error", f"Train split not found: {train_file}")

    validation_file = Path(config.validation_file)
    if validation_file.exists():
        add_check("validation_file", "ok", f"Validation split found: {validation_file}")
    else:
        add_check("validation_file", "error", f"Validation split not found: {validation_file}")

    missing_dependencies = get_missing_training_dependencies()
    if missing_dependencies:
        add_check(
            "dependencies",
            "error",
            build_missing_training_dependencies_message(missing_dependencies),
        )
    else:
        add_check("dependencies", "ok", "Training dependencies are installed.")

    if config.strategy == "qlora":
        if importlib.util.find_spec("bitsandbytes") is None:
            add_check(
                "bitsandbytes",
                "error",
                "QLoRA requires `bitsandbytes`, which is not currently installed.",
            )
        else:
            add_check("bitsandbytes", "ok", "bitsandbytes is available for QLoRA.")

        if sys.platform.startswith("win"):
            add_check(
                "platform",
                "warning",
                "QLoRA on native Windows is usually unreliable; LoRA is the safer default.",
            )

        if not missing_dependencies:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                add_check(
                    "cuda",
                    "warning",
                    "QLoRA normally expects CUDA; no GPU was detected in this environment.",
                )

    return {
        "valid": not any(check["status"] == "error" for check in checks),
        "checks": checks,
    }
