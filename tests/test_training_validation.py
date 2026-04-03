from pathlib import Path

import pytest

from app.models.training import TrainingJobConfig
from app.training.validation import validate_training_job_config


def test_validate_training_job_config_reports_missing_train_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation_file = tmp_path / "validation.jsonl"
    validation_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.training.validation.get_missing_training_dependencies",
        lambda: [],
    )

    report = validate_training_job_config(
        TrainingJobConfig(
            base_model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            train_file=tmp_path / "train.jsonl",
            validation_file=validation_file,
        )
    )

    assert report["valid"] is False
    assert any(
        check["name"] == "train_file" and check["status"] == "error"
        for check in report["checks"]
    )


def test_validate_training_job_config_passes_for_existing_lora_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text("{}", encoding="utf-8")
    validation_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.training.validation.get_missing_training_dependencies",
        lambda: [],
    )

    report = validate_training_job_config(
        TrainingJobConfig(
            base_model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            train_file=train_file,
            validation_file=validation_file,
        )
    )

    assert report["valid"] is True
    assert all(check["status"] != "error" for check in report["checks"])
