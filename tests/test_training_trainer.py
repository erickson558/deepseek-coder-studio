from pathlib import Path

from app.gui import tasks as gui_tasks
from app.models.training import TrainingJobConfig
from app.training.config import save_training_config
from app.training.trainer import build_missing_training_dependencies_message


def test_build_missing_training_dependencies_message_is_actionable() -> None:
    message = build_missing_training_dependencies_message(["datasets", "transformers"])

    assert "datasets, transformers" in message
    assert "python -m pip install -e ." in message
    assert "llmstudio.exe" in message


def test_run_auto_training_prepares_missing_dataset_before_training(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    output_dir = tmp_path / "processed"
    train_file = output_dir / "train.jsonl"
    validation_file = output_dir / "validation.jsonl"

    config = TrainingJobConfig.model_validate(
        {
            "job_name": "demo-job",
            "strategy": "lora",
            "base_model": "demo-model",
            "train_file": str(train_file),
            "validation_file": str(validation_file),
        }
    )
    save_training_config(config_path, config)

    prepared_calls: list[tuple[str, str]] = []

    def fake_prepare_dataset(input_path: str, output_dir: str) -> dict[str, object]:
        prepared_calls.append((input_path, output_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_file.write_text("[]", encoding="utf-8")
        validation_file.write_text("[]", encoding="utf-8")
        return {"splits": {"train": 1, "validation": 1, "test": 0}}

    class FakeRunner:
        def run_job(self, training_config: TrainingJobConfig) -> dict[str, object]:
            return {
                "job_name": training_config.job_name,
                "adapter_output_dir": str(tmp_path / "adapter"),
                "merged_output_dir": None,
                "train_file": str(training_config.train_file),
                "validation_file": str(training_config.validation_file),
                "train_samples": 1,
                "validation_samples": 1,
            }

    monkeypatch.setattr(gui_tasks, "prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr(gui_tasks, "FineTuneRunner", lambda: FakeRunner())

    result = gui_tasks.run_auto_training("data/samples/sample_dataset.jsonl", str(output_dir), str(config_path))

    assert prepared_calls == [(Path("data/samples/sample_dataset.jsonl"), output_dir)]
    assert result["dataset_summary"] == {"splits": {"train": 1, "validation": 1, "test": 0}}
    assert result["train_file"] == str(train_file)
    assert result["validation_file"] == str(validation_file)
