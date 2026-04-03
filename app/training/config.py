from pathlib import Path

import yaml

from app.models.training import TrainingJobConfig


def load_training_config(config_path: str | Path) -> TrainingJobConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainingJobConfig.model_validate(payload)


def save_training_config(config_path: str | Path, config: TrainingJobConfig) -> Path:
    """Persist a validated training config to YAML and return the target path."""
    target = Path(config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return target
