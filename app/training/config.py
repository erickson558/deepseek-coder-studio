from pathlib import Path

import yaml

from app.models.training import TrainingJobConfig


def load_training_config(config_path: str | Path) -> TrainingJobConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainingJobConfig.model_validate(payload)
