"""Persistent desktop configuration stored next to the script or executable."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.logging import get_logger
from app.core.runtime import get_config_file

LOGGER = get_logger(__name__)


class GuiConfig(BaseModel):
    """Serializable GUI preferences and last-used values."""

    window_geometry: str = "1280x860+80+60"
    language: str = "es"
    auto_start_backend: bool = False
    auto_close_enabled: bool = False
    auto_close_seconds: int = 60
    host: str = "127.0.0.1"
    port: int = 8000
    dataset_input_path: str = "data/samples/sample_dataset.jsonl"
    dataset_output_path: str = "data/processed"
    training_config_path: str = "configs/training/lora.yaml"
    training_job_name: str = "deepseek-coder-lora-v0-2-0"
    training_strategy: str = "lora"
    training_base_model: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    training_train_file: str = "data/processed/train.jsonl"
    training_validation_file: str = "data/processed/validation.jsonl"
    training_num_train_epochs: int = 1
    training_per_device_train_batch_size: int = 1
    training_gradient_accumulation_steps: int = 8
    training_learning_rate: float = 2e-4
    training_max_seq_length: int = 2048
    training_merge_adapter: bool = True
    evaluation_config_path: str = "configs/evaluation/default.yaml"
    publish_repo_id: str = ""
    publish_token_env_var: str = "HF_TOKEN"
    publish_private_repo: bool = False
    publish_artifact_type: str = "adapter"
    publish_source_dir: str = ""
    last_adapter_output_dir: str = ""
    last_merged_output_dir: str = ""
    inference_task: str = "code_generation"
    inference_language: str = "python"
    inference_prompt: str = ""
    inference_context_file: str = ""
    selected_tab: str = "dashboard"


class GuiConfigStore:
    """Load and save desktop settings automatically."""

    def __init__(self, path: Path | None = None):
        self.path = path or get_config_file()

    def load(self) -> GuiConfig:
        """Load a saved config and fall back to defaults when needed."""
        if not self.path.exists():
            return GuiConfig()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return GuiConfig.model_validate(payload)
        except (OSError, ValidationError, json.JSONDecodeError):
            LOGGER.exception("Failed to load GUI config from %s", self.path)
            return GuiConfig()

    def save(self, config: GuiConfig) -> None:
        """Persist the current GUI state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
