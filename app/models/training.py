from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LoraConfigModel(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingJobConfig(BaseModel):
    version: str = "0.1"
    job_name: str = "deepseek-coder-lora"
    strategy: Literal["lora", "qlora"] = "lora"
    base_model: str
    train_file: Path
    validation_file: Path
    output_dir: Path = Path("outputs/adapters")
    merged_output_dir: Path = Path("outputs/merged")
    logs_dir: Path = Path("outputs/logs")
    report_to: list[str] = Field(default_factory=list)
    max_seq_length: int = 2048
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = True
    save_strategy: Literal["steps", "epoch"] = "steps"
    evaluation_strategy: Literal["steps", "epoch"] = "steps"
    load_best_model_at_end: bool = False
    dataset_text_field: str = "text"
    merge_adapter: bool = True
    use_flash_attention_2: bool = False
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
