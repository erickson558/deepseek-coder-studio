from app.core.exceptions import DependencyUnavailableError
from app.models.training import TrainingJobConfig


def build_lora_config(config: TrainingJobConfig):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise DependencyUnavailableError("peft is required for LoRA fine-tuning") from exc

    return LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_quantization_config(config: TrainingJobConfig):
    if config.strategy != "qlora":
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise DependencyUnavailableError("transformers with bitsandbytes support is required for QLoRA") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
