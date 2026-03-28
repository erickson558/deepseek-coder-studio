"""Fine-tuning runner for LoRA and QLoRA jobs."""

from pathlib import Path
from typing import Any

from app.core.exceptions import DependencyUnavailableError
from app.core.logging import get_logger
from app.models.training import TrainingJobConfig
from app.training.config import load_training_config
from app.training.data import build_text_records
from app.training.lora import build_lora_config, build_quantization_config
from app.training.merge import merge_adapter
from app.utils.files import ensure_directory, write_json

LOGGER = get_logger(__name__)


class FineTuneRunner:
    """Orchestrates dataset loading, tokenization, training and adapter export."""

    def run(self, config_path: str | Path) -> dict[str, Any]:
        """Load a YAML config and execute the configured training job."""
        config = load_training_config(config_path)
        return self.run_job(config)

    def run_job(self, config: TrainingJobConfig) -> dict[str, Any]:
        """Execute a fine-tuning job using the configured strategy."""
        try:
            import torch
            from datasets import Dataset
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Training requires torch, datasets, transformers and peft to be installed."
            ) from exc

        adapter_output_dir = ensure_directory(config.output_dir / config.job_name)
        logs_dir = ensure_directory(config.logs_dir / config.job_name)

        LOGGER.info("Loading tokenizer for base model %s", config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_records = build_text_records(config.train_file, tokenizer=tokenizer)
        validation_records = build_text_records(config.validation_file, tokenizer=tokenizer)

        train_dataset = Dataset.from_list(train_records)
        validation_dataset = Dataset.from_list(validation_records)

        def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            """Tokenize prompts and mirror input IDs into labels for causal LM training."""
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.max_seq_length,
                padding=False,
            )
            encoded["labels"] = [list(input_ids) for input_ids in encoded["input_ids"]]
            return encoded

        train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_dataset.column_names)
        validation_dataset = validation_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=validation_dataset.column_names,
        )

        quantization_config = build_quantization_config(config)
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if device_map:
            model_kwargs["device_map"] = device_map
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if config.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        LOGGER.info("Loading base model with strategy=%s", config.strategy)
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

        if config.strategy == "qlora":
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, build_lora_config(config))
        if config.gradient_checkpointing:
            # Gradient checkpointing reduces memory usage at the cost of extra compute.
            model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=str(adapter_output_dir),
            logging_dir=str(logs_dir),
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            max_grad_norm=config.max_grad_norm,
            bf16=config.bf16,
            fp16=config.fp16,
            report_to=config.report_to,
            save_strategy=config.save_strategy,
            eval_strategy=config.evaluation_strategy,
            load_best_model_at_end=config.load_best_model_at_end,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        LOGGER.info("Starting fine-tuning job %s", config.job_name)
        train_result = trainer.train()
        trainer.save_model(str(adapter_output_dir))
        tokenizer.save_pretrained(str(adapter_output_dir))

        merged_path = None
        if config.merge_adapter:
            # Export a merged model when downstream inference should not depend on PEFT adapters.
            merged_path = merge_adapter(
                base_model=config.base_model,
                adapter_dir=adapter_output_dir,
                output_dir=config.merged_output_dir / config.job_name,
            )

        summary = {
            "job_name": config.job_name,
            "strategy": config.strategy,
            "base_model": config.base_model,
            "adapter_output_dir": str(adapter_output_dir),
            "merged_output_dir": merged_path,
            "training_loss": train_result.training_loss,
            "train_samples": len(train_records),
            "validation_samples": len(validation_records),
        }
        write_json(Path(adapter_output_dir) / "training_summary.json", summary)
        return summary
