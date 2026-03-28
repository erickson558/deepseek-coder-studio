from pathlib import Path

from app.core.logging import get_logger
from app.utils.files import ensure_directory

LOGGER = get_logger(__name__)


def merge_adapter(base_model: str, adapter_dir: str | Path, output_dir: str | Path) -> str:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and peft are required to merge adapters") from exc

    target_dir = ensure_directory(output_dir)
    LOGGER.info("Merging adapter %s into base model %s", adapter_dir, base_model)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()

    merged.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    return str(target_dir)
