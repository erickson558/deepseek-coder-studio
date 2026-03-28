from pathlib import Path
from typing import Any

from app.models.dataset import DatasetExample
from app.training.formatting import render_messages
from app.utils.files import read_jsonl


def load_examples(path: str | Path) -> list[DatasetExample]:
    return [DatasetExample.model_validate(row) for row in read_jsonl(path)]


def build_text_records(path: str | Path, tokenizer: object | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in load_examples(path):
        text = render_messages(example.messages, tokenizer=tokenizer)
        records.append(
            {
                "id": example.id,
                "task": example.task.value,
                "language": example.language,
                "text": text,
            }
        )
    return records
