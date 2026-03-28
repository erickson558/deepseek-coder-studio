from pathlib import Path
from typing import Any

from app.dataset.formatter import normalise_record
from app.dataset.loaders import load_records
from app.dataset.splitter import split_dataset
from app.dataset.validator import validate_examples
from app.models.dataset import DatasetExample
from app.utils.files import ensure_directory, write_json, write_jsonl


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    raw_records = load_records(input_path)
    normalised_examples: list[DatasetExample] = []
    invalid_rows: list[dict[str, str]] = []

    for index, raw in enumerate(raw_records):
        try:
            normalised_examples.append(normalise_record(raw))
        except Exception as exc:  # noqa: BLE001
            invalid_rows.append({"index": str(index), "reason": str(exc)})

    validation = validate_examples(normalised_examples)
    invalid_rows.extend(
        {"index": issue.example_id, "reason": issue.reason} for issue in validation.invalid_examples
    )

    train_split, validation_split, test_split = split_dataset(
        validation.valid_examples,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    target_dir = ensure_directory(output_dir)

    write_jsonl(target_dir / "train.jsonl", [_to_record(item) for item in train_split])
    write_jsonl(target_dir / "validation.jsonl", [_to_record(item) for item in validation_split])
    write_jsonl(target_dir / "test.jsonl", [_to_record(item) for item in test_split])
    write_json(target_dir / "summary.json", _summary_payload(raw_records, train_split, validation_split, test_split, invalid_rows))

    return _summary_payload(raw_records, train_split, validation_split, test_split, invalid_rows)


def _to_record(example: DatasetExample) -> dict[str, Any]:
    return example.model_dump(mode="json")


def _summary_payload(
    raw_records: list[dict[str, Any]],
    train_split: list[DatasetExample],
    validation_split: list[DatasetExample],
    test_split: list[DatasetExample],
    invalid_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "total_raw_records": len(raw_records),
        "total_valid_records": len(train_split) + len(validation_split) + len(test_split),
        "total_invalid_records": len(invalid_rows),
        "splits": {
            "train": len(train_split),
            "validation": len(validation_split),
            "test": len(test_split),
        },
        "invalid_rows": invalid_rows,
    }
