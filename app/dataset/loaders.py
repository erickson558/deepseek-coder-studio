from pathlib import Path
from typing import Any

from app.utils.files import read_csv, read_json, read_jsonl, read_text


def load_records(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir():
        return _load_directory_records(path)

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
            if isinstance(records, list):
                return records
        raise ValueError(f"unsupported JSON shape for {path}")
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"unsupported dataset source: {path}")


def _load_directory_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".json", ".jsonl", ".csv"}:
            records.extend(load_records(file_path))

    if records:
        return records

    for example_dir in sorted(candidate for candidate in directory.iterdir() if candidate.is_dir()):
        instruction_file = example_dir / "instruction.txt"
        response_file = example_dir / "response.txt"
        if not instruction_file.exists() or not response_file.exists():
            continue
        records.append(
            {
                "id": example_dir.name,
                "task": _optional_file(example_dir / "task.txt", default="code_generation"),
                "language": _optional_file(example_dir / "language.txt"),
                "system": _optional_file(example_dir / "system.txt"),
                "instruction": read_text(instruction_file),
                "input": _optional_file(example_dir / "input.txt"),
                "response": read_text(response_file),
                "context": _optional_file(example_dir / "context.txt"),
                "source": str(example_dir),
            }
        )
    return records


def _optional_file(path: Path, default: str | None = None) -> str | None:
    if path.exists():
        return read_text(path).strip()
    return default
