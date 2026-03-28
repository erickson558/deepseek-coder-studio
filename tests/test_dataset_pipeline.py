from pathlib import Path

from app.dataset.pipeline import prepare_dataset


def test_prepare_dataset_creates_expected_files(tmp_path: Path) -> None:
    input_path = Path("data/samples/sample_dataset.jsonl")
    output_dir = tmp_path / "processed"

    summary = prepare_dataset(input_path=input_path, output_dir=output_dir)

    assert summary["total_raw_records"] == 6
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "summary.json").exists()
