from app.training.trainer import build_missing_training_dependencies_message


def test_build_missing_training_dependencies_message_is_actionable() -> None:
    message = build_missing_training_dependencies_message(["datasets", "transformers"])

    assert "datasets, transformers" in message
    assert "python -m pip install -e ." in message
    assert "llmstudio.exe" in message
