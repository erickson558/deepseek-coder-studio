from app.training.hub import build_model_card


def test_build_model_card_includes_summary_details() -> None:
    content = build_model_card(
        repo_id="erickson558/custom-llm",
        artifact_type="adapter",
        summary={
            "base_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "strategy": "lora",
            "train_samples": 128,
            "validation_samples": 16,
            "training_loss": 0.42,
        },
    )

    assert "# erickson558/custom-llm" in content
    assert "base_model: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in content
    assert "- peft" in content
    assert "- Training strategy: `lora`" in content
    assert "- Final training loss: `0.42`" in content
