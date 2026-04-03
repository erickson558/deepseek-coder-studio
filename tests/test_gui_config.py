from pathlib import Path

from app.gui.config import GuiConfig, GuiConfigStore


def test_gui_config_store_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    store = GuiConfigStore(path=config_path)
    config = GuiConfig(
        language="en",
        auto_start_backend=True,
        auto_close_enabled=True,
        auto_close_seconds=90,
        host="127.0.0.1",
        port=9000,
        training_job_name="custom-job",
        training_strategy="qlora",
        publish_repo_id="erickson558/custom-llm",
        publish_artifact_type="merged",
    )

    store.save(config)
    loaded = store.load()

    assert loaded.language == "en"
    assert loaded.auto_start_backend is True
    assert loaded.auto_close_enabled is True
    assert loaded.auto_close_seconds == 90
    assert loaded.port == 9000
    assert loaded.training_job_name == "custom-job"
    assert loaded.training_strategy == "qlora"
    assert loaded.publish_repo_id == "erickson558/custom-llm"
    assert loaded.publish_artifact_type == "merged"
