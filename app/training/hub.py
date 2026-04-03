"""Helpers for publishing trained artifacts to the Hugging Face Hub."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from app.core.exceptions import ConfigurationError, DependencyUnavailableError
from app.core.logging import get_logger
from app.utils.files import read_json, write_text

LOGGER = get_logger(__name__)
REQUIRED_HUB_DEPENDENCIES = ("huggingface_hub",)


def get_missing_hub_dependencies() -> list[str]:
    """Return the Hub client packages that are not importable."""
    return [name for name in REQUIRED_HUB_DEPENDENCIES if importlib.util.find_spec(name) is None]


def build_missing_hub_dependencies_message(missing: list[str]) -> str:
    """Create an actionable error message for missing Hub dependencies."""
    packages = ", ".join(missing)
    return (
        f"Hugging Face Hub dependencies are missing: {packages}. "
        "Install them with `python -m pip install -e .`."
    )


def publish_training_artifacts(
    repo_id: str,
    source_dir: str | Path,
    token_env_var: str = "HF_TOKEN",
    private: bool = False,
    artifact_type: str = "adapter",
) -> dict[str, Any]:
    """Upload a trained adapter or merged model folder to a Hugging Face model repo."""
    missing_dependencies = get_missing_hub_dependencies()
    if missing_dependencies:
        raise DependencyUnavailableError(build_missing_hub_dependencies_message(missing_dependencies))

    repo_id = repo_id.strip()
    if not repo_id or "/" not in repo_id:
        raise ConfigurationError("Use a valid Hugging Face repo id in the form `owner/name`.")

    target_dir = Path(source_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ConfigurationError(f"Artifact folder not found: {target_dir}")

    token_name = token_env_var.strip() or "HF_TOKEN"
    token = os.getenv(token_name)
    if not token:
        raise ConfigurationError(f"Environment variable `{token_name}` is not set.")

    summary = _load_training_summary(target_dir)
    _ensure_model_card(target_dir, repo_id=repo_id, artifact_type=artifact_type, summary=summary)

    api = importlib.import_module("huggingface_hub").HfApi(token=token)
    repo_url = str(api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token))
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=target_dir,
        commit_message=f"Upload {artifact_type} artifacts from DeepSeek Coder Studio",
        token=token,
    )
    LOGGER.info("Uploaded %s to %s", target_dir, repo_id)

    return {
        "repo_id": repo_id,
        "repo_url": repo_url,
        "artifact_type": artifact_type,
        "source_dir": str(target_dir),
        "commit_url": commit_info.commit_url,
        "commit_oid": commit_info.oid,
    }


def _load_training_summary(source_dir: Path) -> dict[str, Any] | None:
    summary_path = source_dir / "training_summary.json"
    if not summary_path.exists():
        return None
    payload = read_json(summary_path)
    return payload if isinstance(payload, dict) else None


def _ensure_model_card(
    source_dir: Path,
    repo_id: str,
    artifact_type: str,
    summary: dict[str, Any] | None,
) -> None:
    model_card_path = source_dir / "README.md"
    if model_card_path.exists():
        return
    write_text(
        model_card_path,
        build_model_card(repo_id=repo_id, artifact_type=artifact_type, summary=summary),
    )


def build_model_card(
    repo_id: str,
    artifact_type: str,
    summary: dict[str, Any] | None = None,
) -> str:
    """Build a minimal model card for Hub uploads."""
    tags = ["transformers", "llmstudio", "code-generation"]
    strategy = None
    base_model = None
    train_samples = None
    validation_samples = None
    training_loss = None

    if summary:
        strategy = summary.get("strategy")
        base_model = summary.get("base_model")
        train_samples = summary.get("train_samples")
        validation_samples = summary.get("validation_samples")
        training_loss = summary.get("training_loss")

    if artifact_type == "adapter":
        tags.extend(["peft", "lora"])
    if strategy == "qlora":
        tags.append("qlora")
    if strategy and strategy not in tags:
        tags.append(str(strategy))

    front_matter = [
        "---",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
    ]
    front_matter.extend(f"- {tag}" for tag in dict.fromkeys(tags))
    if base_model:
        front_matter.append(f"base_model: {base_model}")
    front_matter.append("---")

    lines = [
        *front_matter,
        "",
        f"# {repo_id}",
        "",
        "Artifacts exported with DeepSeek Coder Studio.",
        "",
        "## Contents",
        "",
        f"- Artifact type: `{artifact_type}`",
    ]

    if base_model:
        lines.append(f"- Base model: `{base_model}`")
    if strategy:
        lines.append(f"- Training strategy: `{strategy}`")
    if train_samples is not None:
        lines.append(f"- Train samples: `{train_samples}`")
    if validation_samples is not None:
        lines.append(f"- Validation samples: `{validation_samples}`")
    if training_loss is not None:
        lines.append(f"- Final training loss: `{training_loss}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This repository was prepared for sharing from the desktop GUI workflow.",
            "Review the generated files before publishing the model publicly.",
        ]
    )
    return "\n".join(lines) + "\n"
