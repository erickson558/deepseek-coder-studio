"""Application-wide settings loaded from the local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """Typed settings object shared by CLI, API and inference services."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="LLM_",
        extra="ignore",
    )

    app_name: str = "DeepSeek Coder Studio"
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    default_model_id: str = "deepseek-coder-v2-lite-instruct"
    base_model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    model_source: str = "adapter"
    adapter_path: str = "outputs/adapters/latest"
    merged_model_path: str = "outputs/merged/latest"
    device: str = "auto"

    default_temperature: float = 0.2
    default_max_new_tokens: int = 512
    request_timeout_seconds: int = 120

    api_key: str | None = None
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    @property
    def adapter_dir(self) -> Path:
        """Return the adapter directory as an absolute path."""
        return PROJECT_ROOT / self.adapter_path

    @property
    def merged_model_dir(self) -> Path:
        """Return the merged-model directory as an absolute path."""
        return PROJECT_ROOT / self.merged_model_path


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cache settings so the app reads the environment only once per process."""
    return AppSettings()
