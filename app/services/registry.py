from app.core.config import AppSettings
from app.models.api import ModelInfo


class ModelRegistry:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name=self.settings.app_name,
                model_id=self.settings.default_model_id,
                source=self.settings.model_source,
                base_model=self.settings.base_model_name,
                adapter_path=str(self.settings.adapter_dir),
                merged_path=str(self.settings.merged_model_dir),
                active=True,
            )
        ]
