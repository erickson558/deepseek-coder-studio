"""Service layer that exposes a stable interface for CLI, API and evaluation."""

from app.core.config import AppSettings
from app.inference.engine import InferenceEngine
from app.models.api import ChatRequest, GenerateRequest, InferenceResponse
from app.models.task import TaskType
from app.services.registry import ModelRegistry


class AssistantService:
    """Facade over the model registry and inference engine."""

    def __init__(self, settings: AppSettings):
        self.registry = ModelRegistry(settings)
        self.engine = InferenceEngine(settings)

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def models(self):
        return self.registry.list_models()

    def generate(self, request: GenerateRequest) -> InferenceResponse:
        return self.engine.generate(request)

    def chat(self, request: ChatRequest) -> InferenceResponse:
        return self.engine.chat(request)

    def run_task(self, task: TaskType, prompt: str, parameters, requested_model: str | None = None) -> InferenceResponse:
        return self.engine.run_task(task, prompt, parameters, requested_model)
