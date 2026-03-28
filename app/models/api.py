from typing import Any

from pydantic import BaseModel, Field

from app.models.dataset import Message
from app.models.task import TaskType


class GenerationParameters(BaseModel):
    temperature: float = 0.2
    max_new_tokens: int = 512
    top_p: float = 0.95
    do_sample: bool = True
    response_format: str = "text"


class GenerateRequest(BaseModel):
    prompt: str
    context: str | None = None
    language: str | None = None
    model: str | None = None
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class TaskRequest(BaseModel):
    prompt: str | None = None
    selection: str | None = None
    language: str | None = None
    file_path: str | None = None
    file_content: str | None = None
    task_context: str | None = None
    model: str | None = None
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class InferenceResponse(BaseModel):
    task: TaskType | str
    model_id: str
    output_text: str
    latency_ms: float
    output_json: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    name: str
    model_id: str
    source: str
    base_model: str
    adapter_path: str | None = None
    merged_path: str | None = None
    active: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
