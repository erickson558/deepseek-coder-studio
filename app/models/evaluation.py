from pathlib import Path

from pydantic import BaseModel, Field

from app.models.task import TaskType


class BenchmarkCase(BaseModel):
    id: str
    task: TaskType
    prompt: str
    language: str | None = None
    context: str | None = None
    expected_substrings: list[str] = Field(default_factory=list)
    forbidden_substrings: list[str] = Field(default_factory=list)
    reference_answer: str | None = None


class EvaluationConfig(BaseModel):
    benchmark_file: Path
    output_dir: Path = Path("outputs/eval")
    model_id: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 512
