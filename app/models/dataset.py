from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.task import TaskType


ALLOWED_ROLES = {"system", "user", "assistant"}


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in ALLOWED_ROLES:
            raise ValueError(f"unsupported role: {value}")
        return normalised

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message content cannot be empty")
        return stripped


class DatasetExample(BaseModel):
    id: str
    task: TaskType = TaskType.CODE_GENERATION
    language: str | None = None
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[Message]) -> list[Message]:
        if len(value) < 2:
            raise ValueError("dataset example needs at least two messages")
        if value[-1].role != "assistant":
            raise ValueError("last message must be assistant")
        if not any(message.role == "user" for message in value):
            raise ValueError("dataset example requires at least one user message")
        return value
