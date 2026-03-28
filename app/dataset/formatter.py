from typing import Any
from uuid import uuid4

from app.models.dataset import DatasetExample, Message
from app.models.task import TaskType


DEFAULT_SYSTEM_PROMPT = (
    "You are a senior coding assistant. Produce robust, concise, production-grade help."
)


def normalise_record(raw: dict[str, Any]) -> DatasetExample:
    if "messages" in raw:
        messages = [Message.model_validate(message) for message in raw["messages"]]
        task = _coerce_task(raw.get("task"))
        return DatasetExample(
            id=str(raw.get("id") or uuid4()),
            task=task,
            language=raw.get("language"),
            messages=messages,
            metadata=raw.get("metadata", {}),
        )

    system_prompt = raw.get("system") or DEFAULT_SYSTEM_PROMPT
    user_content = _build_user_message(raw)
    assistant_content = _pick_value(raw, "response", "assistant", "output", "target")
    if not assistant_content:
        raise ValueError("missing assistant/response/output field")

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
        Message(role="assistant", content=str(assistant_content)),
    ]
    metadata = {
        "source": raw.get("source"),
        "tags": raw.get("tags", []),
    }
    return DatasetExample(
        id=str(raw.get("id") or uuid4()),
        task=_coerce_task(raw.get("task")),
        language=raw.get("language"),
        messages=messages,
        metadata={key: value for key, value in metadata.items() if value},
    )


def _build_user_message(raw: dict[str, Any]) -> str:
    instruction = _pick_value(raw, "instruction", "prompt", "question", "user")
    if not instruction:
        raise ValueError("missing instruction/prompt/question/user field")

    sections = [str(instruction).strip()]
    for label, key in (
        ("Input", "input"),
        ("Context", "context"),
        ("Code", "code"),
        ("File Content", "file_content"),
    ):
        value = raw.get(key)
        if value:
            sections.append(f"{label}:\n{value}")
    return "\n\n".join(sections)


def _pick_value(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return None


def _coerce_task(value: str | None) -> TaskType:
    if not value:
        return TaskType.CODE_GENERATION
    normalised = value.strip().lower()
    aliases = {
        "generation": TaskType.CODE_GENERATION,
        "generate": TaskType.CODE_GENERATION,
        "bugfix": TaskType.BUG_FIXING,
        "bug_fix": TaskType.BUG_FIXING,
        "fix": TaskType.BUG_FIXING,
        "refactoring": TaskType.REFACTOR,
        "tests": TaskType.TEST_GENERATION,
        "test_generation": TaskType.TEST_GENERATION,
        "explain": TaskType.CODE_EXPLANATION,
        "explanation": TaskType.CODE_EXPLANATION,
        "file_edit": TaskType.FILE_EDITING,
    }
    if normalised in aliases:
        return aliases[normalised]
    try:
        return TaskType(normalised)
    except ValueError:
        return TaskType.CODE_GENERATION
