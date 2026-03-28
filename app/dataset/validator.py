from dataclasses import dataclass, field

from app.models.dataset import DatasetExample


@dataclass(slots=True)
class ValidationIssue:
    example_id: str
    reason: str


@dataclass(slots=True)
class ValidationSummary:
    valid_examples: list[DatasetExample] = field(default_factory=list)
    invalid_examples: list[ValidationIssue] = field(default_factory=list)

    @property
    def total_valid(self) -> int:
        return len(self.valid_examples)

    @property
    def total_invalid(self) -> int:
        return len(self.invalid_examples)


def validate_examples(examples: list[DatasetExample]) -> ValidationSummary:
    summary = ValidationSummary()
    for example in examples:
        issues = _collect_issues(example)
        if issues:
            summary.invalid_examples.extend(
                ValidationIssue(example_id=example.id, reason=reason) for reason in issues
            )
        else:
            summary.valid_examples.append(example)
    return summary


def _collect_issues(example: DatasetExample) -> list[str]:
    issues: list[str] = []
    if len(example.messages) < 2:
        issues.append("requires at least two messages")
    if example.messages[-1].role != "assistant":
        issues.append("assistant must be the final message")
    if not any(message.role == "user" for message in example.messages):
        issues.append("requires at least one user message")
    return issues
