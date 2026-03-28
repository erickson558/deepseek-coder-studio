from app.models.api import GenerateRequest, TaskRequest
from app.models.task import TaskType


TASK_DESCRIPTIONS = {
    TaskType.CODE_GENERATION: "Generate production-ready code that satisfies the request.",
    TaskType.BUG_FIXING: "Fix the bug and explain the root cause briefly before the patch.",
    TaskType.REFACTOR: "Refactor the code for readability, maintainability and safety.",
    TaskType.TEST_GENERATION: "Generate meaningful automated tests for the provided code.",
    TaskType.CODE_EXPLANATION: "Explain what the code does, risks, and improvement opportunities.",
    TaskType.FILE_EDITING: "Edit the file content according to the request and return the updated snippet.",
}


def build_generation_prompt(request: GenerateRequest) -> str:
    sections = [f"Task: {request.prompt.strip()}"]
    if request.language:
        sections.append(f"Language: {request.language}")
    if request.context:
        sections.append(f"Context:\n{request.context}")
    sections.append("Return a high-quality programming answer.")
    return "\n\n".join(sections)


def build_task_prompt(task: TaskType, request: TaskRequest) -> str:
    sections = [f"Task Type: {task.value}", TASK_DESCRIPTIONS[task]]
    if request.prompt:
        sections.append(f"Instruction:\n{request.prompt}")
    if request.language:
        sections.append(f"Language: {request.language}")
    if request.file_path:
        sections.append(f"File Path: {request.file_path}")
    if request.task_context:
        sections.append(f"Additional Context:\n{request.task_context}")
    if request.file_content:
        sections.append(f"File Content:\n{request.file_content}")
    if request.selection:
        sections.append(f"Selected Code:\n{request.selection}")
    return "\n\n".join(section for section in sections if section)
