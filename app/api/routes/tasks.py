from fastapi import APIRouter, Depends

from app.api.dependencies import get_assistant_service, verify_api_key
from app.inference.prompts import build_generation_prompt, build_task_prompt
from app.models.api import ChatRequest, GenerateRequest, InferenceResponse, TaskRequest
from app.models.task import TaskType
from app.services.assistant import AssistantService

router = APIRouter(tags=["assistant"], dependencies=[Depends(verify_api_key)])


@router.post("/generate", response_model=InferenceResponse)
def generate(
    request: GenerateRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    prompt = build_generation_prompt(request)
    wrapped = request.model_copy(update={"prompt": prompt})
    return service.generate(wrapped)


@router.post("/chat", response_model=InferenceResponse)
def chat(
    request: ChatRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return service.chat(request)


@router.post("/explain", response_model=InferenceResponse)
def explain(
    request: TaskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return _execute_task(TaskType.CODE_EXPLANATION, request, service)


@router.post("/fix", response_model=InferenceResponse)
def fix(
    request: TaskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return _execute_task(TaskType.BUG_FIXING, request, service)


@router.post("/refactor", response_model=InferenceResponse)
def refactor(
    request: TaskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return _execute_task(TaskType.REFACTOR, request, service)


@router.post("/tests", response_model=InferenceResponse)
def generate_tests(
    request: TaskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return _execute_task(TaskType.TEST_GENERATION, request, service)


@router.post("/edit", response_model=InferenceResponse)
def edit_file(
    request: TaskRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> InferenceResponse:
    return _execute_task(TaskType.FILE_EDITING, request, service)


def _execute_task(
    task: TaskType,
    request: TaskRequest,
    service: AssistantService,
) -> InferenceResponse:
    prompt = build_task_prompt(task, request)
    return service.run_task(task, prompt, request.parameters, request.model)
