from fastapi import APIRouter, Depends

from app.api.dependencies import get_assistant_service, verify_api_key
from app.models.api import ModelInfo
from app.services.assistant import AssistantService

router = APIRouter(tags=["models"], dependencies=[Depends(verify_api_key)])


@router.get("/models", response_model=list[ModelInfo])
def list_models(service: AssistantService = Depends(get_assistant_service)) -> list[ModelInfo]:
    return service.models()
