from functools import lru_cache

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.services.assistant import AssistantService


@lru_cache(maxsize=1)
def get_assistant_service() -> AssistantService:
    return AssistantService(get_settings())


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
