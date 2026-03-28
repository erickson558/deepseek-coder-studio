from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.tasks import router as tasks_router


def build_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health_router)
    router.include_router(models_router)
    router.include_router(tasks_router)
    return router
