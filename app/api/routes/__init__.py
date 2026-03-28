from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.tasks import router as tasks_router

__all__ = ["health_router", "models_router", "tasks_router"]
