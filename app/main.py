"""FastAPI application factory and lifecycle hooks."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import build_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.version import VERSION

configure_logging()
LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log process startup and shutdown without loading heavy services eagerly."""
    settings = get_settings()
    LOGGER.info("Starting %s v%s in %s", settings.app_name, VERSION, settings.env)
    yield
    LOGGER.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Build the FastAPI app with CORS and route registration."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_router())
    return app


app = create_app()
