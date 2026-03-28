"""Background tasks and API server control used by the desktop GUI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Callable

import httpx
import uvicorn

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.dataset.pipeline import prepare_dataset
from app.evaluation.benchmark import EvaluationRunner
from app.inference.prompts import build_generation_prompt, build_task_prompt
from app.models.api import ChatRequest, GenerateRequest, GenerationParameters, TaskRequest
from app.models.dataset import Message
from app.models.task import TaskType
from app.services.assistant import AssistantService
from app.training.trainer import FineTuneRunner

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TaskResult:
    """Cross-thread result envelope returned to the GUI event loop."""

    name: str
    success: bool
    payload: Any = None
    error_message: str | None = None


class BackgroundTaskRunner:
    """Execute long-running jobs without blocking the Tk event loop."""

    def __init__(self) -> None:
        self.results: Queue[TaskResult] = Queue()
        self._active_tasks = 0
        self._lock = threading.Lock()

    def submit(self, name: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run a callable in a daemon thread and enqueue its result."""

        def worker() -> None:
            with self._lock:
                self._active_tasks += 1
            try:
                LOGGER.info("Starting background task: %s", name)
                payload = operation(*args, **kwargs)
                self.results.put(TaskResult(name=name, success=True, payload=payload))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Background task failed: %s", name)
                self.results.put(TaskResult(name=name, success=False, error_message=str(exc)))
            finally:
                with self._lock:
                    self._active_tasks -= 1

        threading.Thread(target=worker, daemon=True, name=f"gui-task-{name}").start()

    def has_active_tasks(self) -> bool:
        """Return True when any background operation is still running."""
        with self._lock:
            return self._active_tasks > 0


class ApiServerController:
    """Start and stop the FastAPI backend silently inside a background thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._managed = False
        self._startup_exception: Exception | None = None
        self._host = "127.0.0.1"
        self._port = 8000

    def start(self, host: str, port: int) -> dict[str, Any]:
        """Start the backend server or reuse an already running endpoint."""
        self._host = host
        self._port = port

        if self.health(host, port)["reachable"]:
            self._managed = False
            return {"status": "external", "url": self._base_url(host, port)}

        if self.is_running():
            return {"status": "managed", "url": self._base_url(host, port)}

        self._startup_exception = None
        self._managed = True

        def runner() -> None:
            try:
                # File logging remains active while the API is hosted by the GUI.
                configure_logging(include_console=False)
                from app.main import create_app

                config = uvicorn.Config(
                    app=create_app(),
                    host=host,
                    port=port,
                    log_level=get_settings().log_level.lower(),
                    access_log=False,
                    log_config=None,
                )
                self._server = uvicorn.Server(config)
                self._server.run()
            except Exception as exc:  # noqa: BLE001
                self._startup_exception = exc
                LOGGER.exception("API server thread failed to start")

        self._thread = threading.Thread(target=runner, daemon=True, name="gui-api-server")
        self._thread.start()
        self._wait_until_ready(host, port)
        return {"status": "managed", "url": self._base_url(host, port)}

    def stop(self) -> dict[str, Any]:
        """Stop the managed backend server when it belongs to the GUI."""
        if not self._managed:
            return {"status": "external", "url": self._base_url(self._host, self._port)}

        if self._server is None or self._thread is None:
            return {"status": "stopped"}

        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("The API server did not stop within the expected time.")

        self._server = None
        self._thread = None
        self._managed = False
        return {"status": "stopped"}

    def health(self, host: str | None = None, port: int | None = None) -> dict[str, Any]:
        """Check whether the configured API responds to /health."""
        base_url = self._base_url(host or self._host, port or self._port)
        try:
            response = httpx.get(f"{base_url}/health", timeout=3)
            response.raise_for_status()
            payload = response.json()
            return {"reachable": True, "payload": payload, "url": base_url}
        except Exception:  # noqa: BLE001
            return {"reachable": False, "url": base_url}

    def is_running(self) -> bool:
        """Return True when the GUI-managed server thread is still alive."""
        return bool(self._thread and self._thread.is_alive())

    def _wait_until_ready(self, host: str, port: int, timeout_seconds: int = 20) -> None:
        """Wait until the health endpoint answers or fail with the startup cause."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._startup_exception is not None:
                raise self._startup_exception
            if self.health(host, port)["reachable"]:
                return
            time.sleep(0.25)
        raise RuntimeError("The backend API did not become ready before timeout.")

    @staticmethod
    def _base_url(host: str, port: int) -> str:
        return f"http://{host}:{port}"


def run_prepare_dataset(input_path: str, output_dir: str) -> dict[str, Any]:
    """Prepare dataset files using the same pipeline exposed by the CLI."""
    return prepare_dataset(input_path=input_path, output_dir=output_dir)


def run_training(config_path: str) -> dict[str, Any]:
    """Execute a training job through the shared training runner."""
    return FineTuneRunner().run(config_path)


def run_evaluation(config_path: str) -> dict[str, Any]:
    """Execute evaluation through the shared benchmark runner."""
    return EvaluationRunner().run(config_path)


def run_inference(
    task_name: str,
    prompt: str,
    language: str | None = None,
    context_file: str | None = None,
) -> dict[str, Any]:
    """Run local inference without requiring the API to be started first."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    settings = get_settings()
    service = AssistantService(settings)
    context = None
    if context_file:
        context = Path(context_file).read_text(encoding="utf-8")

    parameters = GenerationParameters(
        temperature=settings.default_temperature,
        max_new_tokens=settings.default_max_new_tokens,
        response_format="text",
    )
    task = TaskType(task_name)

    if task == TaskType.CHAT:
        response = service.chat(
            ChatRequest(
                messages=[Message(role="user", content=prompt)],
                parameters=parameters,
            )
        )
    elif task == TaskType.CODE_GENERATION:
        request = GenerateRequest(
            prompt=prompt,
            context=context,
            language=language,
            parameters=parameters,
        )
        response = service.generate(request.model_copy(update={"prompt": build_generation_prompt(request)}))
    else:
        request = TaskRequest(
            prompt=prompt,
            task_context=context,
            language=language,
            parameters=parameters,
        )
        response = service.run_task(task, build_task_prompt(task, request), parameters)

    return response.model_dump(mode="json")
