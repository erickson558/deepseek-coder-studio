"""Tareas en background y control del servidor API usados por la GUI de escritorio.

Arquitectura de concurrencia
────────────────────────────
La GUI de Tk es single-threaded. Para no bloquear el event loop con operaciones
lentas (training, evaluación, carga de modelos, llamadas HTTP), este módulo
proporciona dos herramientas:

1. BackgroundTaskRunner
   - Ejecuta callables en hilos daemon.
   - Deposita resultados en una Queue[TaskResult].
   - La GUI consume esa Queue cada 200 ms via _poll_background_results().

2. ApiServerController
   - Levanta (o reutiliza) el servidor FastAPI dentro de un hilo daemon.
   - Expone métodos start() / stop() / health() seguros para llamar desde workers.

Funciones de operación
──────────────────────
Cada run_*() es un wrapper delgado que adapta los parámetros de la GUI a la
API interna del proyecto. Se ejecutan SIEMPRE dentro de BackgroundTaskRunner,
nunca en el hilo principal de Tk.
"""

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
from app.training.config import load_training_config, save_training_config
from app.training.hub import publish_training_artifacts
from app.training.trainer import FineTuneRunner

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TaskResult:
    """Sobre de resultado que cruza del hilo worker al hilo principal de Tk.

    Atributos:
        name:          Identificador de la tarea (ej: "prepare_dataset", "train_model").
        success:       True si la operación terminó sin excepción.
        payload:       Datos devueltos por la operación (dict, objeto, None).
        error_message: Mensaje de error en formato str si success=False.
    """

    name: str
    success: bool
    payload: Any = None
    error_message: str | None = None


class BackgroundTaskRunner:
    """Ejecuta operaciones largas en hilos daemon sin bloquear el event loop de Tk.

    Patrón de uso:
        runner = BackgroundTaskRunner()
        runner.submit("nombre", funcion, arg1, arg2, kwarg=valor)
        # La GUI consume runner.results (Queue) cada 200 ms.
    """

    def __init__(self) -> None:
        # Queue thread-safe que accumula resultados para que el hilo Tk los consuma.
        self.results: Queue[TaskResult] = Queue()
        # Contador de tareas activas protegido por lock para evitar race conditions.
        self._active_tasks = 0
        self._lock = threading.Lock()

    def submit(self, name: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Lanzar un callable en un hilo daemon y encolar su resultado al terminar.

        Args:
            name:      Nombre identificador de la tarea (usado por la GUI para dispatch).
            operation: Función a ejecutar en background.
            *args:     Argumentos posicionales para operation.
            **kwargs:  Argumentos nombrados para operation.
        """

        def worker() -> None:
            # Incrementar contador de tareas activas (protegido con lock).
            with self._lock:
                self._active_tasks += 1
            try:
                LOGGER.info("Starting background task: %s", name)
                # Ejecutar la operación y capturar el resultado.
                payload = operation(*args, **kwargs)
                # Éxito: encolar resultado positivo.
                self.results.put(TaskResult(name=name, success=True, payload=payload))
            except Exception as exc:  # noqa: BLE001
                # Cualquier excepción se convierte en TaskResult de fallo.
                LOGGER.exception("Background task failed: %s", name)
                self.results.put(TaskResult(name=name, success=False, error_message=str(exc)))
            finally:
                # Decrementar siempre el contador, incluso si hubo excepción.
                with self._lock:
                    self._active_tasks -= 1

        # daemon=True garantiza que el hilo no impide el cierre de la app.
        threading.Thread(target=worker, daemon=True, name=f"gui-task-{name}").start()

    def has_active_tasks(self) -> bool:
        """Devolver True si algún hilo worker sigue corriendo."""
        with self._lock:
            return self._active_tasks > 0


class ApiServerController:
    """Levantar y detener el backend FastAPI silenciosamente dentro de un hilo daemon.

    Diseñado para ser llamado SIEMPRE desde un worker de BackgroundTaskRunner,
    nunca directamente desde el hilo principal de Tk, ya que start() y
    _wait_until_ready() hacen llamadas HTTP bloqueantes.

    Estados posibles:
        - "external": Había una API en el puerto antes de que la GUI intentara arrancarlo.
        - "managed":  La GUI lanzó el servidor; ella es responsable de detenerlo.
        - "stopped":  El servidor fue detenido o nunca arrancó.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None   # hilo que corre uvicorn
        self._server: uvicorn.Server | None = None     # instancia del servidor
        self._managed = False                          # True si la GUI lo inició
        self._startup_exception: Exception | None = None  # captura errores del hilo
        self._host = "127.0.0.1"
        self._port = 8000

    def start(self, host: str, port: int) -> dict[str, Any]:
        """Iniciar el servidor o reutilizar uno ya corriendo en el puerto dado.

        Lógica:
          1. Si hay una API externa respondiendo: marcar como "external" (no gestionada).
          2. Si el hilo managed ya está corriendo: devolver URL directamente.
          3. Si no hay nada: lanzar uvicorn en un hilo daemon y esperar a que responda.
        """
        self._host = host
        self._port = port

        # Verificar si ya existe una API respondiendo en ese host:puerto.
        if self.health(host, port)["reachable"]:
            self._managed = False
            return {"status": "external", "url": self._base_url(host, port)}

        # Si el hilo managed ya está corriendo, no lanzar uno nuevo.
        if self.is_running():
            return {"status": "managed", "url": self._base_url(host, port)}

        # Lanzar un nuevo servidor gestionado por la GUI.
        self._startup_exception = None
        self._managed = True

        def runner() -> None:
            try:
                # Desactivar logs de consola en el hilo del servidor para no
                # mezclar output con la GUI que ya captura logs en archivo.
                configure_logging(include_console=False)
                from app.main import create_app

                config = uvicorn.Config(
                    app=create_app(),
                    host=host,
                    port=port,
                    log_level=get_settings().log_level.lower(),
                    access_log=False,
                    log_config=None,  # usa nuestro logging ya configurado
                )
                self._server = uvicorn.Server(config)
                self._server.run()   # bloqueante hasta que should_exit=True
            except Exception as exc:  # noqa: BLE001
                # Capturar para que _wait_until_ready() pueda re-lanzar la causa.
                self._startup_exception = exc
                LOGGER.exception("API server thread failed to start")

        self._thread = threading.Thread(target=runner, daemon=True, name="gui-api-server")
        self._thread.start()
        # Esperar hasta que /health responda antes de devolver el resultado.
        self._wait_until_ready(host, port)
        return {"status": "managed", "url": self._base_url(host, port)}

    def stop(self) -> dict[str, Any]:
        """Detener el servidor gestionado y esperar a que el hilo termine.

        Si la API era externa (no manejada por la GUI), devuelve sin hacer nada.
        Lanza RuntimeError si el hilo no termina en 10 segundos.
        """
        if not self._managed:
            # No es nuestra responsabilidad: no tocar la API externa.
            return {"status": "external", "url": self._base_url(self._host, self._port)}

        if self._server is None or self._thread is None:
            return {"status": "stopped"}

        # Señalizar a uvicorn que debe detenerse limpiamente.
        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("The API server did not stop within the expected time.")

        # Limpiar referencias para que is_running() devuelva False.
        self._server = None
        self._thread = None
        self._managed = False
        return {"status": "stopped"}

    def health(self, host: str | None = None, port: int | None = None) -> dict[str, Any]:
        """Verificar si el endpoint /health responde con status 200.

        Returns:
            dict con claves "reachable" (bool), "url" (str), y opcionalmente "payload".
        """
        base_url = self._base_url(host or self._host, port or self._port)
        try:
            response = httpx.get(f"{base_url}/health", timeout=3)
            response.raise_for_status()
            payload = response.json()
            return {"reachable": True, "payload": payload, "url": base_url}
        except Exception:  # noqa: BLE001
            # Cualquier error de conexión o HTTP se trata como "no disponible".
            return {"reachable": False, "url": base_url}

    def is_running(self) -> bool:
        """Devolver True si el hilo managed del servidor sigue vivo."""
        return bool(self._thread and self._thread.is_alive())

    def _wait_until_ready(self, host: str, port: int, timeout_seconds: int = 20) -> None:
        """Polling hasta que /health responda o expire el timeout.

        Si el hilo del servidor lanzó una excepción durante el startup, la
        re-lanza aquí para que BackgroundTaskRunner la capture y genere un
        TaskResult de fallo con mensaje descriptivo.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            # Propagar excepciones del hilo del servidor al worker que llamó start().
            if self._startup_exception is not None:
                raise self._startup_exception
            if self.health(host, port)["reachable"]:
                return
            time.sleep(0.25)   # polling cada 250 ms
        raise RuntimeError("The backend API did not become ready before timeout.")

    @staticmethod
    def _base_url(host: str, port: int) -> str:
        """Construir la URL base del servidor a partir de host y puerto."""
        return f"http://{host}:{port}"


# ── Funciones de operación (wrappers delgados para BackgroundTaskRunner) ────────


def run_prepare_dataset(input_path: str, output_dir: str) -> dict[str, Any]:
    """Preparar el dataset (Paso 1) usando el mismo pipeline del CLI.

    Convierte archivos de entrada (JSONL/JSON/CSV/directorio) al formato instruct
    y genera splits train/validation/test en output_dir.
    """
    return prepare_dataset(input_path=input_path, output_dir=output_dir)


def run_training(config_path: str) -> dict[str, Any]:
    """Lanzar el entrenamiento LoRA/QLoRA (Paso 2) con el runner compartido del CLI.

    Lee la config YAML en config_path y ejecuta FineTuneRunner.run().
    """
    return FineTuneRunner().run(config_path)


def _resolve_dataset_input_path(dataset_input_path: str | Path) -> Path:
    """Resolve the dataset input path and fall back to the bundled sample when needed."""
    candidate = Path(dataset_input_path)
    if candidate.exists():
        return candidate

    fallback = Path("data/samples/sample_dataset.jsonl")
    if fallback.exists():
        LOGGER.warning("Dataset input %s was not found; falling back to %s", candidate, fallback)
        return fallback

    raise FileNotFoundError(f"Dataset input not found: {candidate}")


def run_auto_training(dataset_input_path: str, dataset_output_dir: str, config_path: str) -> dict[str, Any]:
    """Prepare missing dataset splits and then run training in one background task.

    The training config remains the source of truth for the train/validation file
    locations, so the auto-flow can fill in missing files without changing the
    existing manual training workflow.
    """
    config = load_training_config(config_path)
    dataset_input = _resolve_dataset_input_path(dataset_input_path)
    target_train_file = Path(config.train_file)
    target_validation_file = Path(config.validation_file)

    dataset_summary: dict[str, Any] | None = None
    if not target_train_file.exists() or not target_validation_file.exists():
        # Bootstrap the dataset only when the configured train/validation files are missing.
        dataset_output = Path(dataset_output_dir) if dataset_output_dir.strip() else target_train_file.parent
        dataset_summary = prepare_dataset(input_path=dataset_input, output_dir=dataset_output)
        target_train_file = dataset_output / "train.jsonl"
        target_validation_file = dataset_output / "validation.jsonl"
        config = config.model_copy(update={"train_file": target_train_file, "validation_file": target_validation_file})
        save_training_config(config_path, config)

    training_summary = FineTuneRunner().run_job(config)
    return {
        "dataset_summary": dataset_summary,
        "training_summary": training_summary,
        "train_file": str(target_train_file),
        "validation_file": str(target_validation_file),
    }


def run_evaluation(config_path: str) -> dict[str, Any]:
    """Ejecutar el benchmark de evaluación (Paso 3) con el runner compartido.

    Lee la config YAML de evaluación y ejecuta EvaluationRunner.run().
    """
    return EvaluationRunner().run(config_path)


def run_publish_model(
    repo_id: str,
    source_dir: str,
    token_env_var: str = "HF_TOKEN",
    private: bool = False,
    artifact_type: str = "adapter",
) -> dict[str, Any]:
    """Publicar artefactos de entrenamiento en Hugging Face Hub (Paso 4).

    Args:
        repo_id:       ID del repositorio HF (ej: "usuario/nombre-repo").
        source_dir:    Ruta local con el adapter o modelo fusionado.
        token_env_var: Nombre de la variable de entorno que contiene el HF token.
        private:       True para crear/actualizar un repositorio privado.
        artifact_type: "adapter" (LoRA weights) o "merged" (modelo completo).
    """
    return publish_training_artifacts(
        repo_id=repo_id,
        source_dir=source_dir,
        token_env_var=token_env_var,
        private=private,
        artifact_type=artifact_type,
    )


def run_inference(
    task_name: str,
    prompt: str,
    language: str | None = None,
    context_file: str | None = None,
) -> dict[str, Any]:
    """Ejecutar inferencia local sin necesidad de arrancar la API FastAPI.

    Carga el AssistantService con lazy loading del modelo. La primera llamada
    puede tardar varios segundos mientras se carga el modelo en memoria.

    Args:
        task_name:    Valor del enum TaskType (ej: "code_generation", "chat").
        prompt:       Texto del prompt del usuario. No puede estar vacío.
        language:     Lenguaje de programación (relevante para tareas de código).
        context_file: Ruta a un archivo de texto usado como contexto adicional.

    Returns:
        dict con claves "output_text" y "latency_ms" (compatible con InferenceResponse).
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    settings = get_settings()
    service = AssistantService(settings)

    # Leer el archivo de contexto si fue especificado.
    context = None
    if context_file:
        context = Path(context_file).read_text(encoding="utf-8")

    # Construir parámetros de generación desde la configuración de la app.
    parameters = GenerationParameters(
        temperature=settings.default_temperature,
        max_new_tokens=settings.default_max_new_tokens,
        response_format="text",
    )

    # Resolver el tipo de tarea. ValueError si task_name no es un valor válido de TaskType.
    task = TaskType(task_name)

    if task == TaskType.CHAT:
        # Modo chat: envolver el prompt en un mensaje de usuario.
        response = service.chat(
            ChatRequest(
                messages=[Message(role="user", content=prompt)],
                parameters=parameters,
            )
        )
    elif task == TaskType.CODE_GENERATION:
        # Generación de código: construir un prompt enriquecido con instrucción de tarea.
        request = GenerateRequest(
            prompt=prompt,
            context=context,
            language=language,
            parameters=parameters,
        )
        response = service.generate(request.model_copy(update={"prompt": build_generation_prompt(request)}))
    else:
        # Otras tareas (bug fixing, refactor, tests, explain, edit):
        # usar el template de prompt específico de la tarea.
        request = TaskRequest(
            prompt=prompt,
            task_context=context,
            language=language,
            parameters=parameters,
        )
        response = service.run_task(task, build_task_prompt(task, request), parameters)

    # Serializar a dict para que BackgroundTaskRunner pueda encolarlo sin problemas.
    return response.model_dump(mode="json")
