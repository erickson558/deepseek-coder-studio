"""Configuración persistente de la GUI de escritorio.

El archivo config.json se guarda junto al ejecutable/script para que
cada usuario o instalación tenga sus propias preferencias independientes.
Se usa get_config_file() de app.core.runtime para resolver la ruta correcta
tanto cuando la app corre como script Python como cuando es un .exe compilado.

Flujo de persistencia:
  1. Al arrancar la GUI: GuiConfigStore.load() → GuiConfig con valores guardados.
  2. Al cambiar cualquier campo: _save_config() → GuiConfigStore.save() → config.json.
  3. Si config.json está corrupto o ausente: GuiConfig() con defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.logging import get_logger
from app.core.runtime import get_config_file

LOGGER = get_logger(__name__)


class GuiConfig(BaseModel):
    """Preferencias de la GUI serializables a JSON.

    Todos los campos tienen defaults razonables para que la primera ejecución
    funcione sin que el usuario configure nada. Los campos se guardan
    en config.json y se restauran al volver a abrir la app.

    Los campos están agrupados según los 4 pasos del flujo de trabajo:
      - Preferencias generales (idioma, ventana, API)
      - Paso 1: Preparar Dataset
      - Paso 2: Entrenamiento
      - Paso 3: Evaluación
      - Paso 4: Publicar en HF
      - Asistente (inferencia local)
    """

    # ── Preferencias generales ─────────────────────────────────────────────
    window_geometry: str = "1280x860+80+60"  # posición y tamaño de la ventana
    language: str = "es"                      # código de idioma ("es" o "en")
    auto_start_backend: bool = False           # arrancar API al abrir la GUI
    auto_close_enabled: bool = False           # cerrar por inactividad
    auto_close_seconds: int = 60              # segundos de inactividad para cerrar
    host: str = "127.0.0.1"                   # host del backend FastAPI
    port: int = 8000                          # puerto del backend FastAPI

    # ── Paso 1: Preparar Dataset ───────────────────────────────────────────
    dataset_input_path: str = "data/samples/sample_dataset.jsonl"
    dataset_output_path: str = "data/processed"

    # ── Paso 2: Entrenamiento ──────────────────────────────────────────────
    training_config_path: str = "configs/training/lora.yaml"
    training_job_name: str = "deepseek-coder-lora-v0-2-0"
    training_strategy: str = "lora"           # "lora" o "qlora"
    training_base_model: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    training_train_file: str = "data/processed/train.jsonl"
    training_validation_file: str = "data/processed/validation.jsonl"
    training_num_train_epochs: int = 1
    training_per_device_train_batch_size: int = 1
    training_gradient_accumulation_steps: int = 8
    training_learning_rate: float = 2e-4
    training_max_seq_length: int = 2048
    training_merge_adapter: bool = True       # fusionar adapter con modelo base al terminar

    # ── Paso 3: Evaluación ─────────────────────────────────────────────────
    evaluation_config_path: str = "configs/evaluation/default.yaml"

    # ── Paso 4: Publicar en Hugging Face ───────────────────────────────────
    publish_repo_id: str = ""                 # "usuario/nombre-repo"
    publish_token_env_var: str = "HF_TOKEN"   # variable de entorno con el token HF
    publish_private_repo: bool = False
    publish_artifact_type: str = "adapter"    # "adapter" o "merged"
    publish_source_dir: str = ""              # carpeta con los artefactos a subir
    # Rutas del último entrenamiento; usadas para auto-rellenar publish_source_dir.
    last_adapter_output_dir: str = ""
    last_merged_output_dir: str = ""

    # ── Asistente / inferencia local ───────────────────────────────────────
    inference_task: str = "code_generation"   # valor del enum TaskType
    inference_language: str = "python"
    inference_prompt: str = ""
    inference_context_file: str = ""

    # ── Estado de la UI ────────────────────────────────────────────────────
    selected_tab: str = "dashboard"           # pestaña activa al cerrar


class GuiConfigStore:
    """Carga y guarda la configuración de escritorio de forma automática.

    Uso típico:
        store = GuiConfigStore()
        config = store.load()          # obtener config (o defaults si no existe)
        config.language = "en"
        store.save(config)             # persistir cambios a disco
    """

    def __init__(self, path: Path | None = None):
        # Usar ruta personalizada (útil en tests) o la ruta del runtime.
        self.path = path or get_config_file()

    def load(self) -> GuiConfig:
        """Cargar la config guardada y caer a defaults si no existe o está corrupta.

        Maneja silenciosamente errores de:
          - Archivo inexistente (primera ejecución).
          - JSON malformado (archivo truncado o editado a mano con errores).
          - Validación de Pydantic (campo con tipo incorrecto).
        """
        if not self.path.exists():
            # Primera ejecución: usar defaults.
            return GuiConfig()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return GuiConfig.model_validate(payload)
        except (OSError, ValidationError, json.JSONDecodeError):
            LOGGER.exception("Failed to load GUI config from %s", self.path)
            # Archivo corrupto: caer silenciosamente a defaults.
            return GuiConfig()

    def save(self, config: GuiConfig) -> None:
        """Persistir el estado completo de la GUI a config.json.

        Crea los directorios padres si no existen (útil para ejecuciones desde
        directorios temporales del .exe compilado).
        """
        # Asegurar que el directorio padre exista antes de escribir.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False preserva caracteres latinos en el archivo JSON.
        self.path.write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
