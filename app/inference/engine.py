import time
from typing import Any

from app.core.config import AppSettings
from app.core.exceptions import DependencyUnavailableError
from app.core.logging import get_logger
from app.models.api import ChatRequest, GenerateRequest, GenerationParameters, InferenceResponse
from app.models.task import TaskType
from app.training.formatting import render_messages
from app.utils.serialization import try_parse_json_block

LOGGER = get_logger(__name__)


class InferenceEngine:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._active_model_id = settings.default_model_id

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": self.settings.default_model_id,
                "base_model": self.settings.base_model_name,
                "source": self.settings.model_source,
                "adapter_path": str(self.settings.adapter_dir),
                "merged_path": str(self.settings.merged_model_dir),
            }
        ]

    def generate(self, request: GenerateRequest) -> InferenceResponse:
        prompt = request.prompt if not request.context else f"{request.prompt}\n\n{request.context}"
        return self._run_prompt(
            prompt=prompt,
            task=TaskType.CODE_GENERATION,
            parameters=request.parameters,
            requested_model=request.model,
        )

    def chat(self, request: ChatRequest) -> InferenceResponse:
        self._ensure_loaded(request.model)
        prompt = render_messages(request.messages, tokenizer=self._tokenizer)
        return self._run_prompt(
            prompt=prompt,
            task=TaskType.CHAT,
            parameters=request.parameters,
            requested_model=request.model,
        )

    def run_task(
        self,
        task: TaskType,
        prompt: str,
        parameters: GenerationParameters,
        requested_model: str | None = None,
    ) -> InferenceResponse:
        return self._run_prompt(
            prompt=prompt,
            task=task,
            parameters=parameters,
            requested_model=requested_model,
        )

    def _run_prompt(
        self,
        prompt: str,
        task: TaskType,
        parameters: GenerationParameters,
        requested_model: str | None = None,
    ) -> InferenceResponse:
        self._ensure_loaded(requested_model)
        start = time.perf_counter()

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device is not None:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": parameters.max_new_tokens,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "do_sample": parameters.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        outputs = self._model.generate(**inputs, **generation_kwargs)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        output_json = try_parse_json_block(output_text) if parameters.response_format == "json" else None
        return InferenceResponse(
            task=task,
            model_id=self._active_model_id,
            output_text=output_text,
            latency_ms=latency_ms,
            output_json=output_json,
        )

    def _ensure_loaded(self, requested_model: str | None) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._load_model(requested_model)

    def _load_model(self, requested_model: str | None) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Inference requires torch, transformers and peft to be installed."
            ) from exc

        if requested_model and requested_model != self.settings.default_model_id:
            LOGGER.warning(
                "Requested model %s is not registered, using default %s",
                requested_model,
                self.settings.default_model_id,
            )

        model_path = self.settings.base_model_name
        source = self.settings.model_source.lower()
        if source == "merged" and self.settings.merged_model_dir.exists():
            model_path = str(self.settings.merged_model_dir)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if source == "adapter" and self.settings.adapter_dir.exists():
            model = PeftModel.from_pretrained(model, str(self.settings.adapter_dir))

        if device_map is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(self._device)
        self._model = model.eval()
