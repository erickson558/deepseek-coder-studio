from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.metrics import compute_metrics
from app.evaluation.reporter import write_reports
from app.inference.prompts import build_task_prompt
from app.models.api import GenerationParameters, TaskRequest
from app.models.evaluation import BenchmarkCase, EvaluationConfig
from app.services.assistant import AssistantService
from app.utils.files import read_json

LOGGER = get_logger(__name__)


class EvaluationRunner:
    def __init__(self) -> None:
        self.service = AssistantService(get_settings())

    def run(self, config_path: str | Path) -> dict[str, Any]:
        config = self._load_config(config_path)
        cases = self._load_cases(config.benchmark_file)
        results: list[dict[str, Any]] = []

        for case in cases:
            LOGGER.info("Evaluating case %s", case.id)
            request = TaskRequest(
                prompt=case.prompt,
                task_context=case.context,
                language=case.language,
                parameters=GenerationParameters(
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                    response_format="text",
                ),
            )
            prompt = build_task_prompt(case.task, request)
            response = self.service.run_task(case.task, prompt, request.parameters, request.model)
            metrics = compute_metrics(case, response.output_text)
            results.append(
                {
                    "id": case.id,
                    "task": case.task.value,
                    "prompt": case.prompt,
                    "generated_text": response.output_text,
                    "metrics": metrics,
                }
            )

        passed_cases = sum(1 for item in results if item["metrics"]["passed"])
        average_score = round(sum(item["metrics"]["score"] for item in results) / max(len(results), 1), 4)
        summary = {
            "model_id": config.model_id or get_settings().default_model_id,
            "total_cases": len(results),
            "passed_cases": passed_cases,
            "average_score": average_score,
        }
        report_paths = write_reports(config.output_dir, summary, results)
        return {"summary": summary, "results": results, "report_paths": report_paths}

    def _load_config(self, config_path: str | Path) -> EvaluationConfig:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return EvaluationConfig.model_validate(payload)

    def _load_cases(self, benchmark_path: str | Path) -> list[BenchmarkCase]:
        payload = read_json(benchmark_path)
        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]
        return [BenchmarkCase.model_validate(item) for item in payload]
