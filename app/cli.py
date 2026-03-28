"""Typer-based command line entry point for project operations."""

from pathlib import Path

import typer
import uvicorn
from rich.console import Console

from app.core.config import get_settings
from app.core.version import VERSION
from app.dataset.pipeline import prepare_dataset
from app.evaluation.benchmark import EvaluationRunner
from app.inference.prompts import build_generation_prompt, build_task_prompt
from app.models.api import ChatRequest, GenerateRequest, GenerationParameters, TaskRequest
from app.models.dataset import Message
from app.models.task import TaskType
from app.services.assistant import AssistantService
from app.training.trainer import FineTuneRunner

cli = typer.Typer(no_args_is_help=True, help="DeepSeek Coder Studio CLI")
console = Console()


@cli.command("version")
def version() -> None:
    """Print the current application version."""
    console.print(VERSION)


@cli.command("prepare-dataset")
def prepare_dataset_command(
    input_path: str = typer.Option(..., help="Input dataset source"),
    output_dir: str = typer.Option("data/processed", help="Output directory"),
    train_ratio: float = typer.Option(0.8, help="Train split ratio"),
    validation_ratio: float = typer.Option(0.1, help="Validation split ratio"),
    seed: int = typer.Option(42, help="Shuffle seed"),
) -> None:
    """Prepare a raw dataset for training."""
    summary = prepare_dataset(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    console.print_json(data=summary)


@cli.command("train")
def train_command(config: str = typer.Option(..., help="Training config YAML path")) -> None:
    """Run a fine-tuning job from a YAML config."""
    summary = FineTuneRunner().run(config)
    console.print_json(data=summary)


@cli.command("evaluate")
def evaluate_command(config: str = typer.Option(..., help="Evaluation config YAML path")) -> None:
    """Execute the configured benchmark suite."""
    report = EvaluationRunner().run(config)
    console.print_json(data=report["summary"])


@cli.command("infer")
def infer_command(
    task: TaskType = typer.Option(TaskType.CODE_GENERATION, help="Task type"),
    prompt: str = typer.Option(..., help="Prompt or instruction"),
    language: str | None = typer.Option(None, help="Language name"),
    context_file: Path | None = typer.Option(None, help="Optional context file"),
    response_format: str = typer.Option("text", help="text or json"),
    max_new_tokens: int = typer.Option(512, help="Generation length"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
) -> None:
    """Run a single inference task from the terminal."""
    service = AssistantService(get_settings())
    context = context_file.read_text(encoding="utf-8") if context_file else None
    parameters = GenerationParameters(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        response_format=response_format,
    )
    if task == TaskType.CHAT:
        request = ChatRequest(
            messages=[Message(role="user", content=prompt)],
            parameters=parameters,
        )
        response = service.chat(request)
    elif task == TaskType.CODE_GENERATION:
        request = GenerateRequest(
            prompt=prompt,
            context=context,
            language=language,
            parameters=parameters,
        )
        request = request.model_copy(update={"prompt": build_generation_prompt(request)})
        response = service.generate(request)
    else:
        request = TaskRequest(
            prompt=prompt,
            task_context=context,
            language=language,
            parameters=parameters,
        )
        response = service.run_task(task, build_task_prompt(task, request), parameters)
    console.print(response.output_text)


@cli.command("serve")
def serve_command(
    host: str = typer.Option(None, help="Host override"),
    port: int = typer.Option(None, help="Port override"),
) -> None:
    """Serve the FastAPI backend."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )


@cli.command("chat")
def chat_command(
    message: str = typer.Option(..., help="User message"),
    max_new_tokens: int = typer.Option(512, help="Generation length"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
) -> None:
    """Run a simple chat exchange from the terminal."""
    service = AssistantService(get_settings())
    request = ChatRequest(
        messages=[Message(role="user", content=message)],
        parameters=GenerationParameters(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ),
    )
    response = service.chat(request)
    console.print(response.output_text)


def run() -> None:
    """Entrypoint used by both `python -m app.cli` and the console script."""
    cli()
