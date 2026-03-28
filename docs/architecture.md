# Architecture

## Core decisions

- Python package under `app/` for backend, CLI, training, inference and evaluation.
- Desktop GUI layer under `app/gui` to expose the same capabilities without a console.
- FastAPI API isolated from the inference engine through `services/`.
- Dataset pipeline separated from training so new sources and transforms can evolve independently.
- Centralized semantic version in `VERSION`, mirrored to Python and the VS Code extension.
- VS Code extension built in TypeScript with a thin HTTP client and an isolated assistant panel.

## Main folders

- `app/api`: FastAPI routers and dependencies.
- `app/core`: settings, logging, versioning and shared exceptions.
- `app/dataset`: data loading, normalization, validation and splitting.
- `app/training`: fine-tuning config, token preparation, LoRA/QLoRA helpers and trainer.
- `app/inference`: prompt construction and runtime model loading.
- `app/evaluation`: benchmark execution, metrics and reports.
- `app/services`: application services between API/CLI and the model runtime.
- `app/gui`: Tk desktop frontend, autosaved config, silent background workers and backend controller.
- `vscode-extension`: extension commands, webview panel and backend client.

## Evolution path

The design leaves room for:

- Multi-model registries and external providers.
- RAG and retrieval services under `app/services`.
- Multi-file context builders in `app/inference`.
- Git-aware editing flows.
- New backends such as vLLM, TGI or OpenAI-compatible gateways.
