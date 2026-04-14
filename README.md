# DeepSeek Coder Studio

DeepSeek Coder Studio is a production-oriented starter platform for fine-tuning and serving a coding LLM based on `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`. It now includes a desktop GUI for Windows-friendly operation, plus dataset preparation, LoRA/QLoRA training, evaluation, local or remote inference, a FastAPI backend, and a VS Code extension that consumes the backend from inside the editor.

Version: `V0.3.6`

License: Apache License 2.0

## Phase 1: Architecture

### Selected architecture

The solution is split into a Python platform under `app/`, a desktop GUI layer under `app/gui`, and a VS Code client under `vscode-extension/`.

- `app/dataset` prepares and validates training data.
- `app/training` handles LoRA or QLoRA fine-tuning jobs.
- `app/inference` loads the base model plus adapters or a merged model.
- `app/evaluation` runs benchmark prompts and writes JSON/Markdown reports.
- `app/api` exposes the assistant as a FastAPI service.
- `app/services` decouples API and CLI orchestration from runtime model logic.
- `app/gui` provides a non-blocking desktop frontend with silent background execution.
- `vscode-extension` provides editor commands and a side panel over the backend API.

This baseline is prepared to evolve toward RAG, multi-file context, Git-aware workflows, multiple providers and richer editor-assisted file editing.

### Repository structure

```text
.
├── .github/workflows/release.yml
├── .vscode/
├── app/
│   ├── api/
│   ├── core/
│   ├── dataset/
│   ├── evaluation/
│   ├── gui/
│   ├── inference/
│   ├── models/
│   ├── services/
│   ├── training/
│   └── utils/
├── configs/
│   ├── evaluation/
│   └── training/
├── data/
│   ├── processed/
│   ├── raw/
│   └── samples/
├── docs/
├── outputs/
│   ├── adapters/
│   ├── checkpoints/
│   ├── eval/
│   ├── logs/
│   └── merged/
├── scripts/
├── tests/
├── vscode-extension/
├── .env.example
├── LICENSE
├── README.md
├── VERSION
├── llmstudio.py
├── llmstudio_cli.py
├── pyproject.toml
└── requirements.txt
```

### System flow

1. Raw examples are loaded from JSONL, JSON, CSV or directories.
2. The dataset pipeline normalizes them to an instruct/chat format and splits them into train, validation and test sets.
3. The training runner tokenizes the processed dataset and fine-tunes the base model using LoRA or QLoRA.
4. The training output stores adapters and can optionally export a merged model.
5. The inference engine loads the base model plus adapter or the merged model and serves prompts from CLI or API.
6. The evaluation runner executes benchmark prompts and generates JSON and Markdown reports.
7. The desktop GUI orchestrates the backend, dataset jobs, evaluation and local inference without opening a console window.
8. The VS Code extension sends editor selections or prompts to the backend and lets the user insert, replace or inspect the answer.

## Installation

### Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

`pyproject.toml` is the primary dependency source. `requirements.txt` is kept as a convenience mirror for environments that prefer flat requirements files.

### Desktop GUI

```bash
python llmstudio.py
```

The GUI stores `config.json` and `log.txt` next to the `.py` or packaged `.exe`.
All long-running actions execute in background threads so the window stays responsive.

Recommended GUI workflow:

1. Prepare the dataset from the `Operations` tab.
2. Review the guided training form, validate the setup and launch training.
3. Let the GUI generate the YAML it will use for the job.
4. Publish the adapter or merged model to Hugging Face after training finishes.

To publish from the GUI, export a Hugging Face token first:

```bash
set HF_TOKEN=hf_xxx
```

### VS Code extension

```bash
cd vscode-extension
npm install
npm run compile
```

## GUI Features

- Silent desktop frontend without console windows in the packaged `.exe`
- Start/stop/check the FastAPI backend from the GUI
- Prepare datasets, launch training and run evaluation from tabs
- Guided training form with a one-click prepare-and-train button that auto-generates missing train/validation splits and saves the YAML before launching
- Pre-flight validation for dataset paths, dependencies and QLoRA constraints
- Save the current GUI training form as YAML before launching the job
- Publish either the adapter folder or the merged model directly to Hugging Face
- Local inference panel for prompts and task-specific coding requests
- Auto-save of GUI preferences to `config.json`
- File logging to `log.txt`
- Auto-start API on launch
- Auto-close countdown on inactivity
- About dialog with visible version
- Spanish/English interface labels

## CLI

The original CLI remains available for automation and power users:

```bash
python llmstudio_cli.py --help
python -m app.cli --help
```

### Prepare dataset

```bash
python -m app.cli prepare-dataset --input-path data/samples/sample_dataset.jsonl --output-dir data/processed
```

### Train with LoRA

```bash
python -m app.cli train --config configs/training/lora.yaml
```

### Train with QLoRA

```bash
python -m app.cli train --config configs/training/qlora.yaml
```

### Evaluate

```bash
python -m app.cli evaluate --config configs/evaluation/default.yaml
```

### Local inference

```bash
python -m app.cli infer --task code_generation --prompt "Create a Python retry decorator" --language python
```

### Chat

```bash
python -m app.cli chat --message "Explain how to test asynchronous FastAPI routes with pytest."
```

### Start API

```bash
python -m app.cli serve --host 127.0.0.1 --port 8000
```

## FastAPI backend

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Generate code

```bash
curl -X POST http://127.0.0.1:8000/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Build a Python function that validates IPv4 strings\",\"language\":\"python\",\"parameters\":{\"temperature\":0.2,\"max_new_tokens\":256,\"top_p\":0.95,\"do_sample\":true,\"response_format\":\"text\"}}"
```

### Explain code

```bash
curl -X POST http://127.0.0.1:8000/explain ^
  -H "Content-Type: application/json" ^
  -d "{\"selection\":\"def flatten(rows): return [item for row in rows for item in row]\",\"language\":\"python\",\"parameters\":{\"temperature\":0.2,\"max_new_tokens\":256,\"top_p\":0.95,\"do_sample\":true,\"response_format\":\"text\"}}"
```

## VS Code extension usage

1. Start the Python API.
2. Open `vscode-extension` in VS Code and run `npm install` and `npm run compile`.
3. Press `F5` to launch an Extension Host.
4. Configure the backend URL in VS Code settings if needed.
5. Select code and run any command from the command palette:
   - `DeepSeek Coder Studio: Explain Selected Code`
   - `DeepSeek Coder Studio: Fix Selected Code`
   - `DeepSeek Coder Studio: Refactor Selected Code`
   - `DeepSeek Coder Studio: Generate Tests for Selection`
   - `DeepSeek Coder Studio: Ask Coding Assistant`
   - `DeepSeek Coder Studio: Generate Code From Prompt`
   - `DeepSeek Coder Studio: Open Assistant Panel`

## Dataset examples

Sample dataset and benchmark prompts are included in:

- `data/samples/sample_dataset.jsonl`
- `data/samples/eval_prompts.json`

Supported task families:

- code generation
- bug fixing
- refactor
- test generation
- code explanation
- file editing

## Build the executable

The project includes a PowerShell build script that compiles the GUI entrypoint `llmstudio.py` into `llmstudio.exe` in the same folder as the `.py` file, using the `.ico` file found at the repository root and packaging it without an attached console window.

Install the project dependencies first so PyInstaller can bundle the training stack correctly:

```bash
python -m pip install -e .
```

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1
```

## Versioning

Semantic versioning is centralized and must be bumped before every commit pushed to `main`.

```bash
python scripts/bump_version.py patch
```

Current synced locations:

- `VERSION`
- `app/core/version.py`
- `pyproject.toml`
- `configs/app.yaml`
- `vscode-extension/package.json`

Visible Git tags and GitHub releases use the `Vx.x.x` format. Tooling files that require strict semver keep the numeric form `x.x.x`.

Best practice used in this repository:

- Every commit pushed to `main` must carry a new version.
- `VERSION` is the source of truth and is mirrored to the app, packaging metadata, tests and docs.
- The release workflow fails if the target `Vx.x.x` tag already exists, which prevents accidental duplicate releases.

## GitHub best practices included

- Apache License 2.0
- semantic versioning
- release workflow on each push to `main`
- project and extension READMEs
- sample data and benchmark prompts
- test scaffolding with `pytest`
- clean modular architecture ready for future RAG or multi-provider work

## Troubleshooting

- If `/generate` or `/chat` fails immediately, check that the model path or adapter path exists and that `torch`, `transformers` and `peft` are installed.
- On Windows, prefer LoRA for local development; QLoRA usually requires Linux plus compatible CUDA and `bitsandbytes`.
- If GUI publishing fails, verify that `HF_TOKEN` is set and that the target repo id uses the `owner/name` format.
- If the VS Code extension cannot connect, verify `deepseekCoderStudio.backendUrl`, `timeoutMs` and optional `apiKey`.
- If release creation fails in GitHub Actions, confirm that `VERSION` was bumped before pushing to `main`.

## Additional docs

- `docs/architecture.md`
- `docs/api.md`
- `docs/release.md`
