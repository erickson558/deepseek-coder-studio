# Changelog

## V0.3.4

- Added a one-click GUI training flow that auto-generates missing train/validation splits before launching fine-tuning.
- Synchronized version metadata across the Python app, README, packaged extension and training defaults.

## V0.3.0

- Added a guided training workflow in the desktop GUI with editable hyperparameters, setup validation and YAML generation.
- Added direct publishing of adapter or merged model artifacts to Hugging Face from the GUI, including a generated model card.
- Hardened release discipline with synced version metadata, duplicate-tag protection in GitHub Actions and Windows executable builds emitted next to `llmstudio.py`.

## V0.2.0

- Added a desktop GUI with tabs for backend control, dataset preparation, training, evaluation and local inference.
- Introduced autosaved `config.json`, file-based `log.txt` logging and silent background execution for long-running tasks.
- Switched Windows executable packaging to a windowed build without a console and kept the original CLI entrypoint available separately.

## V0.1.1

- Added high-signal module and flow comments across the Python backend and VS Code extension.
- Improved version propagation so the bump script updates README, tests and training job identifiers.
- Standardized GitHub release tags to the visible `Vx.x.x` format.

## V0.1.0

- Bootstrap release with dataset preparation, LoRA/QLoRA training, evaluation, inference API and VS Code extension.

## V0.3.5

- Describe the release changes here.

## V0.3.6

- Describe the release changes here.

