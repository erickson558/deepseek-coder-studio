# Changelog

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

