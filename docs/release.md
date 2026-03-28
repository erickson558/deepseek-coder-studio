# Release and Versioning

## Version source of truth

The project version is stored in:

- `VERSION`
- `app/core/version.py`
- `pyproject.toml`
- `configs/app.yaml`
- `vscode-extension/package.json`

Use the bump script:

```bash
python scripts/bump_version.py patch
python scripts/bump_version.py minor
python scripts/bump_version.py major
python scripts/bump_version.py --set-version 1.0.0
```

The bump script also updates the visible README version, the API version test, and versioned training job names so releases stay aligned.

## GitHub release workflow

Every push to `main` triggers `.github/workflows/release.yml`.

The workflow:

1. Runs tests.
2. Builds the Python package.
3. Builds the VS Code extension `.vsix`.
4. Builds the Windows `.exe`.
5. Creates a GitHub release using the current semantic version.

GitHub tags and releases use the visible format `Vx.x.x`, while package metadata keeps the numeric semver form `x.x.x` required by Python and npm tooling.

Manual tags are also supported. If you create and push `Vx.x.x` yourself, the workflow reuses that tag instead of failing.
