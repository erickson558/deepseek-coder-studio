import json
from pathlib import Path

import yaml

from app.core.version import VERSION


def test_version_matches_project_files() -> None:
    version_file = Path("VERSION").read_text(encoding="utf-8").strip()
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    app_config = yaml.safe_load(Path("configs/app.yaml").read_text(encoding="utf-8"))
    package_json = json.loads(Path("vscode-extension/package.json").read_text(encoding="utf-8"))

    assert VERSION == version_file
    assert package_json["version"] == VERSION
    assert f'version = "{VERSION}"' in pyproject
    assert f"Version: `V{VERSION}`" in readme
    assert app_config["app"]["version"] == VERSION
