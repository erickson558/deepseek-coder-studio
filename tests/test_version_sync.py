import json
from pathlib import Path

from app.core.version import VERSION


def test_version_matches_project_files() -> None:
    version_file = Path("VERSION").read_text(encoding="utf-8").strip()
    package_json = json.loads(Path("vscode-extension/package.json").read_text(encoding="utf-8"))

    assert VERSION == version_file
    assert package_json["version"] == VERSION
