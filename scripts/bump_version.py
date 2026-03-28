from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump semantic version across the project.")
    parser.add_argument("part", choices=["major", "minor", "patch"], nargs="?", default="patch")
    parser.add_argument("--set-version", dest="set_version", help="Set an explicit version, e.g. 0.2.0")
    args = parser.parse_args()

    current_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    new_version = args.set_version or increment_version(current_version, args.part)

    (ROOT / "VERSION").write_text(new_version, encoding="utf-8")
    update_python_version(new_version)
    update_pyproject(new_version)
    update_extension_package(new_version)
    update_app_config(new_version)
    update_package_lock(new_version)
    update_readme(new_version)
    update_tests(new_version)
    update_training_config_names(new_version)
    ensure_changelog(new_version)

    print(f"Updated project version to V{new_version}")


def increment_version(version: str, part: str) -> str:
    major, minor, patch = [int(piece) for piece in version.split(".")]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def update_python_version(version: str) -> None:
    version_file = ROOT / "app" / "core" / "version.py"
    content = version_file.read_text(encoding="utf-8")
    content = re.sub(r'VERSION = "[^"]+"', f'VERSION = "{version}"', content)
    version_file.write_text(content, encoding="utf-8")


def update_pyproject(version: str) -> None:
    pyproject = ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    content = re.sub(r'version = "[^"]+"', f'version = "{version}"', content, count=1)
    pyproject.write_text(content, encoding="utf-8")


def update_extension_package(version: str) -> None:
    package_json = ROOT / "vscode-extension" / "package.json"
    payload = json.loads(package_json.read_text(encoding="utf-8"))
    payload["version"] = version
    package_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_package_lock(version: str) -> None:
    package_lock = ROOT / "vscode-extension" / "package-lock.json"
    if not package_lock.exists():
        return
    payload = json.loads(package_lock.read_text(encoding="utf-8"))
    payload["version"] = version
    if "packages" in payload and "" in payload["packages"]:
        payload["packages"][""]["version"] = version
    package_lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_app_config(version: str) -> None:
    config_path = ROOT / "configs" / "app.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["app"]["version"] = version
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def update_readme(version: str) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    content = re.sub(r"Version:\s*`V?\d+\.\d+\.\d+`", f"Version: `V{version}`", content)
    README_PATH.write_text(content, encoding="utf-8")


def update_tests(version: str) -> None:
    api_test = ROOT / "tests" / "test_api.py"
    content = api_test.read_text(encoding="utf-8")
    content = re.sub(r'payload\["version"\] == "\d+\.\d+\.\d+"', f'payload["version"] == "{version}"', content)
    api_test.write_text(content, encoding="utf-8")


def update_training_config_names(version: str) -> None:
    dashed = version.replace(".", "-")
    for path in (ROOT / "configs" / "training").glob("*.yaml"):
        content = path.read_text(encoding="utf-8")
        content = re.sub(r'job_name:\s*"([^"]+)-v\d+-\d+-\d+"', f'job_name: "\\1-v{dashed}"', content)
        path.write_text(content, encoding="utf-8")


def ensure_changelog(version: str) -> None:
    if CHANGELOG_PATH.exists():
        return
    CHANGELOG_PATH.write_text(
        "# Changelog\n\n"
        f"## V{version}\n\n"
        "- Initial public release of DeepSeek Coder Studio.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
