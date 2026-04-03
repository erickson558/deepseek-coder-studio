from pathlib import Path

import pytest

from app.gui.window import _open_path_windows, parse_positive_int


def test_parse_positive_int_returns_fallback_for_invalid_values() -> None:
    assert parse_positive_int("", 60) == 60
    assert parse_positive_int("abc", 60) == 60
    assert parse_positive_int("0", 60) == 1
    assert parse_positive_int("-9", 60) == 1


def test_parse_positive_int_uses_numeric_value_when_valid() -> None:
    assert parse_positive_int("45", 60) == 45


def test_open_path_windows_falls_back_to_notepad_for_text_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    started: list[list[str]] = []

    def fake_startfile(_: str) -> None:
        raise OSError("No file association")

    def fake_popen(command: list[str]) -> object:
        started.append(command)
        return object()

    monkeypatch.setattr("app.gui.window.os.startfile", fake_startfile, raising=False)
    monkeypatch.setattr("app.gui.window.subprocess.Popen", fake_popen)

    _open_path_windows(config_path)

    assert started == [["notepad.exe", str(config_path)]]
