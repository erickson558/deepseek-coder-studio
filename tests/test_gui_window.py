from app.gui.window import parse_positive_int


def test_parse_positive_int_returns_fallback_for_invalid_values() -> None:
    assert parse_positive_int("", 60) == 60
    assert parse_positive_int("abc", 60) == 60
    assert parse_positive_int("0", 60) == 1
    assert parse_positive_int("-9", 60) == 1


def test_parse_positive_int_uses_numeric_value_when_valid() -> None:
    assert parse_positive_int("45", 60) == 45
