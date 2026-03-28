import json
import re
from typing import Any


def try_parse_json_block(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    for candidate in (stripped, _extract_code_fence_json(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_code_fence_json(text: str) -> str | None:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None
