import re
from collections import Counter

from app.models.evaluation import BenchmarkCase


def compute_metrics(case: BenchmarkCase, generated_text: str) -> dict[str, float | bool]:
    expected_hits = _ratio_hits(case.expected_substrings, generated_text)
    forbidden_hits = _ratio_hits(case.forbidden_substrings, generated_text)
    overlap = _token_overlap(case.reference_answer or "", generated_text)
    absent_ok = forbidden_hits == 0.0
    score = round((expected_hits + overlap + (1.0 if absent_ok else 0.0)) / 3, 4)
    return {
        "expected_hit_ratio": expected_hits,
        "forbidden_hit_ratio": forbidden_hits,
        "reference_overlap": overlap,
        "passed": expected_hits >= 0.5 and absent_ok,
        "score": score,
    }


def _ratio_hits(needles: list[str], haystack: str) -> float:
    if not needles:
        return 1.0
    lowered = haystack.lower()
    hits = sum(1 for needle in needles if needle.lower() in lowered)
    return round(hits / len(needles), 4)


def _token_overlap(reference: str, generated: str) -> float:
    if not reference.strip():
        return 1.0
    reference_tokens = Counter(_tokenise(reference))
    generated_tokens = Counter(_tokenise(generated))
    intersection = sum((reference_tokens & generated_tokens).values())
    total = max(sum(reference_tokens.values()), 1)
    return round(intersection / total, 4)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
