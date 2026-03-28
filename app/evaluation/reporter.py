from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.utils.files import ensure_directory, write_json, write_text


def write_reports(output_dir: str | Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, str]:
    target_dir = ensure_directory(output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    json_path = target_dir / f"evaluation-{timestamp}.json"
    md_path = target_dir / f"evaluation-{timestamp}.md"
    payload = {"summary": summary, "results": results}

    write_json(json_path, payload)
    write_text(md_path, _render_markdown(summary, results))
    return {"json": str(json_path), "markdown": str(md_path)}


def _render_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Total cases: `{summary['total_cases']}`",
        f"- Passed: `{summary['passed_cases']}`",
        f"- Average score: `{summary['average_score']}`",
        "",
        "| Case | Task | Passed | Score |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result['id']}` | `{result['task']}` | `{result['metrics']['passed']}` | `{result['metrics']['score']}` |"
        )
    lines.append("")
    return "\n".join(lines)
