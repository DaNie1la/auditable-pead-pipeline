"""Publish machine-readable aggregate audit evidence without private paths or rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPORT_NAMES = (
    "phase1_data_quality.json",
    "phase2_transcript_quality.json",
    "phase3_finbert_quality.json",
    "phase4_market_quality.json",
    "phase4_fundamental_quality.json",
    "phase5_modeling_quality.json",
)
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|<LOCAL_PATH>)")


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if not key.lower().endswith("path") and key != "source_values_corrected"
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if PRIVATE_PATH.search(value):
            raise ValueError("private path found in public audit content")
        return value.replace("point-in-time shares", "shares filtered by historical record date")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report-dir", type=Path, required=True)
    parser.add_argument("--phase5-report", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "public_audit_summary.json",
    )
    args = parser.parse_args()
    reports = {}
    for name in REPORT_NAMES:
        path = args.phase5_report if name.startswith("phase5_") else args.source_report_dir / name
        reports[name.removesuffix(".json")] = sanitize(
            json.loads(path.read_text(encoding="utf-8"))
        )
    output = {
        "schema_version": 1,
        "scope": "Aggregate audit only; no event-level records or source transcripts",
        "data_vintage_limitation": (
            "Yahoo histories were downloaded after the studied events. Date filtering "
            "does not establish what an immutable vendor snapshot showed at event time."
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"public_audit_complete phases={len(reports)} output={args.output}")


if __name__ == "__main__":
    main()
