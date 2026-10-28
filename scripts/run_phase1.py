from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.events import (  # noqa: E402
    audit_master,
    audit_nlp_scores,
    inspect_raw_transcripts,
    serialize_timestamps,
)
from pead_pipeline.reporting import (  # noqa: E402
    build_quality_report,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 PEAD event audit tables.")
    parser.add_argument(
        "--master",
        type=Path,
        default=PROJECT_ROOT / "private" / "legacy" / "master_dataset_multi_window.csv",
    )
    parser.add_argument(
        "--nlp-scores",
        type=Path,
        default=PROJECT_ROOT / "private" / "legacy" / "thesis_nlp_scores_new.csv",
    )
    parser.add_argument(
        "--raw-transcripts",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "motley-fool-data.pkl",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.master, args.nlp_scores]:
        if not path.exists():
            raise FileNotFoundError(path)

    interim = args.output_root / "data" / "interim"
    processed = args.output_root / "data" / "processed"
    reports = args.output_root / "reports"
    for directory in [interim, processed, reports]:
        directory.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master)
    nlp_scores = pd.read_csv(args.nlp_scores)
    nlp_audited = audit_nlp_scores(nlp_scores)
    master_audited, safe, conflicts = audit_master(
        master, reference_timestamps=nlp_audited
    )
    raw_status = inspect_raw_transcripts(args.raw_transcripts)

    serialize_timestamps(master_audited).to_csv(
        interim / "master_events_audited.csv", index=False
    )
    serialize_timestamps(nlp_audited).to_csv(
        interim / "nlp_scores_audited.csv", index=False
    )
    serialize_timestamps(safe).to_csv(
        processed / "events_unique_safe.csv", index=False
    )
    serialize_timestamps(conflicts).to_csv(
        processed / "events_conflicts.csv", index=False
    )
    unresolved = master_audited[
        master_audited["event_status"] == "unresolved_timestamp"
    ]
    serialize_timestamps(unresolved).to_csv(
        processed / "events_unresolved_timestamps.csv", index=False
    )

    report = build_quality_report(
        args.master,
        args.nlp_scores,
        master_audited,
        safe,
        conflicts,
        nlp_audited,
        raw_status,
    )
    write_json_report(report, reports / "phase1_data_quality.json")
    write_markdown_report(report, reports / "phase1_data_quality.md")

    print(f"safe_events={len(safe)}")
    print(f"conflict_groups={conflicts['event_id'].nunique()}")
    print(f"conflict_rows={len(conflicts)}")
    print(f"unresolved_timestamp_rows={len(unresolved)}")
    print(f"report={reports / 'phase1_data_quality.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

