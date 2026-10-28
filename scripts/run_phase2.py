from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.phase2_reporting import (  # noqa: E402
    build_phase2_report,
    sha256_file,
    write_phase2_reports,
)
from pead_pipeline.transcripts import (  # noqa: E402
    build_transcript_audit,
    serialize_audit_timestamps,
    write_section_corpus,
)


EXPECTED_RAW_SHA256 = (
    "f84a4b8073799d9b67417008bbe4001466bd006756850f1815469502bb0638d4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 audited Prepared Remarks/Q&A corpus."
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
    if not args.raw_transcripts.exists():
        raise FileNotFoundError(args.raw_transcripts)

    raw_hash = sha256_file(args.raw_transcripts)
    if raw_hash.lower() != EXPECTED_RAW_SHA256:
        raise ValueError(
            "Raw transcript checksum does not match the audited source file. "
            f"Expected {EXPECTED_RAW_SHA256}, received {raw_hash}."
        )

    raw = pd.read_pickle(args.raw_transcripts)
    audited, safe, review = build_transcript_audit(raw)

    interim = args.output_root / "data" / "interim"
    processed = args.output_root / "data" / "processed"
    reports = args.output_root / "reports"
    for directory in [interim, processed, reports]:
        directory.mkdir(parents=True, exist_ok=True)

    serialize_audit_timestamps(audited).to_csv(
        interim / "transcripts_audited.csv", index=False
    )
    serialize_audit_timestamps(review).to_csv(
        processed / "transcripts_manual_review.csv", index=False
    )
    corpus_path = processed / "transcript_sections.jsonl.gz"
    written = write_section_corpus(raw, safe, corpus_path)
    if written != len(safe):
        raise RuntimeError(f"Expected {len(safe)} output rows, wrote {written}.")

    report = build_phase2_report(
        args.raw_transcripts, raw_hash, audited, safe, review, corpus_path
    )
    write_phase2_reports(
        report,
        reports / "phase2_transcript_quality.json",
        reports / "phase2_transcript_quality.md",
    )

    print(f"raw_rows={len(raw)}")
    print(f"split_complete_rows={(audited['split_status'] == 'split_complete').sum()}")
    print(f"safe_section_events={len(safe)}")
    print(f"manual_review_rows={len(review)}")
    print(f"corpus={corpus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
