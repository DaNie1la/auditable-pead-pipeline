from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.finbert import (  # noqa: E402
    FinBertWindowScorer,
    ScoreConfig,
)
from pead_pipeline.phase3_reporting import (  # noqa: E402
    sha256_file,
    write_phase3_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Prepared Remarks and Q&A with pinned FinBERT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "transcript_sections.jsonl.gz",
    )
    parser.add_argument(
        "--phase2-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase2_transcript_quality.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "finbert_scores_checkpoint.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "finbert_section_scores.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase3_finbert_quality.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase3_finbert_quality.md",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "huggingface",
    )
    parser.add_argument("--event-batch-size", type=int, default=8)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_checkpoint(path: Path, config_hash: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON at line {line_number}: {exc}"
                ) from exc
            if row.get("config_hash") != config_hash:
                raise ValueError("Checkpoint was created with a different score config")
            rows.append(row)
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("Checkpoint contains duplicate event IDs")
    return rows


def iter_input(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for input_order, line in enumerate(handle):
            record = json.loads(line)
            record["input_order"] = input_order
            yield record


def score_batch(scorer, config: ScoreConfig, records):
    texts: list[str] = []
    for record in records:
        texts.extend([record["prepared_text"], record["qa_text"]])
    section_scores = scorer.score_texts(texts)
    output = []
    for index, record in enumerate(records):
        pres = section_scores[index * 2]
        qa = section_scores[index * 2 + 1]
        output.append(
            {
                "input_order": record["input_order"],
                "event_id": record["event_id"],
                "ticker": record["ticker"],
                "event_timestamp_et": record["event_timestamp_et"],
                "quarter": record["quarter"],
                "transcript_sha256": record["transcript_sha256"],
                "Tone_Pres": pres.tone,
                "Pres_Positive": pres.positive,
                "Pres_Negative": pres.negative,
                "Pres_Neutral": pres.neutral,
                "Pres_Windows": pres.windows,
                "Pres_Tokens": pres.tokens,
                "Tone_QA": qa.tone,
                "QA_Positive": qa.positive,
                "QA_Negative": qa.negative,
                "QA_Neutral": qa.neutral,
                "QA_Windows": qa.windows,
                "QA_Tokens": qa.tokens,
                "config_hash": config.sha256,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if args.event_batch_size <= 0 or args.window_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    phase2 = json.loads(args.phase2_report.read_text(encoding="utf-8"))
    input_hash = sha256_file(args.input)
    if input_hash != phase2["output"]["sha256"]:
        raise ValueError("Input corpus hash does not match the Phase 2 report")

    config = ScoreConfig(
        event_batch_size=args.event_batch_size,
        window_batch_size=args.window_batch_size,
    )
    checkpoint_rows = load_checkpoint(args.checkpoint, config.sha256)
    completed = {row["event_id"] for row in checkpoint_rows}

    scorer = FinBertWindowScorer(
        config,
        args.cache_dir,
        device="cuda",
    )
    import torch
    import transformers

    start_time = time.perf_counter()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    eligible_total = phase2["output"]["rows"]
    target_total = min(eligible_total, args.limit) if args.limit else eligible_total
    seen_input = 0
    newly_scored = 0

    with args.checkpoint.open("a", encoding="utf-8", newline="\n") as checkpoint:
        for record in iter_input(args.input):
            if args.limit is not None and seen_input >= args.limit:
                break
            seen_input += 1
            if record["event_id"] in completed:
                continue
            pending.append(record)
            if len(pending) < args.event_batch_size:
                continue
            rows = score_batch(scorer, config, pending)
            for row in rows:
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            completed.update(row["event_id"] for row in rows)
            checkpoint_rows.extend(rows)
            newly_scored += len(rows)
            pending.clear()
            elapsed = time.perf_counter() - start_time
            if newly_scored % (args.event_batch_size * 10) == 0:
                rate = newly_scored / elapsed
                remaining = max(0, target_total - len(completed))
                print(
                    f"completed={len(completed)}/{target_total} "
                    f"rate={rate:.2f}_events_per_sec eta_min={remaining / rate / 60:.1f}",
                    flush=True,
                )

        if pending:
            rows = score_batch(scorer, config, pending)
            for row in rows:
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            completed.update(row["event_id"] for row in rows)
            checkpoint_rows.extend(rows)
            newly_scored += len(rows)

    relevant = [
        row for row in checkpoint_rows if int(row["input_order"]) < target_total
    ]
    relevant.sort(key=lambda row: int(row["input_order"]))
    if len(relevant) != target_total:
        raise RuntimeError(f"Expected {target_total} scores, found {len(relevant)}")

    scores = pd.DataFrame(relevant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, float_format="%.10f")

    runtime = {
        "device": scorer.device_name,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_seconds": time.perf_counter() - start_time,
        "newly_scored": newly_scored,
        "resumed_rows": target_total - newly_scored,
        "event_batch_size": args.event_batch_size,
        "window_batch_size": args.window_batch_size,
    }
    config_report = asdict(config) | {"config_hash": config.sha256}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    write_phase3_report(
        scores,
        args.input,
        input_hash,
        args.output,
        config_report,
        runtime,
        args.report_json,
        args.report_md,
    )
    print(f"scores={len(scores)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
