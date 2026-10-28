import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pead_pipeline.transcripts import (  # noqa: E402
    build_transcript_audit,
    split_transcript,
    write_section_corpus,
)


def raw_row(**overrides):
    values = {
        "date": "Jan 28, 2020, 5:00 p.m. ET",
        "exchange": "NASDAQ: AAPL",
        "q": "2020-Q1",
        "ticker": "AAPL",
        "transcript": (
            "Prepared Remarks: Operator Welcome to the quarterly earnings call. "
            "Tim Cook: Thank you for joining us today. We will discuss revenue, "
            "margins, products, operations, guidance, and our outlook for next year. "
            "Questions & Answers: Operator We will now take the first question. "
            "Analyst: Could you explain the revenue outlook and margin assumptions "
            "for the coming quarter? Tim Cook: Certainly, demand remains resilient "
            "and we expect operating leverage to improve gradually through the year."
            "\nDuration: 45 minutes\nCall participants:\nTim Cook"
        ),
    }
    values.update(overrides)
    return values


class Phase2Tests(unittest.TestCase):
    def test_explicit_markers_produce_two_clean_sections(self):
        text = raw_row()["transcript"]
        split = split_transcript(text)
        self.assertEqual(split.status, "split_complete")
        self.assertEqual(
            text[split.prepared_start : split.prepared_end].strip(),
            "Operator Welcome to the quarterly earnings call. Tim Cook: Thank you "
            "for joining us today. We will discuss revenue, margins, products, "
            "operations, guidance, and our outlook for next year.",
        )
        self.assertEqual(
            text[split.qa_start : split.qa_end].strip(),
            "Operator We will now take the first question. Analyst: Could you explain "
            "the revenue outlook and margin assumptions for the coming quarter? "
            "Tim Cook: Certainly, demand remains resilient and we expect operating "
            "leverage to improve gradually through the year.",
        )

    def test_missing_qa_marker_is_not_inferred(self):
        split = split_transcript(
            "Prepared Remarks: Management spoke. Operator invited questions."
        )
        self.assertEqual(split.status, "qa_marker_missing")
        self.assertIsNone(split.qa_start)

    def test_operator_signoff_is_not_model_ready_qa(self):
        text = (
            "Prepared Remarks: " + "management discussion " * 30
            + "Questions & Answers: Operator [Operator signoff]"
            + "\nDuration: 30 minutes"
        )
        self.assertEqual(split_transcript(text).status, "qa_text_too_short")

    def test_exact_duplicate_collapses_but_conflict_does_not(self):
        same = pd.DataFrame([raw_row(), raw_row()])
        audited, safe, review = build_transcript_audit(same)
        self.assertEqual(set(audited.event_status), {"exact_duplicate"})
        self.assertEqual(len(safe), 1)
        self.assertEqual(len(review), 1)

        conflict = pd.DataFrame(
            [raw_row(), raw_row(transcript=raw_row()["transcript"] + " Changed")]
        )
        audited, safe, review = build_transcript_audit(conflict)
        self.assertEqual(set(audited.event_status), {"conflict"})
        self.assertEqual(len(safe), 0)
        self.assertEqual(len(review), 2)

    def test_writer_emits_one_json_record_per_safe_event(self):
        raw = pd.DataFrame([raw_row()])
        _, safe, _ = build_transcript_audit(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sections.jsonl.gz"
            self.assertEqual(write_section_corpus(raw, safe, path), 1)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                record = json.loads(handle.readline())
                self.assertEqual(record["ticker"], "AAPL")
                self.assertIn("quarterly earnings call", record["prepared_text"])
                self.assertIn("first question", record["qa_text"])
                self.assertEqual(handle.readline(), "")

    def test_writer_is_byte_deterministic(self):
        raw = pd.DataFrame([raw_row()])
        _, safe, _ = build_transcript_audit(raw)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl.gz"
            second = Path(directory) / "second.jsonl.gz"
            write_section_corpus(raw, safe, first)
            write_section_corpus(raw, safe, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
