import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pead_pipeline.events import (  # noqa: E402
    MASTER_REQUIRED_COLUMNS,
    audit_master,
    build_event_id,
    classify_market_session,
    normalize_ticker,
    parse_earnings_timestamp,
)


def row(**overrides):
    values = {
        "ticker": "AAPL",
        "date": "Jan 28, 2020, 5:00 p.m. ET",
        "Tone_Pres": 0.1,
        "Tone_QA": 0.2,
        "CAR_5": 0.01,
        "CAR_10": 0.02,
        "CAR_60": 0.03,
        "SUE_Proxy": 0.04,
        "Size": 20.0,
        "VOL": 0.02,
        "Turnover": 12.0,
    }
    values.update(overrides)
    return values


class Phase1Tests(unittest.TestCase):
    def test_date_parser_handles_source_and_master_formats(self):
        values = pd.Series(
            ["Jan 28, 2020, 5:00 p.m. ET", "2020/1/28 17:00"]
        )
        parsed = parse_earnings_timestamp(values)
        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.iloc[0], parsed.iloc[1])
        self.assertEqual(classify_market_session(parsed.iloc[0]), "post_market")

    def test_date_parser_handles_legacy_title_date_list(self):
        values = pd.Series(
            [
                "['Brunswick (BC 0.66%) Q4 2018 ', "
                "'Jan. 31, 2019 11:00 a.m. ET']"
            ]
        )
        parsed = parse_earnings_timestamp(values)
        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.iloc[0].isoformat(), "2019-01-31T11:00:00-05:00")

    def test_date_parser_handles_actual_legacy_list_object(self):
        values = pd.Series(
            [["Brunswick (BC) Q4 2018", "Jan. 31, 2019 11:00 a.m. ET"]]
        )
        parsed = parse_earnings_timestamp(values)
        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.iloc[0].isoformat(), "2019-01-31T11:00:00-05:00")

    def test_date_parser_extracts_timestamp_attached_to_legacy_title(self):
        values = pd.Series(
            [
                "['Moog Inc. Q2 2021 Earnings CallApril 30, 2021, "
                "10:00 a.m. ET']",
                "['TTEC Q4 2017 ', ' TranscriptMarch 13, 2018, 8:30 a.m. ET']",
            ]
        )
        parsed = parse_earnings_timestamp(values)
        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.iloc[0].isoformat(), "2021-04-30T10:00:00-04:00")
        self.assertEqual(parsed.iloc[1].isoformat(), "2018-03-13T08:30:00-04:00")

    def test_event_id_is_stable(self):
        timestamp = parse_earnings_timestamp(
            pd.Series(["Jan 28, 2020, 5:00 p.m. ET"])
        ).iloc[0]
        self.assertEqual(
            build_event_id("aapl", timestamp, 1),
            build_event_id(" AAPL ", timestamp, 999),
        )

    def test_ticker_normalization_removes_unmatched_trailing_parenthesis(self):
        self.assertEqual(normalize_ticker(" CIB) "), "CIB")
        self.assertEqual(normalize_ticker("BRK.B"), "BRK.B")

    def test_conflicting_duplicate_is_not_in_safe_output(self):
        frame = pd.DataFrame([row(), row(CAR_5=0.5)])
        audited, safe, conflicts = audit_master(frame)
        self.assertEqual(set(audited["event_status"]), {"conflict"})
        self.assertEqual(len(safe), 0)
        self.assertEqual(len(conflicts), 2)

    def test_exact_duplicate_collapses_to_one_safe_event(self):
        frame = pd.DataFrame([row(), row()])
        audited, safe, conflicts = audit_master(frame)
        self.assertEqual(set(audited["event_status"]), {"exact_duplicate"})
        self.assertEqual(len(safe), 1)
        self.assertEqual(len(conflicts), 0)

    def test_midnight_placeholder_recovers_only_from_unique_reference(self):
        frame = pd.DataFrame([row(date="2019/1/31 0:00")])
        reference = pd.DataFrame(
            {
                "ticker_normalized": ["AAPL"],
                "event_timestamp_et": parse_earnings_timestamp(
                    pd.Series(["Jan. 31, 2019 11:00 a.m. ET"])
                ),
            }
        )
        audited, safe, conflicts = audit_master(frame, reference)
        self.assertEqual(
            audited.loc[0, "timestamp_resolution"], "recovered_from_nlp_scores"
        )
        self.assertEqual(
            audited.loc[0, "event_timestamp_et"].isoformat(),
            "2019-01-31T11:00:00-05:00",
        )
        self.assertEqual(len(safe), 1)
        self.assertEqual(len(conflicts), 0)

    def test_midnight_placeholder_without_reference_is_unresolved(self):
        frame = pd.DataFrame([row(date="2019/1/31 0:00")])
        reference = pd.DataFrame(
            {
                "ticker_normalized": ["MSFT"],
                "event_timestamp_et": parse_earnings_timestamp(
                    pd.Series(["Jan. 31, 2019 11:00 a.m. ET"])
                ),
            }
        )
        audited, safe, _ = audit_master(frame, reference)
        self.assertEqual(
            audited.loc[0, "timestamp_resolution"], "unresolved_no_candidate"
        )
        self.assertTrue(pd.isna(audited.loc[0, "event_timestamp_et"]))
        self.assertEqual(audited.loc[0, "market_session"], "unknown")
        self.assertEqual(audited.loc[0, "target_start_rule"], "manual_review")
        self.assertEqual(audited.loc[0, "event_status"], "unresolved_timestamp")
        self.assertEqual(len(safe), 0)

    def test_required_schema_is_complete(self):
        self.assertEqual(set(row()), set(MASTER_REQUIRED_COLUMNS))


if __name__ == "__main__":
    unittest.main()

