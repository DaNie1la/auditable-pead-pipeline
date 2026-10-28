from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.market import (  # noqa: E402
    MarketConfig,
    compute_event_market_features,
    yahoo_ticker_candidates,
)


def bars(dates: pd.DatetimeIndex, daily_return: float, volume: float) -> pd.DataFrame:
    prices = [100.0]
    for _ in dates[1:]:
        prices.append(prices[-1] * (1 + daily_return))
    return pd.DataFrame(
        {"Close": prices, "Adj Close": prices, "Volume": volume}, index=dates
    )


class Phase4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2020-01-01", periods=160)
        self.stock = bars(self.dates, 0.01, 1_000_000)
        self.market = bars(self.dates, 0.005, 10_000_000)
        self.config = MarketConfig()

    def event(self, position: int, hour: int) -> dict[str, object]:
        stamp = self.dates[position].tz_localize("America/New_York") + pd.DateOffset(
            hours=hour
        )
        return {
            "input_order": 0,
            "event_id": "event_test",
            "ticker": "TEST",
            "event_timestamp_et": stamp.isoformat(),
        }

    def test_pre_market_starts_same_day_and_uses_prior_controls(self) -> None:
        result = compute_event_market_features(
            self.event(80, 8), self.stock, self.market, self.config
        )
        self.assertEqual(result["market_status"], "complete")
        self.assertEqual(result["response_start_date"], self.dates[80].date().isoformat())
        self.assertEqual(result["control_end_date"], self.dates[79].date().isoformat())
        self.assertAlmostEqual(result["CAR_5"], 0.025, places=10)
        self.assertAlmostEqual(result["CAR_60"], 0.300, places=10)
        self.assertAlmostEqual(result["PreRet_60"], 0.600, places=10)
        self.assertAlmostEqual(result["VOL_60"], 0.0, places=10)
        self.assertAlmostEqual(result["LogAvgVolume_60"], math.log(1_000_000))

    def test_post_market_starts_next_day_and_includes_event_day_controls(self) -> None:
        result = compute_event_market_features(
            self.event(80, 17), self.stock, self.market, self.config
        )
        self.assertEqual(result["market_status"], "complete")
        self.assertEqual(result["response_start_date"], self.dates[81].date().isoformat())
        self.assertEqual(result["control_end_date"], self.dates[80].date().isoformat())

    def test_regular_hours_excludes_same_day_from_controls(self) -> None:
        result = compute_event_market_features(
            self.event(80, 12), self.stock, self.market, self.config
        )
        self.assertEqual(result["response_start_date"], self.dates[81].date().isoformat())
        self.assertEqual(result["control_end_date"], self.dates[79].date().isoformat())

    def test_missing_post_return_is_reported_not_filled(self) -> None:
        stock = self.stock.copy()
        stock.loc[self.dates[82], "Adj Close"] = float("nan")
        result = compute_event_market_features(
            self.event(80, 17), stock, self.market, self.config
        )
        self.assertEqual(result["market_status"], "missing_post_event_return")

    def test_class_share_ticker_candidates_are_conservative(self) -> None:
        self.assertEqual(yahoo_ticker_candidates("BF.B"), ["BF.B", "BF-B", "BFB"])
        self.assertEqual(yahoo_ticker_candidates("AX.DL"), ["AX.DL"])


if __name__ == "__main__":
    unittest.main()

