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

from pead_pipeline.fundamentals import (  # noqa: E402
    FundamentalConfig,
    compute_share_features,
    compute_sue_features,
)


class Phase4FundamentalTests(unittest.TestCase):
    def test_sue_uses_only_prior_forecast_errors(self) -> None:
        dates = pd.date_range(
            "2018-01-01 16:00", periods=10, freq="90D", tz="America/New_York"
        )
        estimates = pd.Series(range(10), index=dates, dtype=float)
        errors = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 4, 100], index=dates, dtype=float)
        earnings = pd.DataFrame(
            {
                "EPS Estimate": estimates,
                "Reported EPS": estimates + errors,
                "Surprise(%)": errors * 10,
            }
        )
        call = dates[8] + pd.DateOffset(hours=1)
        result = compute_sue_features(call.isoformat(), earnings)
        expected_sd = errors.iloc[:8].std(ddof=1)
        self.assertEqual(result["sue_status"], "complete")
        self.assertAlmostEqual(result["SUE_Analyst_8Q"], 4 / expected_sd)
        self.assertEqual(result["SUE_History_Count"], 8)

    def test_sue_requires_minimum_history(self) -> None:
        dates = pd.date_range(
            "2020-01-01 16:00", periods=4, freq="90D", tz="America/New_York"
        )
        earnings = pd.DataFrame(
            {
                "EPS Estimate": [1, 1, 1, 1],
                "Reported EPS": [2, 2, 2, 2],
                "Surprise(%)": [100, 100, 100, 100],
            },
            index=dates,
        )
        result = compute_sue_features(
            (dates[-1] + pd.DateOffset(hours=1)).isoformat(), earnings
        )
        self.assertEqual(result["sue_status"], "insufficient_sue_history")

    def test_share_features_are_point_in_time(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=80)
        stock = pd.DataFrame(
            {"Close": 10.0, "Adj Close": 10.0, "Volume": 1_000_000.0},
            index=dates,
        )
        market = pd.DataFrame(
            {"Close": 100.0, "Adj Close": 100.0, "Volume": 10_000_000.0},
            index=dates,
        )
        shares = pd.Series(
            [100_000_000.0, 200_000_000.0],
            index=[dates[0], dates[-1] + pd.DateOffset(days=1)],
        )
        result = compute_share_features(
            dates[-1].date().isoformat(), stock, market, shares
        )
        self.assertEqual(result["share_status"], "complete")
        self.assertEqual(result["Shares_AsOf"], 100_000_000.0)
        self.assertAlmostEqual(result["Size_LogMarketCap"], math.log(1_000_000_000))
        self.assertAlmostEqual(result["Turnover_60_Mean"], 0.01)


if __name__ == "__main__":
    unittest.main()

