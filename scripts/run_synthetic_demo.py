"""Exercise the public pipeline interfaces with invented data only.

The FinBERT probabilities below are fixtures, not model output. This demo tests
stage integration and privacy-safe execution; it does not reproduce study results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pead_pipeline.events import build_event_id  # noqa: E402
from pead_pipeline.finbert import aggregate_window_probabilities  # noqa: E402
from pead_pipeline.fundamentals import compute_share_features, compute_sue_features  # noqa: E402
from pead_pipeline.market import compute_event_market_features  # noqa: E402
from pead_pipeline.modeling import (  # noqa: E402
    ModelingConfig, model_feature_sets, prediction_metrics, prepare_modeling_frame,
)
from pead_pipeline.transcripts import split_transcript  # noqa: E402


EVENT_DATES = (
    "2018-06-01", "2019-06-03", "2020-06-01", "2021-06-01",
    "2021-08-02", "2022-06-01", "2022-08-01",
)


def synthetic_bars(calendar: pd.DatetimeIndex, ticker_index: int) -> pd.DataFrame:
    step = np.arange(len(calendar), dtype=float)
    daily = 0.0003 + 0.0005 * np.sin(step / 13 + ticker_index)
    daily += ticker_index * 0.00004
    close = 100 * np.cumprod(1 + daily)
    return pd.DataFrame(
        {"Close": close, "Adj Close": close, "Volume": 1_000_000 + 5_000 * ticker_index},
        index=calendar,
    )


def main() -> None:
    calendar = pd.bdate_range("2016-01-01", "2023-06-30")
    benchmark = synthetic_bars(calendar, 0)
    records = []
    for ticker_index in range(1, 5):
        ticker = f"DEMO{ticker_index}"
        stock = synthetic_bars(calendar, ticker_index)
        shares = pd.Series(
            100_000_000 + ticker_index * 1_000_000,
            index=pd.to_datetime(["2016-01-01", "2018-01-01", "2019-01-01",
                                  "2020-01-01", "2021-01-01", "2022-01-01"]),
        )
        earnings_dates = ["2017-03-01", "2017-06-01", "2017-09-01", "2017-12-01"]
        earnings_dates.extend(EVENT_DATES)
        earnings_index = pd.to_datetime(earnings_dates).tz_localize("America/New_York")
        errors = np.linspace(-0.10, 0.12, len(earnings_index)) + ticker_index * 0.007
        earnings = pd.DataFrame(
            {"EPS Estimate": 1.0, "Reported EPS": 1.0 + errors,
             "Surprise(%)": errors * 100},
            index=earnings_index,
        )
        for date in EVENT_DATES:
            call = pd.Timestamp(date).tz_localize("America/New_York") + pd.Timedelta(8, unit="h")
            transcript = (
                "Prepared Remarks: " + "The company discusses sales, costs and guidance. " * 8
                + "Questions and Answers: " + "Analysts ask about margins and demand. " * 8
                + "\nDuration: 40 minutes"
            )
            split = split_transcript(transcript)
            assert split.status == "split_complete"
            # Fixture probabilities only; production Phase 3 uses the fixed FinBERT revision.
            score = aggregate_window_probabilities(
                [[0.55, 0.25, 0.20], [0.45, 0.35, 0.20]],
                [0, 1], [100, 100], 2, 64,
                {"positive": 0, "negative": 1, "neutral": 2},
            )
            event = {
                "input_order": len(records), "event_id": build_event_id(ticker, call),
                "ticker": ticker, "event_timestamp_et": call.isoformat(),
            }
            market = compute_event_market_features(event, stock, benchmark)
            assert market["market_status"] == "complete"
            fundamental = compute_sue_features(call.isoformat(), earnings)
            share = compute_share_features(
                market["control_end_date"], stock, benchmark, shares
            )
            assert fundamental["sue_status"] == share["share_status"] == "complete"
            records.append({
                **event, **market, **fundamental, **share,
                "Tone_Pres": score[0].tone, "Tone_QA": score[1].tone,
            })
    frame, transformer = prepare_modeling_frame(pd.DataFrame(records), ModelingConfig())
    training = frame.loc[frame["sample_split"].eq("train")]
    testing = frame.loc[frame["sample_split"].eq("test")]
    features = model_feature_sets()["financial"]
    model = Ridge(alpha=1.0).fit(training[features], training["CAR_5"])
    predicted = pd.Series(model.predict(testing[features]), index=testing.index)
    metrics = prediction_metrics(
        testing["CAR_5"], predicted, testing["event_month"], minimum_month_size=3
    )
    assert len(frame) == 28 and len(training) == 20 and len(testing) == 8
    assert metrics["n"] == 8 and np.isfinite(metrics["rmse"])
    assert set(transformer.export_parameters()["feature"]) == {
        "SUE_Analyst_8Q", "Size_LogMarketCap", "VOL_60", "LogTurnover_60",
        "PreRet_60", "Tone_Pres", "Tone_QA",
    }
    print(json.dumps({
        "status": "ok", "synthetic_events": len(frame),
        "training_events": len(training), "test_events": len(testing),
        "test_rmse": round(metrics["rmse"], 6), "real_data_used": False,
        "finbert_inference_used": False,
    }))


if __name__ == "__main__":
    main()
