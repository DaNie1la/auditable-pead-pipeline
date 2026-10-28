from __future__ import annotations

import numpy as np
import pandas as pd

from pead_pipeline.modeling import (
    ModelingConfig,
    TrainFittedTransformer,
    monthly_long_short_spreads,
    prediction_metrics,
    prepare_modeling_frame,
    safe_spearman,
)


def _sample_frame() -> pd.DataFrame:
    dates = [
        "2020-01-01 08:00:00-05:00",
        "2021-06-01 08:00:00-04:00",
        "2022-01-02 08:00:00-05:00",
    ]
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "ticker": ["A", "B", "C"],
            "event_timestamp_et": dates,
            "SUE_Analyst_8Q": [0.0, 1.0, 1000.0],
            "Size_LogMarketCap": [20.0, 21.0, 22.0],
            "VOL_60": [0.01, 0.02, 0.03],
            "LogTurnover_60": [-5.0, -4.0, -3.0],
            "PreRet_60": [-0.1, 0.1, 0.2],
            "Tone_Pres": [-0.2, 0.2, 0.9],
            "Tone_QA": [-0.1, 0.3, 0.8],
            "CAR_5": [0.0, 0.1, 0.2],
            "CAR_10": [0.0, 0.1, 0.2],
            "CAR_60": [0.0, 0.1, 0.2],
        }
    )


def test_transformer_uses_training_bounds_for_future_extreme() -> None:
    transformer = TrainFittedTransformer(0.0, 1.0).fit(
        pd.DataFrame({"x": [0.0, 1.0]})
    )
    transformed = transformer.transform(pd.DataFrame({"x": [1000.0]}))
    assert transformed.iloc[0, 0] == 1.0


def test_prepare_modeling_frame_is_strictly_chronological() -> None:
    result, _ = prepare_modeling_frame(
        _sample_frame(), ModelingConfig(winsor_lower=0.0, winsor_upper=1.0)
    )
    assert result["sample_split"].tolist() == ["train", "train", "test"]
    assert np.isfinite(result["SUE_x_Tone_Pres"]).all()
    assert result.loc[2, "Z_SUE_Analyst_8Q"] == 1.0


def test_safe_spearman_handles_constant_predictions() -> None:
    assert np.isnan(safe_spearman([1, 2, 3], [1, 1, 1]))
    assert safe_spearman([1, 2, 3], [1, 2, 3]) == 1.0


def test_prediction_metrics_reports_monthly_rank_ic() -> None:
    actual = pd.Series(np.arange(40, dtype=float))
    predicted = actual.copy()
    months = pd.Series(["2022-01"] * 20 + ["2022-02"] * 20)
    metrics = prediction_metrics(actual, predicted, months)
    assert metrics["monthly_rank_ic_months"] == 2
    assert metrics["monthly_rank_ic_mean"] == 1.0
    assert metrics["directional_accuracy"] == 1.0


def test_monthly_long_short_spreads_preserves_time_order() -> None:
    frame = pd.DataFrame(
        {
            "event_month": ["2022-02"] * 20 + ["2022-01"] * 20,
            "actual": np.tile(np.arange(20, dtype=float), 2),
            "pred": np.tile(np.arange(20, dtype=float), 2),
        }
    )
    result = monthly_long_short_spreads(frame, "actual", "pred")
    assert result["event_month"].tolist() == ["2022-01", "2022-02"]
    assert (result["long_short_spread"] > 0).all()
