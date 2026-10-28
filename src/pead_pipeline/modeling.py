from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


FINANCIAL_FEATURES = [
    "SUE_Analyst_8Q",
    "Size_LogMarketCap",
    "VOL_60",
    "LogTurnover_60",
    "PreRet_60",
]
TEXT_FEATURES = ["Tone_Pres", "Tone_QA"]
INTERACTION_FEATURES = ["SUE_x_Tone_Pres", "SUE_x_Tone_QA"]
TARGETS = ["CAR_5", "CAR_10", "CAR_60"]


@dataclass(frozen=True)
class ModelingConfig:
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    random_seed: int = 42
    n_estimators: int = 150
    learning_rate: float = 0.03
    max_depth: int = 3
    subsample: float = 0.7
    colsample_bytree: float = 0.7


class TrainFittedTransformer:
    """Train-only winsorization and standardization for continuous predictors."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        if not 0 <= lower < upper <= 1:
            raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower = lower
        self.upper = upper
        self.bounds_: pd.DataFrame | None = None
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "TrainFittedTransformer":
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError("training predictors contain missing or non-numeric values")
        self.bounds_ = pd.DataFrame(
            {"lower": numeric.quantile(self.lower), "upper": numeric.quantile(self.upper)}
        )
        clipped = numeric.clip(
            lower=self.bounds_["lower"], upper=self.bounds_["upper"], axis="columns"
        )
        self.mean_ = clipped.mean()
        scale = clipped.std(ddof=0)
        self.scale_ = scale.mask(scale.eq(0), 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.bounds_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("transformer must be fitted before transform")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError("predictors contain missing or non-numeric values")
        clipped = numeric.clip(
            lower=self.bounds_["lower"], upper=self.bounds_["upper"], axis="columns"
        )
        return (clipped - self.mean_) / self.scale_

    def export_parameters(self) -> pd.DataFrame:
        if self.bounds_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("transformer must be fitted before export")
        result = self.bounds_.copy()
        result["mean_after_clipping"] = self.mean_
        result["sd_after_clipping"] = self.scale_
        result.index.name = "feature"
        return result.reset_index()


def prepare_modeling_frame(
    frame: pd.DataFrame, config: ModelingConfig | None = None
) -> tuple[pd.DataFrame, TrainFittedTransformer]:
    config = config or ModelingConfig()
    required = {
        "event_id",
        "ticker",
        "event_timestamp_et",
        *FINANCIAL_FEATURES,
        *TEXT_FEATURES,
        *TARGETS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"analysis data are missing required columns: {missing}")
    result = frame.copy()
    timestamp = pd.to_datetime(result["event_timestamp_et"], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("event_timestamp_et contains unparseable timestamps")
    local = timestamp.dt.tz_convert("America/New_York")
    result["event_date"] = local.dt.tz_localize(None).dt.normalize()
    result["event_month"] = result["event_date"].dt.to_period("M").astype(str)
    result["event_quarter"] = result["event_date"].dt.to_period("Q").astype(str)
    train_end = pd.Timestamp(config.train_end)
    test_start = pd.Timestamp(config.test_start)
    if test_start <= train_end:
        raise ValueError("test_start must be after train_end")
    result["sample_split"] = np.select(
        [result["event_date"].le(train_end), result["event_date"].ge(test_start)],
        ["train", "test"],
        default="gap",
    )
    if result["sample_split"].eq("gap").any():
        raise ValueError("events fall into the gap between training and testing periods")
    source_features = FINANCIAL_FEATURES + TEXT_FEATURES
    transformer = TrainFittedTransformer(
        lower=config.winsor_lower, upper=config.winsor_upper
    ).fit(result.loc[result["sample_split"].eq("train"), source_features])
    transformed = transformer.transform(result[source_features])
    result[[f"Z_{column}" for column in source_features]] = transformed.to_numpy()
    result["SUE_x_Tone_Pres"] = (
        result["Z_SUE_Analyst_8Q"] * result["Z_Tone_Pres"]
    )
    result["SUE_x_Tone_QA"] = result["Z_SUE_Analyst_8Q"] * result["Z_Tone_QA"]
    return result, transformer


def model_feature_sets() -> dict[str, list[str]]:
    financial = [f"Z_{column}" for column in FINANCIAL_FEATURES]
    text = [f"Z_{column}" for column in TEXT_FEATURES]
    return {
        "financial": financial,
        "additive": financial + text,
        "interaction": financial + text + INTERACTION_FEATURES,
    }


def safe_spearman(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if mask.sum() < 3:
        return float("nan")
    if np.unique(actual_array[mask]).size < 2 or np.unique(predicted_array[mask]).size < 2:
        return float("nan")
    return float(spearmanr(actual_array[mask], predicted_array[mask]).statistic)


def prediction_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    months: pd.Series,
    minimum_month_size: int = 20,
) -> dict[str, float | int]:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()
    actual = actual[valid]
    predicted = predicted[valid]
    months = months[valid]
    monthly_ics: list[float] = []
    for _, index in months.groupby(months).groups.items():
        if len(index) < minimum_month_size:
            continue
        ic = safe_spearman(actual.loc[index], predicted.loc[index])
        if np.isfinite(ic):
            monthly_ics.append(ic)
    ic_mean = float(np.mean(monthly_ics)) if monthly_ics else float("nan")
    ic_sd = float(np.std(monthly_ics, ddof=1)) if len(monthly_ics) > 1 else float("nan")
    return {
        "n": int(len(actual)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "pooled_rank_ic": safe_spearman(actual, predicted),
        "monthly_rank_ic_mean": ic_mean,
        "monthly_rank_ic_sd": ic_sd,
        "monthly_rank_ic_ir": (
            float(ic_mean / ic_sd) if np.isfinite(ic_sd) and ic_sd > 0 else float("nan")
        ),
        "monthly_rank_ic_months": int(len(monthly_ics)),
    }


def descriptive_statistics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append(
            {
                "variable": column,
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "median": float(values.median()),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    pearson = frame[columns].corr(method="pearson")
    spearman = frame[columns].corr(method="spearman")
    rows: list[dict[str, float | str]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "pearson": float(pearson.loc[left, right]),
                    "spearman": float(spearman.loc[left, right]),
                }
            )
    return pd.DataFrame(rows)


def monthly_long_short_spreads(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    quantile: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for month, group in frame.groupby("event_month", sort=True):
        valid = group[[actual_column, prediction_column]].dropna()
        if len(valid) < 20:
            continue
        percentile = valid[prediction_column].rank(method="average", pct=True)
        long_returns = valid.loc[percentile.gt(1 - quantile), actual_column]
        short_returns = valid.loc[percentile.le(quantile), actual_column]
        if long_returns.empty or short_returns.empty:
            continue
        rows.append(
            {
                "event_month": month,
                "n_events": int(len(valid)),
                "n_long": int(len(long_returns)),
                "n_short": int(len(short_returns)),
                "long_mean": float(long_returns.mean()),
                "short_mean": float(short_returns.mean()),
                "long_short_spread": float(long_returns.mean() - short_returns.mean()),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["cumulative_arithmetic_spread"] = result["long_short_spread"].cumsum()
    return result
