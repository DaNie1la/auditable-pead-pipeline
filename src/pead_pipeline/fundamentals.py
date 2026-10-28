from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundamentalConfig:
    sue_history_quarters: int = 8
    sue_min_history: int = 4
    earnings_lookback_days: int = 3
    max_share_staleness_days: int = 400
    turnover_window: int = 60


def normalize_earnings(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["EPS Estimate", "Reported EPS", "Surprise(%)"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"earnings data are missing required columns: {missing}")
    earnings = frame[required].copy()
    index = pd.to_datetime(earnings.index, errors="coerce", utc=True)
    earnings.index = index.tz_convert("America/New_York")
    earnings = earnings[~earnings.index.isna()]
    earnings = earnings[~earnings.index.duplicated(keep="last")].sort_index()
    for column in required:
        earnings[column] = pd.to_numeric(earnings[column], errors="coerce")
    return earnings


def normalize_shares(values: pd.Series) -> pd.Series:
    shares = pd.to_numeric(values, errors="coerce")
    index = pd.to_datetime(shares.index, errors="coerce", utc=True)
    index = index.tz_convert("America/New_York").tz_localize(None).normalize()
    shares.index = index
    shares = shares[~shares.index.isna()]
    shares = shares[shares.gt(0)]
    return shares[~shares.index.duplicated(keep="last")].sort_index()


def _shares_asof(
    shares: pd.Series, dates: pd.DatetimeIndex, max_staleness_days: int
) -> tuple[pd.Series, pd.Series]:
    source = pd.Series(shares.index, index=shares.index)
    union = shares.index.union(dates).sort_values()
    aligned_shares = shares.reindex(union).ffill().reindex(dates)
    source_dates = source.reindex(union).ffill().reindex(dates)
    staleness = pd.Series(
        (dates - pd.DatetimeIndex(source_dates)).days, index=dates, dtype="float64"
    )
    invalid = staleness.gt(max_staleness_days) | staleness.lt(0)
    aligned_shares[invalid] = float("nan")
    source_dates[invalid] = pd.NaT
    return aligned_shares, source_dates


def compute_share_features(
    control_end_date: str,
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    shares: pd.Series,
    config: FundamentalConfig | None = None,
) -> dict[str, object]:
    config = config or FundamentalConfig()
    control_end = pd.Timestamp(control_end_date).normalize()
    calendar = benchmark_bars.index[benchmark_bars["Adj Close"].notna()]
    dates = calendar[calendar <= control_end][-config.turnover_window :]
    if len(dates) < config.turnover_window:
        return {"share_status": "insufficient_turnover_calendar"}
    prices = stock_bars["Close"].reindex(dates)
    volumes = stock_bars["Volume"].reindex(dates)
    if prices.isna().any() or volumes.isna().any():
        return {"share_status": "missing_turnover_market_data"}
    aligned_shares, source_dates = _shares_asof(
        shares, dates, config.max_share_staleness_days
    )
    if aligned_shares.isna().any():
        return {"share_status": "shares_missing_or_stale"}
    market_cap = float(prices.iloc[-1] * aligned_shares.iloc[-1])
    turnover = volumes / aligned_shares
    mean_turnover = float(turnover.mean())
    if market_cap <= 0 or mean_turnover <= 0:
        return {"share_status": "nonpositive_share_feature"}
    asof_date = pd.Timestamp(source_dates.iloc[-1])
    return {
        "share_status": "complete",
        "Shares_AsOf": float(aligned_shares.iloc[-1]),
        "Shares_AsOf_Date": asof_date.date().isoformat(),
        "Shares_Staleness_Days": int((control_end - asof_date).days),
        "Size_LogMarketCap": math.log(market_cap),
        "Turnover_60_Mean": mean_turnover,
        "LogTurnover_60": math.log(mean_turnover),
    }


def compute_sue_features(
    event_timestamp_et: str,
    earnings: pd.DataFrame,
    config: FundamentalConfig | None = None,
) -> dict[str, object]:
    config = config or FundamentalConfig()
    call = pd.Timestamp(event_timestamp_et)
    if call.tzinfo is None:
        call = call.tz_localize("America/New_York")
    else:
        call = call.tz_convert("America/New_York")
    normalized = normalize_earnings(earnings)
    call_date = call.tz_localize(None).normalize()
    earnings_dates = normalized.index.tz_localize(None).normalize()
    day_delta = (call_date - earnings_dates).days
    # The transcript proves earnings were available by the call. Accept the
    # provider's same-calendar-day timestamp even if its intraday time is coarse,
    # but never match an earnings record dated after the call date.
    eligible = normalized[
        (day_delta >= 0) & (day_delta <= config.earnings_lookback_days)
    ].copy()
    if eligible.empty:
        return {"sue_status": "no_earnings_match"}
    eligible["_abs_delta"] = abs(
        (call - eligible.index).total_seconds() / 3600
    )
    matched_timestamp = eligible["_abs_delta"].idxmin()
    matched = normalized.loc[matched_timestamp]
    if pd.isna(matched["Reported EPS"]):
        return {"sue_status": "reported_eps_missing"}

    complete = normalized.dropna(subset=["EPS Estimate", "Reported EPS"]).copy()
    complete["forecast_error"] = complete["Reported EPS"] - complete["EPS Estimate"]
    history = complete[complete.index < matched_timestamp].tail(
        config.sue_history_quarters
    )
    current_error = (
        float(matched["Reported EPS"] - matched["EPS Estimate"])
        if pd.notna(matched["EPS Estimate"])
        else float("nan")
    )
    result: dict[str, object] = {
        "earnings_timestamp_et": matched_timestamp.isoformat(),
        "earnings_match_hours": float(
            (call - matched_timestamp).total_seconds() / 3600
        ),
        "EPS_Estimate": float(matched["EPS Estimate"])
        if pd.notna(matched["EPS Estimate"])
        else float("nan"),
        "Reported_EPS": float(matched["Reported EPS"]),
        "EPS_Forecast_Error": current_error,
        "EPS_Surprise_Decimal": float(matched["Surprise(%)"]) / 100
        if pd.notna(matched["Surprise(%)"])
        else float("nan"),
        "SUE_History_Count": int(len(history)),
    }
    reported_history = normalized.loc[:matched_timestamp, "Reported EPS"].dropna()
    yoy_growth = float("nan")
    if len(reported_history) >= 5:
        prior_year = float(reported_history.iloc[-5])
        if abs(prior_year) > 1e-12:
            yoy_growth = (float(reported_history.iloc[-1]) - prior_year) / abs(
                prior_year
            )
    result["EPS_YoY_Growth"] = yoy_growth
    if pd.isna(current_error):
        return result | {"sue_status": "eps_estimate_missing"}
    if len(history) < config.sue_min_history:
        return result | {"sue_status": "insufficient_sue_history"}
    denominator = float(history["forecast_error"].std(ddof=1))
    if not math.isfinite(denominator) or denominator <= 0:
        return result | {"sue_status": "invalid_sue_denominator"}
    return result | {
        "sue_status": "complete",
        "SUE_Analyst_8Q": current_error / denominator,
        "SUE_History_Error_SD": denominator,
    }

