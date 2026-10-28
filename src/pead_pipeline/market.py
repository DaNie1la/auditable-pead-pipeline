from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import pandas as pd

from .events import classify_market_session


@dataclass(frozen=True)
class MarketConfig:
    benchmark_ticker: str = "SPY"
    pre_window: int = 60
    car_windows: tuple[int, ...] = (5, 10, 60)
    price_field: str = "Adj Close"
    return_method: str = "arithmetic_sum"
    event_clock: str = "america_new_york"

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def yahoo_ticker_candidates(ticker: str) -> list[str]:
    """Return conservative Yahoo symbol candidates without hiding the source ticker."""

    normalized = str(ticker).strip().upper()
    candidates = [normalized]
    if re.fullmatch(r"[A-Z0-9]+\.[AB]", normalized):
        candidates.extend([normalized.replace(".", "-"), normalized.replace(".", "")])
    result: list[str] = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def normalize_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Close", "Adj Close", "Volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"daily bars are missing required columns: {missing}")
    bars = frame[required].copy()
    index = pd.to_datetime(bars.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    bars.index = index.normalize()
    bars = bars[~bars.index.isna()]
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    for column in required:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars


def _failure(base: dict[str, object], status: str) -> dict[str, object]:
    return base | {"market_status": status}


def compute_event_market_features(
    event: dict[str, object] | pd.Series,
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    config: MarketConfig | None = None,
) -> dict[str, object]:
    """Compute audited event-time returns and pre-event market controls."""

    config = config or MarketConfig()
    stock = normalize_daily_bars(stock_bars)
    benchmark = normalize_daily_bars(benchmark_bars)
    timestamp = pd.Timestamp(event["event_timestamp_et"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    else:
        timestamp = timestamp.tz_convert("America/New_York")
    event_date = timestamp.tz_localize(None).normalize()
    session = classify_market_session(timestamp)
    base: dict[str, object] = {
        "input_order": int(event["input_order"]),
        "event_id": str(event["event_id"]),
        "ticker": str(event["ticker"]),
        "event_timestamp_et": timestamp.isoformat(),
        "market_session": session,
        "benchmark_ticker": config.benchmark_ticker,
        "market_config_hash": config.sha256,
    }

    calendar = benchmark.index[benchmark[config.price_field].notna()]
    if session == "pre_market":
        response_days = calendar[calendar >= event_date]
        control_days = calendar[calendar < event_date]
    elif session == "regular_hours":
        response_days = calendar[calendar > event_date]
        # A regular-hours call can contaminate the same-day close and volume.
        control_days = calendar[calendar < event_date]
    elif session == "post_market":
        response_days = calendar[calendar > event_date]
        control_days = calendar[calendar <= event_date]
    else:
        return _failure(base, "invalid_event_session")
    if len(response_days) == 0:
        return _failure(base, "no_response_trading_day")
    if len(control_days) == 0:
        return _failure(base, "no_control_trading_day")

    response_start = response_days[0]
    control_end = control_days[-1]
    aligned = pd.DataFrame(index=calendar)
    aligned["stock_adj_close"] = stock[config.price_field].reindex(calendar)
    aligned["stock_close"] = stock["Close"].reindex(calendar)
    aligned["stock_volume"] = stock["Volume"].reindex(calendar)
    aligned["benchmark_adj_close"] = benchmark[config.price_field].reindex(calendar)
    aligned["stock_return"] = aligned["stock_adj_close"].pct_change(fill_method=None)
    aligned["benchmark_return"] = aligned["benchmark_adj_close"].pct_change(
        fill_method=None
    )
    aligned["abnormal_return"] = (
        aligned["stock_return"] - aligned["benchmark_return"]
    )

    if response_start not in aligned.index or pd.isna(
        aligned.at[response_start, "abnormal_return"]
    ):
        return _failure(base, "missing_response_start_return")
    response = aligned.loc[response_start:]
    longest = max(config.car_windows)
    if len(response) < longest:
        return _failure(base, "insufficient_post_event_calendar")
    post_returns = response["abnormal_return"].iloc[:longest]
    if post_returns.isna().any():
        return _failure(base, "missing_post_event_return")

    pre = aligned.loc[:control_end].tail(config.pre_window)
    if len(pre) < config.pre_window:
        return _failure(base, "insufficient_pre_event_calendar")
    required_pre = ["stock_return", "stock_close", "stock_volume"]
    if pre[required_pre].isna().any().any():
        return _failure(base, "missing_pre_event_market_data")
    if (pre["stock_close"] <= 0).any() or (pre["stock_volume"] <= 0).any():
        return _failure(base, "nonpositive_pre_event_market_data")

    mean_volume = float(pre["stock_volume"].mean())
    mean_dollar_volume = float((pre["stock_close"] * pre["stock_volume"]).mean())
    result = base | {
        "market_status": "complete",
        "response_start_date": response_start.date().isoformat(),
        "control_end_date": control_end.date().isoformat(),
        # This was labelled SUE in the old project, but is price momentum.
        "PreRet_60": float(pre["stock_return"].sum()),
        "VOL_60": float(pre["stock_return"].std(ddof=1)),
        "LogAvgVolume_60": math.log(mean_volume),
        "LogAvgDollarVolume_60": math.log(mean_dollar_volume),
    }
    for window in config.car_windows:
        result[f"CAR_{window}"] = float(post_returns.iloc[:window].sum())
    return result

