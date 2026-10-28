from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.fundamentals import (  # noqa: E402
    FundamentalConfig,
    compute_share_features,
    compute_sue_features,
    normalize_earnings,
    normalize_shares,
)
from pead_pipeline.market import normalize_daily_bars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time SUE, market-cap, and turnover features."
    )
    parser.add_argument(
        "--market-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase4_market_features.csv",
    )
    parser.add_argument(
        "--text-market",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "pead_text_market_complete_cases.csv",
    )
    parser.add_argument(
        "--market-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "yahoo_daily",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "yahoo_fundamentals",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "phase4_fundamental_features.csv",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "pead_analysis_ready.csv",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "phase4_fundamental_ticker_coverage.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase4_fundamental_quality.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase4_fundamental_quality.md",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-tickers", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_paths(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    return (
        cache_dir / "earnings" / f"{ticker}.csv.gz",
        cache_dir / "shares" / f"{ticker}.csv.gz",
    )


def _fetch_ticker(ticker: str, cache_dir: Path) -> dict[str, object]:
    earnings_path, shares_path = _cache_paths(cache_dir, ticker)
    earnings_path.parent.mkdir(parents=True, exist_ok=True)
    shares_path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {"ticker": ticker}
    if earnings_path.exists():
        row["earnings_download_status"] = "cached"
    else:
        try:
            earnings = yf.Ticker(ticker).get_earnings_dates(limit=100)
            if earnings is None or earnings.empty:
                row["earnings_download_status"] = "unavailable"
            else:
                earnings = normalize_earnings(earnings)
                earnings.to_csv(earnings_path, index_label="Earnings Date", compression="gzip")
                row["earnings_download_status"] = "complete"
        except Exception:
            row["earnings_download_status"] = "error"
    if shares_path.exists():
        row["shares_download_status"] = "cached"
    else:
        try:
            shares = yf.Ticker(ticker).get_shares_full(
                start="2017-01-01", end="2023-06-15"
            )
            if shares is None or shares.empty:
                row["shares_download_status"] = "unavailable"
            else:
                shares = normalize_shares(shares)
                shares.rename("Shares").to_csv(
                    shares_path, index_label="Date", compression="gzip"
                )
                row["shares_download_status"] = "complete"
        except Exception:
            row["shares_download_status"] = "error"
    return row


def acquire_fundamentals(
    tickers: list[str], cache_dir: Path, workers: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_ticker, ticker, cache_dir): ticker
            for ticker in tickers
        }
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(tickers):
                print(f"fundamental_tickers={index}/{len(tickers)}", flush=True)
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def load_earnings(path: Path) -> pd.DataFrame:
    # Parse once in normalize_earnings with utc=True.  Cached Yahoo timestamps can
    # contain mixed daylight-saving offsets, which pandas otherwise warns about.
    frame = pd.read_csv(path, index_col="Earnings Date")
    return normalize_earnings(frame)


def load_shares(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return normalize_shares(frame["Shares"])


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return normalize_daily_bars(frame)


def write_report(
    features: pd.DataFrame,
    analysis: pd.DataFrame,
    coverage: pd.DataFrame,
    args: argparse.Namespace,
    config: FundamentalConfig,
    elapsed: float,
) -> None:
    report = {
        "phase": "4B",
        "status": "complete_with_documented_gaps",
        "provider": {
            "name": "Yahoo Finance via yfinance",
            "yfinance_version": yf.__version__,
            "earnings_download_status": coverage[
                "earnings_download_status"
            ].value_counts().to_dict(),
            "shares_download_status": coverage[
                "shares_download_status"
            ].value_counts().to_dict(),
        },
        "config": asdict(config),
        "coverage": {
            "market_complete_events": len(features),
            "sue_complete_events": int(features["sue_status"].eq("complete").sum()),
            "share_complete_events": int(
                features["share_status"].eq("complete").sum()
            ),
            "analysis_ready_events": len(analysis),
            "sue_status_counts": features["sue_status"].value_counts().to_dict(),
            "share_status_counts": features["share_status"].value_counts().to_dict(),
        },
        "definitions": {
            "SUE_Analyst_8Q": "current actual EPS minus consensus estimate, divided by the sample standard deviation of up to eight prior forecast errors; at least four prior errors required",
            "Size_LogMarketCap": "natural log of unadjusted close times the latest shares observation available by the control date",
            "Turnover_60_Mean": "mean daily raw volume divided by point-in-time shares over the 60-session pre-event window",
            "LogTurnover_60": "natural log of Turnover_60_Mean",
        },
        "runtime": {"elapsed_seconds": elapsed},
        "output": {
            "fundamental_features": {
                "path": str(args.output),
                "rows": len(features),
                "sha256": sha256_file(args.output),
            },
            "analysis_ready": {
                "path": str(args.analysis_output),
                "rows": len(analysis),
                "sha256": sha256_file(args.analysis_output),
            },
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 第四阶段基本面与股本变量报告",
        "",
        "## 结论",
        "",
        f"在 {len(features):,} 个市场窗口完整事件中，{report['coverage']['sue_complete_events']:,} 个获得严格 SUE，{report['coverage']['share_complete_events']:,} 个获得点时市值与真实换手率。",
        f"同时满足文本、市场、SUE、历史市值和真实换手率的事件共 {len(analysis):,} 个。",
        "",
        "## 变量口径",
        "",
        "- `SUE_Analyst_8Q`：实际 EPS 减一致预期 EPS，再除以此前最多 8 次预测误差的样本标准差；至少要求 4 次历史预测误差。",
        "- `Size_LogMarketCap`：控制日未复权收盘价乘当时可得股本后的自然对数。",
        "- `Turnover_60_Mean`：公告前 60 个交易日成交量除以当时股本后的均值。",
        "- `LogTurnover_60`：上述真实换手率均值的自然对数。",
        "",
        "## 防泄漏规则",
        "",
        "股本只使用控制日当日或之前的记录；SUE 分母只使用本次公告之前的预测误差。极值处理不在全样本执行，应在后续每个训练折内部完成。",
        "",
        "## 输出",
        "",
        f"- 基本面审计表：{len(features):,} 条，SHA-256 `{report['output']['fundamental_features']['sha256']}`",
        f"- 最终可建模表：{len(analysis):,} 条，SHA-256 `{report['output']['analysis_ready']['sha256']}`",
        "",
    ]
    args.report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    market = pd.read_csv(args.market_features)
    market = market[market["market_status"].eq("complete")].copy()
    tickers = sorted(market["ticker"].unique().tolist())
    if args.limit_tickers is not None:
        tickers = tickers[: args.limit_tickers]
        market = market[market["ticker"].isin(tickers)].copy()
    start_time = time.perf_counter()
    coverage = acquire_fundamentals(tickers, args.cache_dir, args.workers)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_output, index=False)
    benchmark = load_bars(args.market_cache / "SPY.csv.gz")
    config = FundamentalConfig()
    rows: list[dict[str, object]] = []
    for ticker, events in market.groupby("ticker", sort=True):
        earnings_path, shares_path = _cache_paths(args.cache_dir, ticker)
        earnings = load_earnings(earnings_path) if earnings_path.exists() else None
        shares = load_shares(shares_path) if shares_path.exists() else None
        stock = load_bars(args.market_cache / f"{ticker}.csv.gz")
        for _, event in events.iterrows():
            base = {
                "input_order": int(event["input_order"]),
                "event_id": event["event_id"],
                "ticker": ticker,
                "event_timestamp_et": event["event_timestamp_et"],
            }
            sue = (
                compute_sue_features(event["event_timestamp_et"], earnings, config)
                if earnings is not None
                else {"sue_status": "earnings_data_unavailable"}
            )
            share = (
                compute_share_features(
                    event["control_end_date"], stock, benchmark, shares, config
                )
                if shares is not None
                else {"share_status": "shares_data_unavailable"}
            )
            rows.append(base | sue | share)
    features = pd.DataFrame(rows).sort_values("input_order").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, float_format="%.10f")
    text_market = pd.read_csv(args.text_market)
    payload = features.drop(
        columns=["input_order", "ticker", "event_timestamp_et"], errors="ignore"
    )
    analysis = text_market.merge(
        payload, on="event_id", how="inner", validate="one_to_one"
    )
    analysis = analysis[
        analysis["sue_status"].eq("complete")
        & analysis["share_status"].eq("complete")
    ].copy()
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.analysis_output, index=False, float_format="%.10f")
    elapsed = time.perf_counter() - start_time
    write_report(features, analysis, coverage, args, config, elapsed)
    print(
        f"market_complete={len(features)} analysis_ready={len(analysis)}",
        flush=True,
    )
    print(f"output={args.analysis_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

