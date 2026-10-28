from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.market import (  # noqa: E402
    MarketConfig,
    compute_event_market_features,
    normalize_daily_bars,
    yahoo_ticker_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Yahoo daily bars and build audited PEAD market features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "finbert_section_scores.csv",
    )
    parser.add_argument(
        "--phase3-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase3_finbert_quality.json",
    )
    parser.add_argument(
        "--legacy-events",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "events_unique_safe.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "yahoo_daily",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase4_market_features.csv",
    )
    parser.add_argument(
        "--complete-output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "pead_text_market_complete_cases.csv",
    )
    parser.add_argument(
        "--ticker-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "phase4_ticker_coverage.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase4_market_quality.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase4_market_quality.md",
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--limit-tickers", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker}.csv.gz"


def load_cached_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return normalize_daily_bars(frame)


def _extract_symbol(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()
    if symbol in downloaded.columns.get_level_values(0):
        return downloaded[symbol].copy()
    if symbol in downloaded.columns.get_level_values(-1):
        return downloaded.xs(symbol, axis=1, level=-1).copy()
    return pd.DataFrame()


def _download(symbols: list[str], start: str, end: str, threads: int) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        repair=False,
        actions=False,
        group_by="ticker",
        threads=threads,
        progress=False,
        timeout=30,
    )


def acquire_bars(
    tickers: list[str],
    cache_dir: Path,
    start: str,
    end: str,
    chunk_size: int,
    threads: int,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing = [ticker for ticker in tickers if not cache_path(cache_dir, ticker).exists()]
    for offset in range(0, len(missing), chunk_size):
        chunk = missing[offset : offset + chunk_size]
        primary = {ticker: yahoo_ticker_candidates(ticker)[0] for ticker in chunk}
        downloaded = _download(sorted(set(primary.values())), start, end, threads)
        unresolved: list[str] = []
        for ticker in chunk:
            symbol = primary[ticker]
            frame = _extract_symbol(downloaded, symbol)
            try:
                bars = normalize_daily_bars(frame)
            except ValueError:
                bars = pd.DataFrame()
            if bars.empty or not bars["Close"].notna().any():
                unresolved.append(ticker)
                continue
            bars.to_csv(cache_path(cache_dir, ticker), index_label="Date", compression="gzip")
            rows.append(
                {
                    "ticker": ticker,
                    "provider_ticker": symbol,
                    "download_status": "complete",
                    "bars": int(len(bars)),
                }
            )

        # Only class-share tickers have conservative alternate Yahoo spellings.
        for ticker in unresolved:
            saved = False
            for symbol in yahoo_ticker_candidates(ticker)[1:]:
                frame = _download([symbol], start, end, 1)
                try:
                    bars = normalize_daily_bars(frame)
                except ValueError:
                    bars = pd.DataFrame()
                if bars.empty or not bars["Close"].notna().any():
                    continue
                bars.to_csv(
                    cache_path(cache_dir, ticker), index_label="Date", compression="gzip"
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "provider_ticker": symbol,
                        "download_status": "complete_alias",
                        "bars": int(len(bars)),
                    }
                )
                saved = True
                break
            if not saved:
                rows.append(
                    {
                        "ticker": ticker,
                        "provider_ticker": primary[ticker],
                        "download_status": "unavailable",
                        "bars": 0,
                    }
                )
        print(
            f"downloaded_or_checked={min(offset + len(chunk), len(missing))}/{len(missing)}",
            flush=True,
        )

    prior = set(missing)
    for ticker in tickers:
        if ticker in prior:
            continue
        path = cache_path(cache_dir, ticker)
        rows.append(
            {
                "ticker": ticker,
                "provider_ticker": "cached",
                "download_status": "cached",
                "bars": int(len(load_cached_bars(path))),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def write_report(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    args: argparse.Namespace,
    input_hash: str,
    config: MarketConfig,
    elapsed: float,
) -> None:
    complete = features[features["market_status"].eq("complete")]
    statuses = features["market_status"].value_counts().to_dict()
    downloads = coverage["download_status"].value_counts().to_dict()
    event_tickers = set(features["ticker"].dropna().unique())
    event_ticker_coverage = coverage[coverage["ticker"].isin(event_tickers)]
    event_tickers_with_bars = int(
        event_ticker_coverage["download_status"].isin(
            ["complete", "complete_alias", "cached"]
        ).sum()
    )
    legacy_audit = None
    if args.legacy_events.exists():
        legacy = pd.read_csv(args.legacy_events)
        comparable = complete.merge(
            legacy[
                [
                    "event_id",
                    "SUE_Proxy",
                    "VOL",
                    "Turnover",
                    "CAR_5",
                    "CAR_10",
                    "CAR_60",
                ]
            ],
            on="event_id",
            suffixes=("_new", "_old"),
        )
        pairs = {
            "PreRet_60_vs_legacy_SUE_Proxy": ("PreRet_60", "SUE_Proxy"),
            "VOL_60_vs_legacy_VOL": ("VOL_60", "VOL"),
            "LogAvgVolume_60_vs_legacy_Turnover": (
                "LogAvgVolume_60",
                "Turnover",
            ),
            "CAR_5_new_vs_old": ("CAR_5_new", "CAR_5_old"),
            "CAR_10_new_vs_old": ("CAR_10_new", "CAR_10_old"),
            "CAR_60_new_vs_old": ("CAR_60_new", "CAR_60_old"),
        }
        comparisons = {}
        for label, (new_column, old_column) in pairs.items():
            values = comparable[[new_column, old_column]].dropna()
            comparisons[label] = {
                "rows": len(values),
                "pearson_correlation": float(values[new_column].corr(values[old_column])),
                "exact_within_1e_6": int(
                    ((values[new_column] - values[old_column]).abs() < 1e-6).sum()
                ),
            }
        legacy_audit = {
            "comparable_events": len(comparable),
            "comparisons": comparisons,
            "conclusion": "legacy market columns are not event-aligned and must not be reused",
        }
    report = {
        "phase": 4,
        "status": "complete_with_documented_gaps"
        if len(complete) < len(features)
        else "complete",
        "input": {"path": str(args.input), "sha256": input_hash, "rows": len(features)},
        "provider": {
            "name": "Yahoo Finance via yfinance",
            "yfinance_version": yf.__version__,
            "download_status_counts": downloads,
        },
        "config": asdict(config) | {"config_hash": config.sha256},
        "coverage": {
            "events_total": len(features),
            "events_complete": len(complete),
            "event_coverage_rate": len(complete) / len(features),
            "event_status_counts": statuses,
            "event_tickers_total": len(event_tickers),
            "event_tickers_with_bars": event_tickers_with_bars,
            "provider_symbols_total_including_benchmark": int(
                coverage["ticker"].nunique()
            ),
        },
        "definitions": {
            "CAR_n": "sum of adjusted-close stock return minus SPY adjusted-close return over the first n tradable sessions after the call",
            "PreRet_60": "sum of the 60 pre-event adjusted-close stock returns; this was mislabeled SUE_Proxy in the legacy dataset",
            "VOL_60": "sample standard deviation of the same 60 pre-event stock returns",
            "LogAvgVolume_60": "natural log of mean raw volume over the 60-session pre-event window",
            "LogAvgDollarVolume_60": "natural log of mean Close times Volume over the 60-session pre-event window",
        },
        "legacy_alignment_audit": legacy_audit,
        "runtime": {"elapsed_seconds": elapsed},
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(features),
            "complete_cases_path": str(args.complete_output),
            "complete_cases_sha256": sha256_file(args.complete_output),
            "complete_cases_rows": len(complete),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    legacy_lines: list[str] = []
    if legacy_audit is not None:
        legacy_lines = [
            "## 旧主表对齐审计",
            "",
            f"在 {legacy_audit['comparable_events']:,} 个可比较事件中，旧市场列与重建结果的相关性接近零，证明旧列存在大规模事件错位，不能继续用于建模。",
            f"- 公告前收益与旧 `SUE_Proxy`：相关系数 {legacy_audit['comparisons']['PreRet_60_vs_legacy_SUE_Proxy']['pearson_correlation']:.4f}",
            f"- 新旧 CAR_5：相关系数 {legacy_audit['comparisons']['CAR_5_new_vs_old']['pearson_correlation']:.4f}",
            f"- 新旧 CAR_10：相关系数 {legacy_audit['comparisons']['CAR_10_new_vs_old']['pearson_correlation']:.4f}",
            f"- 新旧 CAR_60：相关系数 {legacy_audit['comparisons']['CAR_60_new_vs_old']['pearson_correlation']:.4f}",
            "",
        ]
    lines = [
        "# 第四阶段市场变量重建报告",
        "",
        "## 结论",
        "",
        f"已完成 {len(complete):,} / {len(features):,} 个事件的可审计市场变量，覆盖率为 {len(complete) / len(features):.2%}。",
        "Yahoo 当前无法返回的代码（其中包括退市或并购证券）保持缺失并单列，不使用零值或旧错配结果填充。",
        "",
        "## 修正后的口径",
        "",
        "- CAR：个股复权收盘收益减 SPY 复权收盘收益，按 5、10、60 个交易日算术累加。",
        "- 盘前会议从当日开始；盘中和盘后会议从下一交易日开始。",
        "- `PreRet_60`：公告前 60 个交易日收益之和；旧项目将其误标为 `SUE_Proxy`。",
        "- `VOL_60`：公告前 60 个交易日收益的样本标准差。",
        "- `LogAvgVolume_60` 与 `LogAvgDollarVolume_60`：交易活跃度代理；不再误称公司规模或真实换手率。",
        "",
        "## 覆盖",
        "",
        f"- 事件总数：{len(features):,}",
        f"- 完整事件：{len(complete):,}",
        f"- 事件股票代码总数：{len(event_tickers):,}",
        f"- 有行情的事件代码：{event_tickers_with_bars:,}",
        f"- 输出 SHA-256：`{report['output']['sha256']}`",
        f"- 文本与市场完整交集：{len(complete):,} 条",
        f"- 完整交集 SHA-256：`{report['output']['complete_cases_sha256']}`",
        "",
        *legacy_lines,
        "## 尚未混入的变量",
        "",
        "规范的 SUE 需要逐期实际 EPS 与当时分析师一致预期；历史市值和真实换手率需要点时流通股本。",
        "在取得相应点时数据前，这三项不会由价格或成交量冒充。",
        "",
    ]
    args.report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.threads <= 0:
        raise ValueError("chunk size and thread count must be positive")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    phase3 = json.loads(args.phase3_report.read_text(encoding="utf-8"))
    input_hash = sha256_file(args.input)
    if input_hash != phase3["output"]["sha256"]:
        raise ValueError("Input scores hash does not match the Phase 3 report")
    events = pd.read_csv(args.input)
    if not events["event_id"].is_unique:
        raise ValueError("Phase 3 event IDs are not unique")

    config = MarketConfig()
    tickers = sorted(events["ticker"].dropna().unique().tolist())
    if args.limit_tickers is not None:
        tickers = tickers[: args.limit_tickers]
        events = events[events["ticker"].isin(tickers)].copy()
    all_tickers = [config.benchmark_ticker] + [
        ticker for ticker in tickers if ticker != config.benchmark_ticker
    ]
    start_time = time.perf_counter()
    coverage = acquire_bars(
        all_tickers,
        args.cache_dir,
        start="2017-07-01",
        end="2023-06-15",
        chunk_size=args.chunk_size,
        threads=args.threads,
    )
    args.ticker_report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.ticker_report, index=False)
    benchmark_path = cache_path(args.cache_dir, config.benchmark_ticker)
    if not benchmark_path.exists():
        raise RuntimeError("Benchmark data could not be downloaded")
    benchmark = load_cached_bars(benchmark_path)

    feature_rows: list[dict[str, object]] = []
    for ticker, ticker_events in events.groupby("ticker", sort=True):
        path = cache_path(args.cache_dir, ticker)
        if not path.exists():
            for _, event in ticker_events.iterrows():
                feature_rows.append(
                    {
                        "input_order": int(event["input_order"]),
                        "event_id": event["event_id"],
                        "ticker": ticker,
                        "event_timestamp_et": event["event_timestamp_et"],
                        "market_session": None,
                        "benchmark_ticker": config.benchmark_ticker,
                        "market_config_hash": config.sha256,
                        "market_status": "ticker_data_unavailable",
                    }
                )
            continue
        stock = load_cached_bars(path)
        for _, event in ticker_events.iterrows():
            feature_rows.append(
                compute_event_market_features(event, stock, benchmark, config)
            )
    features = pd.DataFrame(feature_rows).sort_values("input_order").reset_index(
        drop=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, float_format="%.10f")
    feature_payload = features.drop(
        columns=[
            "input_order",
            "ticker",
            "event_timestamp_et",
            "market_config_hash",
        ],
        errors="ignore",
    )
    complete_cases = events.merge(
        feature_payload, on="event_id", how="inner", validate="one_to_one"
    )
    complete_cases = complete_cases[
        complete_cases["market_status"].eq("complete")
    ].copy()
    complete_cases["Tone_Gap_Pres_minus_QA"] = (
        complete_cases["Tone_Pres"] - complete_cases["Tone_QA"]
    )
    args.complete_output.parent.mkdir(parents=True, exist_ok=True)
    complete_cases.to_csv(args.complete_output, index=False, float_format="%.10f")
    elapsed = time.perf_counter() - start_time
    write_report(features, coverage, args, input_hash, config, elapsed)
    print(f"events={len(features)} complete={features.market_status.eq('complete').sum()}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

