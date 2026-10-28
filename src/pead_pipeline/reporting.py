from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .events import MASTER_VALUE_COLUMNS, RawTranscriptStatus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_range(frame: pd.DataFrame) -> dict[str, str | None]:
    values = frame["event_timestamp_et"].dropna()
    if values.empty:
        return {"min": None, "max": None}
    return {"min": values.min().isoformat(), "max": values.max().isoformat()}


def _tail_diagnostics(frame: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    diagnostics: dict[str, dict[str, float | bool]] = {}
    for column in MASTER_VALUE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        q01 = float(values.quantile(0.01))
        q99 = float(values.quantile(0.99))
        minimum = float(values.min())
        maximum = float(values.max())
        diagnostics[column] = {
            "min": minimum,
            "q01": q01,
            "q99": q99,
            "max": maximum,
            "values_outside_1_99_percentiles": bool(minimum < q01 or maximum > q99),
        }
    return diagnostics


def _ticker_correction_count(frame: pd.DataFrame) -> int:
    basic = (
        frame["ticker"].fillna("").astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    )
    return int(basic.ne(frame["ticker_normalized"]).sum())


def build_quality_report(
    master_path: Path,
    nlp_path: Path,
    master_audited: pd.DataFrame,
    safe: pd.DataFrame,
    conflicts: pd.DataFrame,
    nlp_audited: pd.DataFrame,
    raw_status: RawTranscriptStatus,
) -> dict[str, object]:
    conflict_groups = int(conflicts["event_id"].nunique())
    exact_groups = int(
        master_audited.loc[
            master_audited["event_status"] == "exact_duplicate", "event_id"
        ].nunique()
    )
    overlap = len(
        set(master_audited["event_id"]) & set(nlp_audited["event_id"])
    )
    timestamp_resolution = {
        str(key): int(value)
        for key, value in master_audited["timestamp_resolution"].value_counts().items()
    }
    report: dict[str, object] = {
        "phase": 1,
        "source_files": {
            "master": {
                "path": str(master_path),
                "size_bytes": master_path.stat().st_size,
                "sha256": sha256_file(master_path),
            },
            "nlp_scores": {
                "path": str(nlp_path),
                "size_bytes": nlp_path.stat().st_size,
                "sha256": sha256_file(nlp_path),
            },
        },
        "raw_transcripts": asdict(raw_status),
        "master": {
            "rows": len(master_audited),
            "unique_tickers": int(master_audited["ticker_normalized"].nunique()),
            "ticker_corrections": _ticker_correction_count(master_audited),
            "source_timestamps_parsed": int(
                master_audited["source_timestamp_et"].notna().sum()
            ),
            "parsed_timestamps": int(master_audited["event_timestamp_et"].notna().sum()),
            "unparsed_timestamps": int(master_audited["event_timestamp_et"].isna().sum()),
            "timestamp_resolution": timestamp_resolution,
            "timestamp_range": _timestamp_range(master_audited),
            "event_groups": int(master_audited["event_id"].nunique()),
            "safe_unique_events": len(safe),
            "exact_duplicate_groups": exact_groups,
            "conflict_groups": conflict_groups,
            "conflict_rows": len(conflicts),
            "missing_values": {
                column: int(value)
                for column, value in master_audited.isna().sum().items()
                if value
            },
            "tail_diagnostics": _tail_diagnostics(master_audited),
        },
        "nlp_scores": {
            "rows": len(nlp_audited),
            "unique_tickers": int(nlp_audited["ticker_normalized"].nunique()),
            "ticker_corrections": _ticker_correction_count(nlp_audited),
            "parsed_timestamps": int(nlp_audited["event_timestamp_et"].notna().sum()),
            "unparsed_timestamps": int(nlp_audited["event_timestamp_et"].isna().sum()),
            "timestamp_range": _timestamp_range(nlp_audited),
            "event_groups": int(nlp_audited["event_id"].nunique()),
            "exact_duplicate_rows": int(nlp_audited["exact_row_duplicate"].sum()),
            "duplicate_event_groups": int(
                nlp_audited.loc[nlp_audited["event_row_count"] > 1, "event_id"].nunique()
            ),
            "tone_conflict_groups": int(
                nlp_audited.loc[
                    nlp_audited["tone_score_distinct_count"] > 1, "event_id"
                ].nunique()
            ),
        },
        "cross_file": {
            "overlapping_event_ids": overlap,
            "master_rows_relative_to_nlp_rows": len(master_audited) / len(nlp_audited),
            "row_loss_rate_from_nlp_to_master": 1 - len(master_audited) / len(nlp_audited),
        },
        "decisions": [
            "Source files were not overwritten.",
            "Conflicting event groups were excluded from the safe event table.",
            "No conflicting row was selected as correct without source evidence.",
            "Raw transcript absence remains an explicit dependency.",
            "Midnight placeholders were recovered only when the NLP table supplied one unique same-ticker, same-date timestamp.",
        ],
    }
    return report


def write_json_report(report: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_markdown_report(report: dict[str, object], path: Path) -> None:
    master = report["master"]
    nlp = report["nlp_scores"]
    cross = report["cross_file"]
    raw = report["raw_transcripts"]
    lines = [
        "# 第一阶段数据质量报告",
        "",
        "## 结论",
        "",
        f"现有主表共 {master['rows']:,} 行。按股票代码和美国东部会议时间生成事件键后，"
        f"得到 {master['event_groups']:,} 个事件组，其中 {master['conflict_groups']:,} 组存在数值冲突。"
        f"安全事件表保留 {master['safe_unique_events']:,} 个事件。",
        "",
        "冲突组没有被任意去重，全部进入人工复核表；无法恢复会议时间的记录也已从安全表移出。原始会议文本缺失时，不能判断冲突组中哪一行正确。",
        "",
        "## 主表",
        "",
        f"- 行数：{master['rows']:,}",
        f"- 股票数：{master['unique_tickers']:,}",
        f"- 修正尾部多余右括号的 ticker 行：{master['ticker_corrections']:,}",
        f"- 原字段成功解析：{master['source_timestamps_parsed']:,}",
        f"- 可用于事件键的会议时间：{master['parsed_timestamps']:,}",
        f"- 从 NLP 表唯一恢复的午夜占位时间：{master['timestamp_resolution'].get('recovered_from_nlp_scores', 0):,}",
        f"- 无法恢复、需人工复核的会议时间：{master['unparsed_timestamps']:,}",
        f"- 完全重复事件组：{master['exact_duplicate_groups']:,}",
        f"- 冲突事件组：{master['conflict_groups']:,}",
        f"- 冲突行数：{master['conflict_rows']:,}",
        f"- 安全事件数：{master['safe_unique_events']:,}",
        "",
        "## NLP 分数表",
        "",
        f"- 行数：{nlp['rows']:,}",
        f"- 股票数：{nlp['unique_tickers']:,}",
        f"- 修正尾部多余右括号的 ticker 行：{nlp['ticker_corrections']:,}",
        f"- 成功解析会议时间：{nlp['parsed_timestamps']:,}",
        f"- 未解析会议时间：{nlp['unparsed_timestamps']:,}",
        f"- 被标记为完全重复的行：{nlp['exact_duplicate_rows']:,}",
        f"- 重复事件组：{nlp['duplicate_event_groups']:,}",
        f"- 同一事件存在不同 Tone_Score 的组：{nlp['tone_conflict_groups']:,}",
        "",
        "## 跨文件一致性",
        "",
        f"- 两表重合事件键：{cross['overlapping_event_ids']:,}",
        f"- 从 NLP 表到主表的行数减少比例：{cross['row_loss_rate_from_nlp_to_master']:.2%}",
        "",
        "## 原始语料",
        "",
        f"- 是否存在：{'是' if raw['exists'] else '否'}",
        f"- 预期路径：`{raw['path']}`",
        "",
        "## 下一阶段入口条件",
        "",
        "1. 找回原始会议文本，验证 ticker、时间、季度和全文哈希。",
        "2. 重新生成 Prepared Remarks 与 Q&A 分段，而不是沿用不可复现的旧特征。",
        "3. 使用唯一事件键重新下载和合并市场数据。",
        "4. 重算 CAR、SUE、Size、VOL 和 Turnover 后，再开始模型实验。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

