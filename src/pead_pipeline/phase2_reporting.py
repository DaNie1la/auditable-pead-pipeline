from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_phase2_report(
    raw_path: Path,
    raw_hash: str,
    audited: pd.DataFrame,
    safe: pd.DataFrame,
    review: pd.DataFrame,
    corpus_path: Path,
) -> dict[str, object]:
    split_counts = {
        str(key): int(value)
        for key, value in audited["split_status"].value_counts().items()
    }
    event_counts = {
        str(key): int(value)
        for key, value in audited["event_status"].value_counts().items()
    }
    conflicts = audited[audited["event_status"].eq("conflict")]
    conflict_groups = conflicts.groupby("event_id").agg(
        transcript_versions=("transcript_sha256", "nunique"),
        exchange_versions=("exchange", "nunique"),
        quarter_versions=("quarter", "nunique"),
    )
    explicit_boundary_rows = sum(
        split_counts.get(status, 0)
        for status in [
            "split_complete",
            "prepared_text_empty",
            "qa_text_empty",
            "prepared_text_too_short",
            "qa_text_too_short",
        ]
    )
    basic_tickers = (
        audited["ticker"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )
    corrected_ticker_mask = basic_tickers.ne(audited["ticker_normalized"])
    return {
        "phase": 2,
        "source": {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "sha256": raw_hash,
            "rows": len(audited),
            "columns": ["date", "exchange", "q", "ticker", "transcript"],
        },
        "timestamps": {
            "parsed": int(audited["event_timestamp_et"].notna().sum()),
            "unparsed": int(audited["event_timestamp_et"].isna().sum()),
        },
        "tickers": {
            "corrected_rows": int(corrected_ticker_mask.sum()),
            "source_values_corrected": sorted(
                audited.loc[corrected_ticker_mask, "ticker"].astype(str).unique()
            ),
        },
        "splitting": {
            "status_counts": split_counts,
            "explicit_boundary_rows": explicit_boundary_rows,
            "explicit_boundary_rate": explicit_boundary_rows / len(audited),
            "complete_row_rate": split_counts.get("split_complete", 0)
            / len(audited),
        },
        "events": {
            "event_groups": int(audited["event_id"].nunique()),
            "status_row_counts": event_counts,
            "conflict_groups": int(conflicts["event_id"].nunique()),
            "transcript_conflict_groups": int(
                conflict_groups["transcript_versions"].gt(1).sum()
            ),
            "metadata_only_conflict_groups": int(
                (
                    conflict_groups["transcript_versions"].eq(1)
                    & (
                        conflict_groups["exchange_versions"].gt(1)
                        | conflict_groups["quarter_versions"].gt(1)
                    )
                ).sum()
            ),
            "safe_section_events": len(safe),
            "manual_review_rows": len(review),
        },
        "section_lengths": {
            "prepared_words_median": float(safe["prepared_word_count"].median()),
            "qa_words_median": float(safe["qa_word_count"].median()),
            "prepared_words_min": int(safe["prepared_word_count"].min()),
            "qa_words_min": int(safe["qa_word_count"].min()),
        },
        "output": {
            "path": str(corpus_path),
            "size_bytes": corpus_path.stat().st_size,
            "sha256": sha256_file(corpus_path),
            "rows": len(safe),
        },
        "decisions": [
            "Only explicit Prepared Remarks and Questions/Answers headings were accepted.",
            "Both sections must contain at least 20 words to enter the model-ready corpus.",
            "No missing section boundary was inferred from speaker names or conversational text.",
            "Event conflicts and incomplete splits were excluded from the section corpus.",
            "The source pickle was read but not modified.",
            "Unmatched trailing right parentheses were removed from ticker symbols while preserving the source value in the audit table.",
        ],
    }


def write_phase2_reports(
    report: dict[str, object], json_path: Path, markdown_path: Path
) -> None:
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source = report["source"]
    timestamps = report["timestamps"]
    tickers = report["tickers"]
    splitting = report["splitting"]
    events = report["events"]
    lengths = report["section_lengths"]
    output = report["output"]
    statuses = splitting["status_counts"]
    lines = [
        "# 第二阶段语料重建报告",
        "",
        "## 结论",
        "",
        f"原始语料包含 {source['rows']:,} 行。严格依照显式章节标题，"
        f"{splitting['explicit_boundary_rows']:,} 行找到完整章节边界"
        f"（{splitting['explicit_boundary_rate']:.2%}），其中 "
        f"{statuses.get('split_complete', 0):,} 行同时满足双章节至少 20 词。"
        f"去除重复事件并隔离冲突后，"
        f"生成 {events['safe_section_events']:,} 个可复现事件。",
        "",
        "未出现明确章节标题的文本没有被启发式猜分，全部保留在人工复核表。",
        "",
        "## 原始文件",
        "",
        f"- 路径：`{source['path']}`",
        f"- 大小：{source['size_bytes']:,} 字节",
        f"- SHA-256：`{source['sha256']}`",
        f"- 时间成功解析：{timestamps['parsed']:,}",
        f"- 时间未解析：{timestamps['unparsed']:,}",
        f"- 修正尾部多余右括号的 ticker 行：{tickers['corrected_rows']:,}",
        "",
        "## 分段质量",
        "",
        f"- 完整分段：{statuses.get('split_complete', 0):,}",
        f"- 缺少 Q&A 标题：{statuses.get('qa_marker_missing', 0):,}",
        f"- 缺少 Prepared 标题：{statuses.get('prepared_marker_missing', 0):,}",
        f"- 缺少页尾边界：{statuses.get('footer_marker_missing', 0):,}",
        f"- Q&A 正文为空：{statuses.get('qa_text_empty', 0):,}",
        f"- Q&A 少于 20 词：{statuses.get('qa_text_too_short', 0):,}",
        f"- Prepared 少于 20 词：{statuses.get('prepared_text_too_short', 0):,}",
        f"- Prepared 中位词数：{lengths['prepared_words_median']:,.0f}",
        f"- Q&A 中位词数：{lengths['qa_words_median']:,.0f}",
        "",
        "## 事件审计",
        "",
        f"- 唯一事件组：{events['event_groups']:,}",
        f"- 冲突事件组：{events['conflict_groups']:,}",
        f"- 其中文本版本冲突：{events['transcript_conflict_groups']:,}",
        f"- 仅元数据冲突：{events['metadata_only_conflict_groups']:,}",
        f"- 可用分段事件：{events['safe_section_events']:,}",
        f"- 需复核的原始行：{events['manual_review_rows']:,}",
        "",
        "## 输出",
        "",
        f"- 压缩语料：`{output['path']}`",
        f"- 输出行数：{output['rows']:,}",
        f"- 输出 SHA-256：`{output['sha256']}`",
        "",
        "## 第三阶段入口",
        "",
        "1. 按长文本窗口分别计算 Prepared Remarks 与 Q&A 的 FinBERT 分数。",
        "2. 保存模型版本、分词器版本、窗口长度、重叠率及聚合公式。",
        "3. 在文本指标冻结后，重新下载并构造事件期市场变量。",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
