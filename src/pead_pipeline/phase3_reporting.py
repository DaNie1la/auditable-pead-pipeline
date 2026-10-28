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


def write_phase3_report(
    scores: pd.DataFrame,
    input_path: Path,
    input_hash: str,
    output_path: Path,
    config: dict[str, object],
    runtime: dict[str, object],
    json_path: Path,
    markdown_path: Path,
) -> None:
    report = {
        "phase": 3,
        "status": "complete",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "rows": len(scores),
        },
        "model": config,
        "runtime": runtime,
        "scores": {
            "rows": len(scores),
            "event_ids_unique": bool(scores["event_id"].is_unique),
            "tone_pres": {
                "min": float(scores["Tone_Pres"].min()),
                "mean": float(scores["Tone_Pres"].mean()),
                "max": float(scores["Tone_Pres"].max()),
            },
            "tone_qa": {
                "min": float(scores["Tone_QA"].min()),
                "mean": float(scores["Tone_QA"].mean()),
                "max": float(scores["Tone_QA"].max()),
            },
            "pres_windows": int(scores["Pres_Windows"].sum()),
            "qa_windows": int(scores["QA_Windows"].sum()),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(scores),
        },
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 第三阶段 FinBERT 文本指标报告",
        "",
        "## 结论",
        "",
        f"已对 {len(scores):,} 个唯一事件的 Prepared Remarks 与 Q&A 分别完成长文本 FinBERT 打分。",
        "",
        "## 模型与口径",
        "",
        f"- 模型：`{config['model_id']}`",
        f"- 模型提交：`{config['model_revision']}`",
        f"- 最大窗口：{config['max_length']} tokens",
        f"- 重叠：{config['stride']} tokens",
        f"- 聚合：`{config['aggregation']}`",
        f"- 精度：`{config['precision']}`",
        f"- 事件 batch：{config['event_batch_size']}",
        f"- 窗口 batch：{config['window_batch_size']}",
        f"- 配置哈希：`{config['config_hash']}`",
        "- 语调公式：`P(positive) - P(negative)`",
        "",
        "## 运行",
        "",
        f"- 设备：{runtime['device']}",
        f"- PyTorch：{runtime['torch_version']}",
        f"- Transformers：{runtime['transformers_version']}",
        f"- 本次运行耗时：{runtime['elapsed_seconds'] / 60:.1f} 分钟",
        f"- 断点续跑前已有：{runtime['resumed_rows']:,} 条",
        f"- 本次新增：{runtime['newly_scored']:,} 条",
        "",
        "## 输出",
        "",
        f"- 行数：{len(scores):,}",
        f"- Prepared 窗口：{report['scores']['pres_windows']:,}",
        f"- Q&A 窗口：{report['scores']['qa_windows']:,}",
        f"- 文件：`{output_path}`",
        f"- SHA-256：`{report['output']['sha256']}`",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
