"""Create aggregate-only portfolio charts from private phase outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#24597a"
GOLD = "#c79b3b"
GRAY = "#9aa7ae"


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    stages = ["Raw calls", "Sectioned events", "Market complete", "Analysis ready"]
    counts = [18755, 17123, 13160, 11676]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    bars = ax.barh(stages[::-1], counts[::-1], color=[GOLD, BLUE, BLUE, BLUE])
    ax.set_xlim(0, 21000)
    ax.set_xlabel("Events")
    ax.set_title("Sample coverage by audit stage", loc="left", weight="bold")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 250, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.output_dir / "sample_flow.png")

    metrics = pd.read_csv(args.processed_dir / "phase5_model_metrics.csv")
    metrics = metrics.loc[metrics["model"].ne("HistoricalMean")]
    targets = ["CAR_5", "CAR_10", "CAR_60"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, target in zip(axes, targets):
        group = metrics.loc[metrics["target"].eq(target)].sort_values("monthly_rank_ic_mean")
        colors = [GOLD if selected else GRAY for selected in group["selected_on_validation"]]
        ax.barh(group["model"], group["monthly_rank_ic_mean"], color=colors)
        ax.set_title(target)
        ax.axvline(0, color="#444444", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Final 2022–2023 monthly Rank IC", x=0.02, ha="left", weight="bold")
    fig.supxlabel("Gold = model selected on earlier validation years; 14 test months")
    fig.tight_layout()
    save(fig, args.output_dir / "test_rank_ic.png")

    importance = pd.read_csv(args.processed_dir / "phase5_feature_importance.csv")
    importance = importance.loc[importance["model"].eq("TIDE_consensus_permutation")]
    names = {
        "Z_SUE_Analyst_8Q": "Earnings surprise", "Z_Size_LogMarketCap": "Market cap",
        "Z_VOL_60": "Volatility", "Z_LogTurnover_60": "Turnover",
        "Z_PreRet_60": "Prior return", "Z_Tone_Pres": "Prepared tone",
        "Z_Tone_QA": "Q&A tone", "SUE_x_Tone_Pres": "SUE × prepared",
        "SUE_x_Tone_QA": "SUE × Q&A",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, target in zip(axes, targets):
        group = importance.loc[importance["target"].eq(target)].copy()
        group["label"] = group["feature"].map(names)
        group = group.sort_values("normalized_importance")
        ax.barh(group["label"], group["normalized_importance"], color=BLUE)
        ax.set_title(target)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory test-set permutation importance", x=0.02, ha="left", weight="bold")
    fig.supxlabel("Share of positive RMSE increase; associations, not causal effects")
    fig.tight_layout()
    save(fig, args.output_dir / "permutation_importance.png")
    print(f"public_figures_complete count=3 output={args.output_dir}")


if __name__ == "__main__":
    main()
