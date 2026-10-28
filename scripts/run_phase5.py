from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import statsmodels.api as sm
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from statsmodels.stats.multitest import multipletests
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pead_pipeline.modeling import (  # noqa: E402
    FINANCIAL_FEATURES,
    INTERACTION_FEATURES,
    TARGETS,
    TEXT_FEATURES,
    ModelingConfig,
    correlation_table,
    descriptive_statistics,
    model_feature_sets,
    monthly_long_short_spreads,
    prediction_metrics,
    prepare_modeling_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe Phase 5 PEAD models.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "pead_analysis_ready.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase5_modeling_quality.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase5_modeling_quality.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_estimators(config: ModelingConfig) -> dict[str, object]:
    common = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "subsample": config.subsample,
        "random_state": config.random_seed,
    }
    return {
        "XGBoost": XGBRegressor(
            **common,
            colsample_bytree=config.colsample_bytree,
            objective="reg:squarederror",
            n_jobs=2,
        ),
        "LightGBM": LGBMRegressor(
            **common,
            colsample_bytree=config.colsample_bytree,
            objective="regression",
            verbosity=-1,
            n_jobs=2,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=config.n_estimators,
            learning_rate=config.learning_rate,
            depth=config.max_depth,
            subsample=config.subsample,
            random_seed=config.random_seed,
            loss_function="RMSE",
            verbose=False,
            thread_count=2,
            allow_writing_files=False,
        ),
    }


def clip_training_target(values: pd.Series, lower: float, upper: float) -> pd.Series:
    bounds = values.quantile([lower, upper])
    return values.clip(lower=float(bounds.iloc[0]), upper=float(bounds.iloc[1]))


def permutation_rmse_importance(
    estimator: object,
    features: pd.DataFrame,
    actual: pd.Series,
    random_seed: int,
    repeats: int = 10,
) -> pd.DataFrame:
    baseline = float(
        np.sqrt(np.mean(np.square(actual.to_numpy() - estimator.predict(features))))
    )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | str]] = []
    for column in features.columns:
        deltas: list[float] = []
        for _ in range(repeats):
            permuted = features.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            predicted = np.asarray(estimator.predict(permuted), dtype=float)
            permuted_rmse = float(
                np.sqrt(np.mean(np.square(actual.to_numpy() - predicted)))
            )
            deltas.append(permuted_rmse - baseline)
        rows.append(
            {
                "feature": column,
                "raw_importance": float(np.mean(deltas)),
                "importance_sd": float(np.std(deltas, ddof=1)),
            }
        )
    result = pd.DataFrame(rows)
    positive = result["raw_importance"].clip(lower=0)
    result["normalized_importance"] = (
        positive / positive.sum() if positive.sum() > 0 else 0.0
    )
    return result


def fit_clustered_ols(
    frame: pd.DataFrame,
    target: str,
    specification: str,
    features: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    quarter_dummies = pd.get_dummies(
        frame["event_quarter"], prefix="quarter", drop_first=True, dtype=float
    )
    design = pd.concat([frame[features].astype(float), quarter_dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(frame[target].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": frame["ticker"]}
    )
    confidence = model.conf_int(alpha=0.05)
    rows: list[dict[str, object]] = []
    for feature in features:
        rows.append(
            {
                "target": target,
                "specification": specification,
                "term": feature,
                "coefficient": float(model.params[feature]),
                "std_error_cluster_ticker": float(model.bse[feature]),
                "t_value": float(model.tvalues[feature]),
                "p_value": float(model.pvalues[feature]),
                "ci95_lower": float(confidence.loc[feature, 0]),
                "ci95_upper": float(confidence.loc[feature, 1]),
                "n": int(model.nobs),
                "ticker_clusters": int(frame["ticker"].nunique()),
                "quarter_fixed_effects": True,
            }
        )
    summary = {
        "target": target,
        "specification": specification,
        "n": int(model.nobs),
        "ticker_clusters": int(frame["ticker"].nunique()),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    config = ModelingConfig()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, low_memory=False)
    frame, transformer = prepare_modeling_frame(raw, config)
    train = frame[frame["sample_split"].eq("train")].copy()
    test = frame[frame["sample_split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("chronological split produced an empty train or test sample")

    paths = {
        "modeling_sample": args.processed_dir / "phase5_modeling_sample.csv",
        "transform_parameters": args.processed_dir / "phase5_transform_parameters.csv",
        "target_clip_bounds": args.processed_dir / "phase5_target_clip_bounds.csv",
        "descriptive": args.processed_dir / "phase5_descriptive_statistics.csv",
        "correlations": args.processed_dir / "phase5_correlations.csv",
        "ols_coefficients": args.processed_dir / "phase5_ols_coefficients.csv",
        "ols_models": args.processed_dir / "phase5_ols_models.csv",
        "metrics": args.processed_dir / "phase5_model_metrics.csv",
        "predictions": args.processed_dir / "phase5_test_predictions.csv",
        "importance": args.processed_dir / "phase5_feature_importance.csv",
        "portfolio_sorts": args.processed_dir / "phase5_portfolio_sorts.csv",
    }

    frame.to_csv(paths["modeling_sample"], index=False)
    transformer.export_parameters().to_csv(paths["transform_parameters"], index=False)
    target_bound_rows = []
    for target in TARGETS:
        bounds = train[target].quantile([config.winsor_lower, config.winsor_upper])
        target_bound_rows.append(
            {
                "target": target,
                "train_lower_quantile": config.winsor_lower,
                "train_lower_bound": float(bounds.iloc[0]),
                "train_upper_quantile": config.winsor_upper,
                "train_upper_bound": float(bounds.iloc[1]),
            }
        )
    pd.DataFrame(target_bound_rows).to_csv(paths["target_clip_bounds"], index=False)
    descriptive_columns = TARGETS + FINANCIAL_FEATURES + TEXT_FEATURES
    descriptive_statistics(raw, descriptive_columns).to_csv(
        paths["descriptive"], index=False
    )
    correlation_table(raw, descriptive_columns).to_csv(paths["correlations"], index=False)

    feature_sets = model_feature_sets()
    ols_rows: list[dict[str, object]] = []
    ols_summaries: list[dict[str, object]] = []
    for target in TARGETS:
        for specification, features in feature_sets.items():
            rows, summary = fit_clustered_ols(frame, target, specification, features)
            ols_rows.extend(rows)
            ols_summaries.append(summary)
    ols = pd.DataFrame(ols_rows)
    ols["q_value_fdr_bh"] = multipletests(ols["p_value"], method="fdr_bh")[1]
    ols.to_csv(paths["ols_coefficients"], index=False)
    pd.DataFrame(ols_summaries).to_csv(paths["ols_models"], index=False)

    predictions = test[
        ["event_id", "ticker", "event_timestamp_et", "event_date", "event_month"]
        + TARGETS
    ].copy()
    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    fitted_interaction_predictions: dict[tuple[str, str], np.ndarray] = {}
    fitted_interaction_importance: dict[tuple[str, str], np.ndarray] = {}
    fitted_interaction_estimators: dict[tuple[str, str], object] = {}
    permutation_importance: dict[tuple[str, str], pd.DataFrame] = {}

    for target in TARGETS:
        y_train = clip_training_target(
            train[target].astype(float), config.winsor_lower, config.winsor_upper
        )
        historical_mean = np.repeat(float(y_train.mean()), len(test))
        baseline_column = f"pred_{target}_HistoricalMean"
        predictions[baseline_column] = historical_mean
        metric_rows.append(
            {
                "target": target,
                "model": "HistoricalMean",
                "feature_specification": "none",
                **prediction_metrics(
                    test[target], pd.Series(historical_mean, index=test.index), test["event_month"]
                ),
            }
        )

        model_jobs = [
            ("XGBoost", "financial"),
            ("XGBoost", "additive"),
            ("XGBoost", "interaction"),
            ("LightGBM", "interaction"),
            ("CatBoost", "interaction"),
        ]
        for model_name, specification in model_jobs:
            estimator = build_estimators(config)[model_name]
            features = feature_sets[specification]
            estimator.fit(train[features], y_train)
            predicted = np.asarray(estimator.predict(test[features]), dtype=float)
            output_model_name = (
                f"{model_name}_{specification}"
                if model_name == "XGBoost"
                else model_name
            )
            prediction_column = f"pred_{target}_{output_model_name}"
            predictions[prediction_column] = predicted
            metric_rows.append(
                {
                    "target": target,
                    "model": output_model_name,
                    "feature_specification": specification,
                    **prediction_metrics(
                        test[target], pd.Series(predicted, index=test.index), test["event_month"]
                    ),
                }
            )
            raw_importance = np.asarray(estimator.feature_importances_, dtype=float)
            normalized = raw_importance / raw_importance.sum()
            for feature, value in zip(features, normalized):
                importance_rows.append(
                    {
                        "target": target,
                        "model": output_model_name,
                        "feature_specification": specification,
                        "feature": feature,
                        "importance_method": "native_normalized",
                        "raw_importance": float(
                            raw_importance[features.index(feature)]
                        ),
                        "importance_sd": float("nan"),
                        "normalized_importance": float(value),
                    }
                )
            if specification == "interaction":
                fitted_interaction_predictions[(target, model_name)] = predicted
                fitted_interaction_importance[(target, model_name)] = normalized
                fitted_interaction_estimators[(target, model_name)] = estimator
                permutation = permutation_rmse_importance(
                    estimator,
                    test[features],
                    test[target].astype(float),
                    random_seed=config.random_seed,
                )
                permutation_importance[(target, model_name)] = permutation
                for row in permutation.to_dict("records"):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": output_model_name,
                            "feature_specification": specification,
                            "feature": row["feature"],
                            "importance_method": "test_permutation_rmse_increase",
                            "raw_importance": row["raw_importance"],
                            "importance_sd": row["importance_sd"],
                            "normalized_importance": row["normalized_importance"],
                        }
                    )

        tide_predictions = np.mean(
            [
                fitted_interaction_predictions[(target, model_name)]
                for model_name in ["XGBoost", "LightGBM", "CatBoost"]
            ],
            axis=0,
        )
        predictions[f"pred_{target}_TIDE"] = tide_predictions
        metric_rows.append(
            {
                "target": target,
                "model": "TIDE",
                "feature_specification": "interaction",
                **prediction_metrics(
                    test[target], pd.Series(tide_predictions, index=test.index), test["event_month"]
                ),
            }
        )
        consensus = np.mean(
            [
                fitted_interaction_importance[(target, model_name)]
                for model_name in ["XGBoost", "LightGBM", "CatBoost"]
            ],
            axis=0,
        )
        consensus = consensus / consensus.sum()
        for feature, value in zip(feature_sets["interaction"], consensus):
            importance_rows.append(
                {
                    "target": target,
                    "model": "TIDE_consensus_native",
                    "feature_specification": "interaction",
                    "feature": feature,
                    "importance_method": "native_normalized_average",
                    "raw_importance": float("nan"),
                    "importance_sd": float("nan"),
                    "normalized_importance": float(value),
                }
            )
        permutation_consensus = pd.concat(
            [
                permutation_importance[(target, model_name)]
                .set_index("feature")
                .rename(columns={"raw_importance": model_name})[[model_name]]
                for model_name in ["XGBoost", "LightGBM", "CatBoost"]
            ],
            axis=1,
        )
        permutation_consensus["raw_importance"] = permutation_consensus.mean(axis=1)
        positive = permutation_consensus["raw_importance"].clip(lower=0)
        permutation_consensus["normalized_importance"] = (
            positive / positive.sum() if positive.sum() > 0 else 0.0
        )
        for feature, row in permutation_consensus.iterrows():
            importance_rows.append(
                {
                    "target": target,
                    "model": "TIDE_consensus_permutation",
                    "feature_specification": "interaction",
                    "feature": feature,
                    "importance_method": "test_permutation_rmse_increase_average",
                    "raw_importance": float(row["raw_importance"]),
                    "importance_sd": float("nan"),
                    "normalized_importance": float(row["normalized_importance"]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    pd.DataFrame(importance_rows).to_csv(paths["importance"], index=False)

    portfolio_rows: list[pd.DataFrame] = []
    for target in TARGETS:
        for model in ["XGBoost_interaction", "LightGBM", "CatBoost", "TIDE"]:
            prediction_column = f"pred_{target}_{model}"
            spread = monthly_long_short_spreads(predictions, target, prediction_column)
            spread.insert(0, "model", model)
            spread.insert(0, "target", target)
            portfolio_rows.append(spread)
    portfolio_sorts = pd.concat(portfolio_rows, ignore_index=True)
    portfolio_sorts.to_csv(paths["portfolio_sorts"], index=False)

    best_by_target = {}
    for target in TARGETS:
        target_metrics = metrics[
            metrics["target"].eq(target) & ~metrics["model"].eq("HistoricalMean")
        ]
        best = target_metrics.sort_values("monthly_rank_ic_mean", ascending=False).iloc[0]
        best_by_target[target] = {
            "model": str(best["model"]),
            "rmse": float(best["rmse"]),
            "directional_accuracy": float(best["directional_accuracy"]),
            "pooled_rank_ic": float(best["pooled_rank_ic"]),
            "monthly_rank_ic_mean": float(best["monthly_rank_ic_mean"]),
            "monthly_rank_ic_months": int(best["monthly_rank_ic_months"]),
        }

    output_manifest = {
        name: {
            "path": str(path),
            "rows": int(len(pd.read_csv(path, low_memory=False))),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    report = {
        "phase": "5",
        "status": "complete",
        "analysis_unit": "one earnings-call event",
        "config": asdict(config),
        "sample": {
            "all_events": int(len(frame)),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "train_start": str(train["event_date"].min().date()),
            "train_end": str(train["event_date"].max().date()),
            "test_start": str(test["event_date"].min().date()),
            "test_end": str(test["event_date"].max().date()),
            "unique_tickers": int(frame["ticker"].nunique()),
        },
        "inference": {
            "regression": "OLS with calendar-quarter fixed effects and ticker-clustered standard errors",
            "multiplicity": "Benjamini-Hochberg FDR across all reported non-FE coefficients",
            "claim_boundary": "associational, not causal",
        },
        "prediction": {
            "target_training": "training outcomes clipped at train-period 1st/99th percentiles; test outcomes remain raw",
            "rank_ic": "Spearman correlation; monthly mean uses test months with at least 20 events",
            "portfolio_sort": "event-month top-minus-bottom quintile diagnostic; arithmetic cumulative spread, no costs and not a tradeable overlapping-position backtest",
            "feature_importance": "common test-set permutation RMSE increase is the primary cross-model importance; native tree importances are retained only for audit",
            "best_by_target_monthly_rank_ic": best_by_target,
        },
        "software": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "outputs": output_manifest,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 第五阶段统计与样本外预测报告",
        "",
        "## 样本与识别边界",
        "",
        f"- 分析单位：单次业绩电话会事件。",
        f"- 训练期：{report['sample']['train_start']} 至 {report['sample']['train_end']}，{len(train):,} 条。",
        f"- 测试期：{report['sample']['test_start']} 至 {report['sample']['test_end']}，{len(test):,} 条。",
        "- 特征缩尾和标准化参数只由训练期估计；测试期目标保持原值。",
        "- 回归为关联性分析，不支持因果推断。",
        "",
        "## 样本外最佳月度 Rank IC",
        "",
    ]
    for target, values in best_by_target.items():
        lines.append(
            f"- {target}: {values['model']}，月度 Rank IC 均值 {values['monthly_rank_ic_mean']:.4f}，"
            f"汇总 Rank IC {values['pooled_rank_ic']:.4f}，RMSE {values['rmse']:.4f}，"
            f"方向准确率 {values['directional_accuracy']:.2%}。"
        )
    lines.extend(
        [
            "",
            "## 诚信边界",
            "",
            "组合排序只是按事件月计算的多空五分位诊断，未建模持仓重叠、交易成本、停牌或可做空性，因此不应解释为可实现的策略收益。",
            "",
            "## 输出文件",
            "",
        ]
    )
    for name, metadata in output_manifest.items():
        lines.append(
            f"- `{name}`: {metadata['rows']:,} 行，SHA-256 `{metadata['sha256']}`"
        )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"phase5_complete train={len(train)} test={len(test)} outputs={len(output_manifest)}"
    )


if __name__ == "__main__":
    main()
