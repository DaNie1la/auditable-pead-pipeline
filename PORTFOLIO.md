# Auditable PEAD Prediction Pipeline

## Project summary

This project rebuilds an earnings-call-based Post-Earnings Announcement Drift (PEAD) workflow from raw transcripts and market histories. Its central contribution is not a new paper claim. It is an auditable data and modeling pipeline that identifies unreliable legacy variables, reconstructs point-in-time features, prevents temporal leakage, and reports negative as well as positive results.

The pipeline processes 18,755 raw earnings-call records, produces 17,123 uniquely keyed and sectioned events, and yields 11,676 events with complete text, market, earnings-surprise, historical market-capitalization, and turnover features. The final predictive evaluation uses 2017–2021 as the training period and 2022–2023 as a strictly later test period.

## Why this project matters

Financial machine-learning results can look strong while being driven by misaligned timestamps, mislabeled variables, look-ahead bias, or a test set used during preprocessing. This project treats those risks as first-class engineering problems. Every stage produces row-level audit data, machine-readable quality reports, deterministic hashes, and automated tests.

The reconstruction found that several legacy market columns were not aligned to the earnings-call event. Among 9,993 comparable events, the correlation between the rebuilt and legacy CAR variables was only 0.006 for CAR5, 0.023 for CAR10, and 0.041 for CAR60. The legacy variable labeled `SUE_Proxy` was also not a standardized earnings surprise and was replaced with a point-in-time analyst-based SUE measure.

## What I built

```text
18,755 raw transcripts
        |
        v
event-key and conflict audit
        |
        v
17,123 safe Prepared Remarks + Q&A events
        |
        v
fixed-revision FinBERT inference with overlapping token windows
        |
        v
event-time-aware CAR and pre-event market controls
        |
        v
point-in-time analyst SUE, market cap, and true turnover
        |
        v
11,676 complete analysis events
        |
        v
cluster-robust inference + chronological out-of-sample prediction
```

Key implementation work includes:

- Created stable event identifiers using ticker and U.S. Eastern event time, while isolating duplicate and conflicting observations.
- Split transcripts only at explicit Prepared Remarks and Q&A headings; incomplete or ambiguous records remain visible in the audit outputs rather than being silently inferred.
- Scored complete sections with a fixed ProsusAI/FinBERT revision using 512-token windows, 64-token overlap, and new-token-weighted probability aggregation.
- Rebuilt CAR5, CAR10, and CAR60 from adjusted prices with SPY as the benchmark and event-session-aware response windows.
- Replaced mislabeled proxy variables with point-in-time SUE, historical market capitalization, volatility, prior return, and share-based turnover.
- Built leakage-safe preprocessing: winsorization and standardization parameters are estimated only from the training period.
- Compared financial-only, text-augmented, and interaction feature sets across XGBoost, LightGBM, CatBoost, and an equal-weight ensemble.
- Added company-clustered standard errors, calendar-quarter fixed effects, Benjamini–Hochberg false-discovery-rate control, and common test-set permutation importance.
- Added 33 automated tests and deterministic output verification.

## Results I would present

The most robust result is that point-in-time standardized unexpected earnings remains the dominant predictor across all horizons. In the complete regression specification, a one-training-standard-deviation increase in SUE is associated with approximately 1.41, 1.51, and 2.12 percentage-point increases in CAR5, CAR10, and CAR60, respectively. These are associations rather than causal effects.

Text provides narrower incremental value. Presentation and Q&A tone both retain evidence for CAR5 after false-discovery-rate correction; only Q&A tone retains evidence for CAR10; neither tone measure retains evidence for CAR60. In the chronological test set, CatBoost records the highest mean monthly Rank IC for CAR5 (0.217) and CAR10 (0.166). The ensemble is only marginally higher for CAR60 (0.143 versus 0.142 for CatBoost), so the evidence does not support claiming universal ensemble superiority.

The interaction between SUE and tone is not statistically robust after multiple-testing correction and does not consistently improve out-of-sample performance. Reporting this negative result is part of the project's value: the final conclusions follow the rebuilt evidence rather than the desired narrative.

## Technical stack

- Python 3.12, pandas, NumPy and SciPy
- Hugging Face Transformers and PyTorch for fixed-revision FinBERT inference
- yfinance for cached daily market, earnings, and share histories
- statsmodels for clustered inference and multiple-testing control
- scikit-learn, XGBoost, LightGBM and CatBoost for predictive evaluation
- pytest for transformation, temporal-split, metric, and pipeline tests

## Reproduce the project

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the environment, required source-data placement, phase commands, expected row counts, and validation checks. Raw transcripts, cached vendor responses, model weights, legacy working data, and collaborative manuscript files are intentionally excluded from the public repository.

## Scope and ownership

This is an independent reconstruction and audit extension of a collaborative research context. It should not be described as sole authorship of the original paper or its original ideas and results. The portfolio claim is limited to the rebuilt pipeline, audit decisions, validation framework, and evidence-based reinterpretation documented in this repository. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Known limitations

- Yahoo coverage is incomplete for delisted or otherwise unavailable tickers; missing events are not imputed as zero.
- A reliable point-in-time industry classification is not currently available, so the rebuilt analysis does not claim to exclude financial firms or use industry fixed effects.
- The test set contains 14 months with at least 20 events, which limits confidence in fine model rankings.
- Portfolio sorts are diagnostics only. They do not model overlapping holdings, transaction costs, trading restrictions, or short availability and are not presented as investable backtests.

## Resume version in Chinese

**PEAD 多模态数据重建与模型审计个人项目**

- 重建 18,755 份美股业绩电话会的事件级数据流程，完成时间戳恢复、冲突隔离、Prepared Remarks/Q&A 严格分段和逐行质量审计，生成 17,123 个高置信文本事件。
- 使用固定版本 FinBERT 实现滑动窗口情绪推理，并重算事件时点对齐的 CAR、点时 SUE、历史市值与真实换手率，识别出旧数据中变量错标和收益窗口错位。
- 构建无时间泄漏的 2017–2021 训练集与 2022–2023 测试集，比较 XGBoost、LightGBM、CatBoost 与异构集成模型；CatBoost 在 CAR5 与 CAR10 上取得 0.217 和 0.166 的月度 Rank IC。
- 引入公司聚类标准误、季度固定效应、FDR 校正和统一置换重要性，用 33 项自动化测试与 SHA-256 哈希验证流程可复现性。

## Resume version in English

**Auditable Multimodal PEAD Pipeline | Independent Reconstruction and Model Audit**

- Rebuilt an event-level data pipeline for 18,755 U.S. earnings-call transcripts, resolving timestamps, isolating conflicts, and producing 17,123 high-confidence Prepared Remarks/Q&A events with row-level audit trails.
- Implemented fixed-revision FinBERT inference with overlapping token windows and reconstructed event-aligned CAR, point-in-time analyst SUE, historical market capitalization, volatility, and share-based turnover; identified mislabeled and misaligned variables in the legacy dataset.
- Designed a leakage-safe chronological evaluation with 8,872 training events and 2,804 later test events, benchmarking XGBoost, LightGBM, CatBoost, and a heterogeneous ensemble; achieved mean monthly Rank ICs of 0.217 for CAR5 and 0.166 for CAR10 with CatBoost.
- Added ticker-clustered inference, calendar-quarter fixed effects, false-discovery-rate correction, common permutation importance, 33 automated tests, and deterministic SHA-256 output verification.
