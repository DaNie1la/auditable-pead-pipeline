# Auditable PEAD Prediction Pipeline

## Project summary

This project rebuilds an earnings-call-based Post-Earnings Announcement Drift (PEAD) workflow from raw transcripts and market histories. Its central contribution is not a new paper claim. It is an auditable data and modeling pipeline that identifies unreliable legacy variables, aligns features to event dates, limits temporal leakage, and reports negative as well as positive results. Current-vintage vendor histories cannot establish an immutable historical snapshot.

The pipeline processes 18,755 raw earnings-call records, produces 17,123 uniquely keyed and sectioned events, and yields 11,676 events with complete text, market, earnings-surprise, historical market-capitalization, and turnover features. Model specifications are selected on 2020–2021 validation folds. After a conservative CAR60 label embargo, 7,305 events train the final models and 2,804 events from 2022–2023 form the later test set.

## Why this project matters

Financial machine-learning results can look strong while being driven by misaligned timestamps, mislabeled variables, look-ahead bias, or a test set used during preprocessing. This project audits those risks. Private runs produce row-level records and machine-readable reports; the public repository shares aggregate audit JSON, deterministic hashes, 34 automated tests, and a synthetic integration demonstration.

The reconstruction found that several legacy market columns were not aligned to the earnings-call event. Among 9,993 comparable events, the correlation between the rebuilt and legacy CAR variables was only 0.006 for CAR5, 0.023 for CAR10, and 0.041 for CAR60. The legacy variable labeled `SUE_Proxy` was also not a standardized earnings surprise and was replaced with an analyst-based SUE measure filtered by historical record dates.

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
event-time-filtered historical EPS, market cap, and turnover
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
- Replaced mislabeled proxy variables with event-time-filtered historical EPS surprise, market capitalization, volatility, prior return, and share-based turnover. Current-vintage Yahoo data still carry revision and backfill risk.
- Built chronological model selection on earlier validation years, a conservative CAR60 label embargo, and train-only winsorization and standardization.
- Compared financial-only, text-augmented, and interaction feature sets across XGBoost, LightGBM, CatBoost, and an equal-weight ensemble.
- Added company-clustered standard errors, calendar-quarter fixed effects, Benjamini–Hochberg false-discovery-rate control, and common test-set permutation importance.
- Added 34 automated tests, a privacy-safe synthetic integration demonstration, and deterministic output verification.

## Results I would present

The most robust result is that standardized unexpected earnings remains the dominant predictor across all horizons. In the complete regression specification, a one-training-standard-deviation increase in SUE is associated with approximately 1.43, 1.55, and 2.18 percentage-point increases in CAR5, CAR10, and CAR60, respectively. These are associations rather than causal effects.

Text provides narrower incremental value. Presentation and Q&A tone both retain evidence for CAR5 after false-discovery-rate correction; only Q&A tone retains evidence for CAR10; neither tone measure retains evidence for CAR60. CatBoost was selected on earlier validation folds for CAR5 and CAR10; its later test mean monthly Rank ICs are 0.216 and 0.168. LightGBM was selected for CAR60 and achieves 0.095 on the later test set. The evidence does not support claiming universal ensemble superiority.

The interaction between SUE and tone is not statistically robust after multiple-testing correction and does not consistently improve out-of-sample performance. Reporting this negative result is part of the project's value: the final conclusions follow the rebuilt evidence rather than the desired narrative.

## Technical stack

- Python 3.12, pandas, NumPy and SciPy
- Hugging Face Transformers and PyTorch for fixed-revision FinBERT inference
- yfinance for cached daily market, earnings, and share histories
- statsmodels for clustered inference and multiple-testing control
- scikit-learn, XGBoost, LightGBM and CatBoost for predictive evaluation
- pytest for transformation, temporal-split, metric, and pipeline tests

## Reproduce the project

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the environment, restricted source-data placement, phase commands, expected row counts, and validation checks. Raw transcripts, cached vendor responses, model weights, legacy working data, and collaborative manuscript files are intentionally excluded from the public repository. Consequently, external readers can run the synthetic integration demo and inspect aggregate audit evidence, but cannot independently reproduce the full empirical estimates without licensed source data and the original provider snapshots.

## Scope and ownership

This is an independent reconstruction and audit extension of a collaborative research context. It should not be described as sole authorship of the original paper or its original ideas and results. The portfolio claim is limited to the rebuilt pipeline, audit decisions, validation framework, and evidence-based reinterpretation documented in this repository. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Known limitations

- Yahoo coverage is incomplete for delisted or otherwise unavailable tickers; missing events are not imputed as zero.
- Yahoo earnings and share histories were downloaded after the events and may contain later revisions or backfills; date filtering alone does not establish true historical snapshot availability.
- A reliable point-in-time industry classification is not currently available, so the rebuilt analysis does not claim to exclude financial firms or use industry fixed effects.
- Model selection uses earlier validation years; the final test set contains 14 months with at least 20 events, which limits precision of performance estimates.
- Portfolio sorts are diagnostics only. They do not model overlapping holdings, transaction costs, trading restrictions, or short availability and are not presented as investable backtests.

## Resume version in Chinese

**PEAD 多模态数据重建与模型审计个人项目**

- 重建 18,755 份美股业绩电话会的事件级数据流程，完成时间戳恢复、冲突隔离、Prepared Remarks/Q&A 严格分段和逐行质量审计，生成 17,123 个高置信文本事件。
- 使用固定版本 FinBERT 实现滑动窗口情绪推理，并重算事件时点对齐的 CAR、按历史记录日期过滤的 EPS 意外值、市值与换手率；明确披露 Yahoo 当前版本历史数据的修订风险。
- 用 2020–2021 历史验证折预选模型，并对 60 日收益标签设置 95 天缓冲；最终以 7,305 个事件训练、2,804 个后续事件测试，CatBoost 的 CAR5/CAR10 月度 Rank IC 分别为 0.216/0.168。
- 引入公司聚类标准误、季度固定效应、FDR 校正和统一置换重要性，用 34 项自动化测试、公开审计汇总及 SHA-256 哈希验证流程。

## Resume version in English

**Auditable Multimodal PEAD Pipeline | Independent Reconstruction and Model Audit**

- Rebuilt an event-level data pipeline for 18,755 U.S. earnings-call transcripts, resolving timestamps, isolating conflicts, and producing 17,123 high-confidence Prepared Remarks/Q&A events with row-level audit trails.
- Implemented fixed-revision FinBERT inference and reconstructed event-aligned CAR and date-filtered historical EPS, market capitalization, volatility, and share-based turnover; disclosed current-vintage Yahoo revision risk.
- Selected models on 2020–2021 validation folds with a 95-day CAR60 label embargo, then trained on 7,305 events and tested on 2,804 later events; validation-selected CatBoost achieved mean monthly Rank ICs of 0.216 for CAR5 and 0.168 for CAR10.
- Added ticker-clustered inference, calendar-quarter fixed effects, false-discovery-rate correction, common permutation importance, 34 automated tests, and public aggregate audit evidence.
