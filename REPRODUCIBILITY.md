# Reproducibility Guide

## Environment

The validated local environment uses Python 3.12. Install dependencies phase by phase so the exact model and analysis versions remain visible:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-phase3.txt
.venv\Scripts\python.exe -m pip install -r requirements-phase4.txt
.venv\Scripts\python.exe -m pip install -r requirements-phase5.txt
```

Place the source transcript pickle at `data/raw/motley-fool-data.pkl`. Phase 1's legacy audit additionally expects `private/legacy/master_dataset_multi_window.csv` and `private/legacy/thesis_nlp_scores_new.csv`; these inherited comparison files are not distributed publicly. The public repository also excludes cached Yahoo responses, downloaded model weights, and generated event-level data.

## Run order

```powershell
.venv\Scripts\python.exe scripts\run_phase1.py
.venv\Scripts\python.exe scripts\run_phase2.py
.venv\Scripts\python.exe scripts\run_phase3.py
.venv\Scripts\python.exe scripts\run_phase4.py
.venv\Scripts\python.exe scripts\run_phase4_fundamentals.py
.venv\Scripts\python.exe scripts\run_phase5.py
.venv\Scripts\python.exe -m pytest -q
```

Phase 3 uses the fixed FinBERT revision recorded in `reports/phase3_finbert_quality.md`. Phases 4 and 4B cache provider responses per ticker; missing provider data remain explicit and are never filled with zero. The source data and private JSON reports are not included in this public repository.

## Expected checkpoints

| Checkpoint | Expected rows |
|---|---:|
| Raw transcript records | 18,755 |
| Safe sectioned transcript events | 17,123 |
| FinBERT event scores | 17,123 |
| Complete text and market events | 13,160 |
| Complete analysis events | 11,676 |
| Prediction training events after CAR60 label embargo | 7,305 |
| Events excluded near final test boundary | 1,567 |
| Strictly later test events | 2,804 |

Aggregate counts, configurations and SHA-256 hashes are published in `reports/public_audit_summary.json`. Private phase JSON files also contain local paths and remain out of the public repository. Provider changes may alter newly downloaded histories; when that happens, preserve the new cache, rerun all downstream phases, and document the resulting hash change rather than mixing old and new outputs. A third party without the same source data and provider snapshots cannot verify identical research outputs.

## Leakage controls

- Event response windows begin according to the U.S. Eastern call session rather than calendar date alone.
- SUE uses only forecast errors dated before the current earnings event in the downloaded historical series.
- Shares outstanding are taken as of the pre-event control date and are never backfilled from future observations in that series.
- Yahoo histories are fetched in the present, so provider revision or backfill risk remains. This is event-time filtering, not proof of an immutable point-in-time vendor snapshot.
- Model specifications are selected using 2020 and 2021 validation folds. A 95-calendar-day embargo removes events whose CAR60 labels could cross a fold boundary.
- Final preprocessing is fitted on the 7,305 eligible training events, ending 2021-09-28. Training outcomes are clipped using training-period bounds; test outcomes remain unmodified.
- The 2022–2023 set is held out for final reporting. Candidate test metrics are retained for audit but do not determine the selected model.

## Validation

The test suite covers event keys, transcript boundaries, FinBERT aggregation, event-window alignment, historical share and SUE filtering, chronological splitting, label embargo, train-fitted transformations, Rank IC behavior, and monthly portfolio-sort ordering. A complete validated run currently reports 34 passing tests.

Run the privacy-safe integration demonstration with `python scripts/run_synthetic_demo.py`. It uses invented events and fixture sentiment probabilities rather than FinBERT weights; it checks stage integration and is also run in CI. It is not a substitute for rerunning the restricted research dataset.
