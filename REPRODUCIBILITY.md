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

Phase 3 requires the fixed FinBERT revision recorded in `reports/phase3_finbert_quality.json`. Phases 4 and 4B cache provider responses per ticker; missing provider data remain explicit and are never filled with zero.

## Expected checkpoints

| Checkpoint | Expected rows |
|---|---:|
| Raw transcript records | 18,755 |
| Safe sectioned transcript events | 17,123 |
| FinBERT event scores | 17,123 |
| Complete text and market events | 13,160 |
| Complete analysis events | 11,676 |
| Chronological training events | 8,872 |
| Strictly later test events | 2,804 |

The exact SHA-256 hashes for each generated artifact are recorded in the phase JSON reports. Provider changes may alter newly downloaded histories; when that happens, preserve the new cache, rerun all downstream phases, and document the resulting hash change rather than mixing old and new outputs.

## Leakage controls

- Event response windows begin according to the U.S. Eastern call session rather than calendar date alone.
- SUE uses only forecast errors preceding the current earnings event.
- Shares outstanding are taken as of the pre-event control date and are never backfilled from future observations.
- Feature winsorization and standardization parameters are fitted on 2017–2021 training data only.
- Training outcomes are clipped using training-period bounds; test outcomes remain unmodified.
- Model evaluation is performed only on events dated 2022-01-01 or later.

## Validation

The test suite covers event keys, transcript boundaries, FinBERT aggregation, event-window alignment, point-in-time share and SUE logic, chronological splitting, train-fitted transformations, Rank IC behavior, and monthly portfolio-sort ordering. A complete validated run currently reports 33 passing tests.
