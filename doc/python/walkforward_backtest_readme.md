# Walk-Forward Backtest Run Guide

## Purpose

This guide documents the production-style walk-forward backtest flow now supported by the project.

The flow is:

1. Build a split schedule with `python/walkforward_splits.py`
2. Run split-by-split backtest accumulation with `python/run_walkforward_backtest.py`
3. Inspect accumulated runs with `python/check_walkforward_runs.py`

The existing backtest scripts are reused as-is:

- `python/build_backtest_predictions.py`
- `python/build_backtest_ranking.py`
- `python/build_backtest_outcome.py`

Each split creates one `research.dim_model_run` row with:

- `run_type = walkforward_backtest`
- split metadata in `config_json`
- linked rows in:
  - `research.prediction_history`
  - `research.ranking_history`
  - `research.backtest_outcome`

## Current Data Range Reality Check

Current local CSV data range:

- `features.csv`: `2023-03-14` to `2026-03-13`
- `labels.csv`: `2023-03-14` to `2026-01-26`

With the requested split rule:

- `train_window_months = 24`
- `predict_window_months = 1`
- `step_months = 1`

the current local data produces:

- `12` splits from `features.csv` range
- `11` mature-or-near-mature splits if limited by current `labels.csv` end date

So the code supports the requested monthly walk-forward structure, but the current local data window does **not** yet allow `20+` non-overlapping monthly splits.

To reach `20+` splits with the same `24/1/1` rule, the data window must be longer.

For example, with the current start date `2023-03-14`, a full `20` monthly splits requires data through at least:

- `2026-11-13`

## Split Generation

Build the monthly split schedule:

```powershell
python python\walkforward_splits.py ^
  --data-start-date 2023-03-14 ^
  --data-end-date 2026-03-13 ^
  --train-window-months 24 ^
  --predict-window-months 1 ^
  --step-months 1 ^
  --out-csv outputs\walkforward_splits_monthly.csv
```

This writes:

- `outputs/walkforward_splits_monthly.csv`

Required columns:

- `split_id`
- `train_start`
- `train_end`
- `predict_start`
- `predict_end`

## Full Accumulation Run

Run all remaining monthly splits in one invocation:

```powershell
python python\run_walkforward_backtest.py ^
  --splits-csv outputs\walkforward_splits_monthly.csv ^
  --model-pkl data\model.pkl ^
  --model-version wf_monthly_validation ^
  --horizon-days 60 ^
  --top-n 20 ^
  --rebalance-freq monthly ^
  --universe-version universe_20260313 ^
  --score-formula-version ranking_builder_v1 ^
  --score-weights-json "{}" ^
  --summary-prefix outputs\walkforward_monthly_summary ^
  --summary-min-runs 20
```

Behavior:

- Reads the split CSV
- Creates one `research.dim_model_run` row per split
- Logs `split_id` and `run_id` for each stage
- Runs, in order:
  1. `build_backtest_predictions.py`
  2. `build_backtest_ranking.py`
  3. `build_backtest_outcome.py`
- Writes final accumulation reports:
  - `outputs\walkforward_monthly_summary.csv`
  - `outputs\walkforward_monthly_summary.md`

## Restart From a Failed Split

Resume from a specific split:

```powershell
python python\run_walkforward_backtest.py ^
  --splits-csv outputs\walkforward_splits_monthly.csv ^
  --model-pkl data\model.pkl ^
  --start-split-id 6 ^
  --model-version wf_monthly_validation ^
  --horizon-days 60 ^
  --top-n 20 ^
  --rebalance-freq monthly ^
  --universe-version universe_20260313 ^
  --score-formula-version ranking_builder_v1 ^
  --score-weights-json "{}" ^
  --summary-prefix outputs\walkforward_monthly_summary_part2 ^
  --summary-min-runs 20
```

Limit a single batch if needed:

```powershell
python python\run_walkforward_backtest.py ^
  --splits-csv outputs\walkforward_splits_monthly.csv ^
  --model-pkl data\model.pkl ^
  --start-split-id 6 ^
  --max-splits 5 ^
  --model-version wf_monthly_validation ^
  --horizon-days 60 ^
  --top-n 20 ^
  --rebalance-freq monthly ^
  --universe-version universe_20260313 ^
  --score-formula-version ranking_builder_v1 ^
  --score-weights-json "{}" ^
  --summary-prefix outputs\walkforward_monthly_summary_part2 ^
  --summary-min-runs 20
```

## Run Health Check

Inspect accumulated walk-forward runs:

```powershell
python python\check_walkforward_runs.py ^
  --min-runs 20 ^
  --out-csv outputs\walkforward_run_check.csv ^
  --out-md outputs\walkforward_run_check.md
```

This report shows:

- run metadata parsed from `config_json`
- prediction/ranking/outcome row counts
- comparison groups based on model and score settings
- the primary comparison group
- warning flags for:
  - empty runs
  - non-primary-group runs
  - outcome not mature yet

Outcome maturity states:

- `OUTCOME_READY`
- `OUTCOME_NOT_MATURE`
- `OUTCOME_PARTIAL`
- `OUTCOME_EMPTY`

## Files and Roles

- [`python/walkforward_splits.py`](/d:/ai/Lee_trader/python/walkforward_splits.py)
  - builds the split schedule
- [`python/run_walkforward_backtest.py`](/d:/ai/Lee_trader/python/run_walkforward_backtest.py)
  - executes split-by-split accumulation and writes summary reports
- [`python/check_walkforward_runs.py`](/d:/ai/Lee_trader/python/check_walkforward_runs.py)
  - validates accumulated run health and comparison-group consistency
