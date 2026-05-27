# AI Operations

## Purpose

이 문서는 AI 선별/자동매매 모듈의 기본 운영 명령과 점검 순서를 정리합니다.

## Main Commands

### Pipeline

```powershell
docker compose run --rm python-pipeline python python/run_pipeline.py
```

### Operational Refresh

```powershell
docker compose run --rm scheduler-auto-buy python python/run_operational_refresh.py
```

### Live Auto Trade Cycle

```powershell
docker compose run --rm scheduler-auto-buy python python/run_live_auto_trade_cycle.py
```

### Runtime Asset Validation

```powershell
docker compose run --rm scheduler-auto-buy python python/validate_runtime_assets.py --command-set auto_buy --strict
```

### Web Sync

```powershell
docker compose run --rm scheduler-live-account-sync python python/sync_web_display_data.py
```

## Key Outputs

- `data/predictions.csv`
- `data/ranking_final.csv`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`

## Alerts

- `python/score_kpi_monitor.py` sends alerts for walkforward rejection, low top20 score, and zero `BUY_ALLOWED` recommendations.
- `python/run_live_auto_trade_cycle.py` sends critical alerts for order submission failure and fill sync failure.
- Policy reference: `doc/alert_policy.md`
- If `SLACK_WEBHOOK_URL` is not configured, alerts fall back to console/log output without stopping operations.

## US Stock Operations (미운영)

US 주식 관련 수집·피처·랭킹 스크립트(`python/us/`)는 현재 미운영 상태입니다.
스케줄러에서 제외되어 있으며, 재개 시 별도 설계 검토가 필요합니다.
아래 명령은 수동 실행 참고용으로만 보관합니다.

---

## Project C Phase 2-2: US Financial Collector

Current status:

- A standalone `yfinance`-based US financial collector is implemented.
- It is not attached to any production scheduler.
- It must be run manually.

Operational rules:

- `US_FINANCIAL_COLLECT_ENABLED` must remain disabled by default.
- US financial collection failure must not affect Korean AI or RULE auto-trading operations.
- Do not attach the future collector to `run_pipeline.py`, `run_live_auto_trade_cycle.py`, or `run_daily_scheduler.py` in this phase.

Manual execution:

```powershell
python -m python.us.collect_us_financials_yfinance
python -m python.us.collect_us_financials_yfinance --universe NASDAQ100 --limit 10
python -m python.us.collect_us_financials_yfinance --ticker AAPL --ticker MSFT
```

Planned storage:

- `raw.us_stock_financial_statement`
- `raw.us_stock_financial_metric`

Write policy:

- upsert by `ticker + period_type + fiscal_date + source`
- nullable metrics allowed because `yfinance` coverage is inconsistent across tickers and periods

Failure checks:

1. confirm `US_FINANCIAL_COLLECT_ENABLED`
2. confirm `US_FINANCIAL_SOURCE=yfinance`
3. confirm Phase 2-1 financial schema migration is applied in the target DB
4. confirm `market.us_stock_universe` has active US tickers
5. inspect ticker-level retry/failure logs
6. confirm rows in `raw.us_stock_financial_statement` and `raw.us_stock_financial_metric`

Notes:

- `yfinance` may omit fields by ticker or by period.
- Missing fields should not be treated as batch failure.
- Failed tickers are isolated unless `US_FINANCIAL_FAIL_FAST=1`.

## Project C Phase 2-3: US Financial Feature Builder

Current status:

- A standalone US financial feature builder is implemented.
- It reads raw financial tables and writes a separate financial feature table.
- It is not attached to any production scheduler.
- It must be run manually.

Manual execution:

```powershell
python -m python.us.build_us_financial_features
python -m python.us.build_us_financial_features --universe NASDAQ100 --limit 10
python -m python.us.build_us_financial_features --ticker AAPL --ticker MSFT
```

Planned storage:

- source: `raw.us_stock_financial_statement`
- source: `raw.us_stock_financial_metric`
- target: `feature.us_stock_financial_feature`

Write policy:

- upsert by `ticker + period_type + fiscal_date + source`
- daily price features and financial features remain separated
- nullable feature values are allowed when raw denominators are missing or zero

Failure checks:

1. confirm `US_FINANCIAL_FEATURE_BUILD_ENABLED`
2. confirm `US_FINANCIAL_FEATURE_WRITE_MODE=upsert`
3. confirm Phase 2-3 financial feature migration exists in the target DB
4. confirm `market.us_stock_universe` has active US tickers
5. confirm `raw.us_stock_financial_statement` and `raw.us_stock_financial_metric` have source rows
6. inspect `null_ratio_summary`, `duplicate_key_count`, and failed ticker logs

Notes:

- `feature.us_stock_financial_feature` is not used by Korean AI, Korean RULE, or live order flows in this phase.
- Financial growth and margin features may be null when the previous fiscal period is unavailable.
- `info`-derived valuation fields from `yfinance` are snapshot-style and may not perfectly align with `fiscal_date`.

## Project C Phase 2-4: Relative Strength Builder

Current status:

- A standalone relative strength builder is implemented.
- It reads US daily price rows and writes a separate relative strength feature table.
- It is not attached to any production scheduler.
- It must be run manually.

Manual execution:

```powershell
python -m python.us.build_us_relative_strength_features
python -m python.us.build_us_relative_strength_features --universe NASDAQ100 --limit 10
python -m python.us.build_us_relative_strength_features --ticker AAPL
```

Planned storage:

- source: `market.us_stock_daily_price`
- target: `feature.us_stock_relative_strength_daily`

Preconditions:

1. target ticker prices must exist in `market.us_stock_daily_price`
2. `SPY` prices must exist in `market.us_stock_daily_price`
3. `QQQ` prices must exist in `market.us_stock_daily_price`

Missing benchmark policy:

- if `SPY` is missing, `spy_ret_*` and `rs_spy_*` fields remain null
- if `QQQ` is missing, `qqq_ret_*` and `rs_qqq_*` fields remain null
- the builder logs benchmark coverage and continues without affecting Korean systems

Validation checks:

1. inspect benchmark coverage counts for `SPY` and `QQQ`
2. inspect `null_ratio_by_window`
3. inspect duplicate key counts
4. confirm rows in `feature.us_stock_relative_strength_daily`

Notes:

- This builder is independent from Korean AI, Korean RULE, and live order flows.
- Relative strength rank percentiles are calculated per `trade_date`.

## Project C Phase 2-5: Label Builder

Current status:

- A standalone US label builder is implemented.
- It reads US daily price rows and writes a separate label table.
- It is not attached to any production scheduler.

Manual execution:

```powershell
python -m python.us.build_us_stock_labels
python -m python.us.build_us_stock_labels --universe NASDAQ100 --limit 10
python -m python.us.build_us_stock_labels --ticker AAPL
```

Notes:

- recent rows near the right edge of the dataset will naturally have null forward-return labels
- `SPY` and `QQQ` are excluded from top20 label universe by default

## Project C Phase 2-5: Dataset Validator

Current status:

- A standalone US dataset validator is implemented.
- It validates daily features, relative strength features, and labels for join readiness.
- Financial features are not automatically joined because leakage-safe as-of logic is not implemented yet.

Manual execution:

```powershell
python -m python.us.validate_us_stock_ml_dataset
python -m python.us.validate_us_stock_ml_dataset --universe NASDAQ100 --limit 20
```

Outputs:

- markdown report:
  - `reports/us_stock_dataset_validation.md`

Validation checks:

1. feature row count
2. label row count
3. joinable row count
4. label null ratios
5. label distribution
6. duplicate key counts
7. leakage-risk notes
