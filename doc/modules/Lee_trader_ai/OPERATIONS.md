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
