# AI ENV

## Purpose

This document summarizes environment variables used by the AI pipeline and live auto-trading flow.

## Core Data / DB

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `DATABASE_URL` | none | Primary research database connection | pipeline, training, sync |
| `WEB_DATABASE_URL` | none | Web payload sync database connection | `sync_web_display_data.py` |
| `USE_SQLITE_MIRROR` | `0` | Enable SQLite mirror reads | pipeline |
| `USE_SQLITE_FALLBACK_WRITES` | `0` | Enable SQLite fallback writes | pipeline |

## Model / Ranking

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `MODEL_VERSION` | project default | Active model version | train / predict |
| `HORIZON_DAYS` | project default | Prediction horizon | train / predict |
| `TOP_N` | project default | Top-N extraction size | ranking / recommendation |
| `SCORE_FORMULA_VERSION` | blank | Optional score formula override flag | ranking |

## Live Auto Trading

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `AUTO_TRADE_EXECUTE` | `0` | Enable real order submission | `submit_live_orders.py` | Keep disabled by default |
| `AUTO_TRADE_ALLOW_BUY` | `0` | Allow BUY order submission | live auto trade | SELL-only if disabled |
| `AUTO_TRADE_BUY_APPROVAL_REQUIRED` | `0` | Require manual BUY approval | live auto trade | Operational safety gate |
| `AUTO_TRADE_FORCE_RESUBMIT` | `0` | Ignore previous successful request ids | live auto trade | Use carefully |

## Alerts

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `SLACK_WEBHOOK_URL` | blank | Slack Incoming Webhook URL | KPI alerts / live auto-trade alerts |
| `ALERT_MIN_SCORE_THRESHOLD` | `40` | Warning threshold for Top20 mean `final_score` | `score_kpi_monitor.py` |

## KIS Auth

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `KIS_BASE_URL` | none | KIS base URL | AI / RULE shared |
| `KIS_APP_KEY` | none | KIS app key | AI path |
| `KIS_APP_SECRET` | none | KIS app secret | AI path |
| `KIS_CANO` | none | Account number | live sync / live orders |
| `KIS_ACNT_PRDT_CD` | none | Account product code | live sync / live orders |

## KIS Retry

| Variable | Default | Description | Note |
| --- | --- | --- | --- |
| `KIS_MAX_RETRY` | `3` | Maximum retry count | Preferred variable |
| `KIS_RETRY_WAIT_SEC` | `1` | Initial retry wait seconds | Preferred variable |
| `KIS_RETRY_BACKOFF_FACTOR` | `2` | Retry backoff multiplier | Example: `1s -> 2s -> 4s` |
| `KIS_RETRY_BACKOFF_MAX_SEC` | `30` | Maximum retry wait seconds | Backoff cap |
| `KIS_TIMEOUT_SEC` | `20` | Per-request timeout seconds | Minimum practical value is 5 |

## Retry Notes

- `429`: wait using `Retry-After` when present, then retry.
- `5xx`: retry with backoff.
- Other `4xx`: fail immediately.
- `order_cash` and `order_rvsecncl` should remain `no_retry=True` to avoid duplicate orders.
- If retries are exhausted, the system should log at critical level and attempt notifier delivery without stopping the main flow.
