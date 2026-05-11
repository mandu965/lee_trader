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

## Alerts (과제 4-B)

설정된 채널에만 발송. 미설정 시 콘솔로 fallback. 파일 로그는 항상 기록.
웹 확인: `/alerts.html` (`GET /api/alerts`)

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `SLACK_WEBHOOK_URL` | blank | Slack Incoming Webhook URL | KPI / live auto-trade alerts |
| `TELEGRAM_BOT_TOKEN` | blank | Telegram Bot 토큰 | KPI / live auto-trade alerts |
| `TELEGRAM_CHAT_ID` | blank | Telegram 채팅 ID | KPI / live auto-trade alerts |
| `ALERT_EMAIL_SMTP_HOST` | blank | SMTP 서버 호스트 | KPI / live auto-trade alerts |
| `ALERT_EMAIL_SMTP_PORT` | `587` | SMTP 포트 | alerts |
| `ALERT_EMAIL_SMTP_USER` | blank | SMTP 사용자 (미설정 시 FROM 주소 사용) | alerts |
| `ALERT_EMAIL_SMTP_PASSWORD` | blank | SMTP 비밀번호 | alerts |
| `ALERT_EMAIL_FROM` | blank | 발신 이메일 주소 | alerts |
| `ALERT_EMAIL_TO` | blank | 수신 이메일 주소 | alerts |
| `ALERT_MIN_SCORE_THRESHOLD` | `40` | 상위 20개 평균 final_score 경보 기준치 | `score_kpi_monitor.py` |
| `ALERT_LOG_PATH` | `outputs/alert_log.json` | 알림 로그 파일 경로 | notifier |
| `ALERT_LOG_MAX_ENTRIES` | `200` | 로그 파일 최대 보관 건수 | notifier |

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

## Project C Phase 2-2: US Financial Collector

The following variables are used by the standalone US stock financial collector.
The collector is implemented, but it is not attached to production schedulers.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_FINANCIAL_COLLECT_ENABLED` | `0` | Master switch for the standalone US financial collector | Project C US financial | Must stay disabled by default |
| `US_FINANCIAL_SOURCE` | `yfinance` | Financial data source | standalone collector | Only `yfinance` is supported now |
| `US_FINANCIAL_PERIOD_TYPES` | `annual,quarterly` | Period types to request | standalone collector | Only `annual` and `quarterly` are supported now |
| `US_FINANCIAL_LOOKBACK_YEARS` | `5` | History depth filter for fiscal periods | standalone collector | Older periods are skipped |
| `US_FINANCIAL_MAX_TICKERS_PER_RUN` | `100` | Per-run ticker cap | standalone collector | Prevents accidental large runs |
| `US_FINANCIAL_SLEEP_SEC` | `1.0` | Sleep between ticker requests | standalone collector | Conservative default |
| `US_FINANCIAL_RETRY_COUNT` | `3` | Retry count on temporary ticker failures | standalone collector | Applied per ticker |
| `US_FINANCIAL_RETRY_SLEEP_SEC` | `5` | Retry wait seconds | standalone collector | Applied between retries |
| `US_FINANCIAL_FAIL_FAST` | `0` | Stop on first ticker failure when enabled | standalone collector | Default keeps ticker failures isolated |
| `US_FINANCIAL_WRITE_MODE` | `upsert` | DB write mode | standalone collector | Only `upsert` is supported now |
| `US_FINANCIAL_LOG_LEVEL` | `INFO` | Collector log level | standalone collector | Safe default |

### Operating Notes

- `US_FINANCIAL_COLLECT_ENABLED=0` keeps the collector detached from current production schedulers.
- The collector is standalone only and is not wired into Korean schedulers.
- `US_FINANCIAL_*` settings must remain independent from Korean AI, RULE, and live-order flows.
