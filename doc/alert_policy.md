# Alert Policy

## Purpose

Define abnormal-condition alerts for KPI monitoring and live auto-trading.

## Channel

환경변수가 설정된 채널에만 발송됩니다. 미설정 시 콘솔(`logging.warning`)으로 fallback.
알림은 채널 발송 성공 여부와 무관하게 `outputs/alert_log.json` 에 항상 기록됩니다.
웹 페이지에서 확인: `/alerts.html`

| 채널 | 활성화 조건 환경변수 |
| --- | --- |
| 콘솔 / 파일 로그 | 항상 |
| Slack | `SLACK_WEBHOOK_URL` |
| Telegram | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| Email (SMTP) | `ALERT_EMAIL_SMTP_HOST` + `ALERT_EMAIL_FROM` + `ALERT_EMAIL_TO` |

## Levels

| Level | Meaning | Operator Action |
| --- | --- | --- |
| `INFO` | Informational, non-blocking | Review during routine checks |
| `WARNING` | KPI degradation or no-buy condition | Review the same day |
| `CRITICAL` | Live execution or fill-sync failure | Review immediately |

## KPI Alert Conditions

| Source | Condition | Level |
| --- | --- | --- |
| `python/score_kpi_monitor.py` | `walkforward_acceptance.status == REJECTED` | `WARNING` |
| `python/score_kpi_monitor.py` | Top20 mean `final_score <= ALERT_MIN_SCORE_THRESHOLD` | `WARNING` |
| `python/score_kpi_monitor.py` | `buy_eligibility.status == BUY_ALLOWED` count is `0` | `WARNING` |

## Live Trading Alert Conditions

| Source | Condition | Level |
| --- | --- | --- |
| `python/run_live_auto_trade_cycle.py` | `submit_live_orders.py` fails | `CRITICAL` |
| `python/run_live_auto_trade_cycle.py` | `sync_live_order_fills.py` fails | `CRITICAL` |

## Rules

- Alert delivery failure must never stop the main pipeline.
- Missing `SLACK_WEBHOOK_URL` must not stop operations.
- Live failure alerts should include failed step, exit code, and command.
