# Alert Policy

## Purpose

Define abnormal-condition alerts for KPI monitoring and live auto-trading.

## Channel

- Primary: Slack Incoming Webhook via `SLACK_WEBHOOK_URL`
- Fallback: console/log output when webhook is missing or delivery fails

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
