# RULE Operations

## Purpose

이 문서는 RULE 자동매매 운영 시 기본 실행 순서와 점검 포인트를 정리합니다.

## Main Flows

### After-Close

명령:

```powershell
docker compose run --rm scheduler-rule-after-close python python/run_rule_after_close_cycle.py
```

### Before-Open

명령:

```powershell
docker compose run --rm scheduler-rule-before-open python python/run_rule_before_open_cycle.py
```

### Live Account Sync

명령:

```powershell
docker compose run --rm scheduler-live-account-sync python python/rule_live_account_snapshot.py
```

### Web Payload Sync

명령:

```powershell
docker compose run --rm scheduler-live-account-sync python python/sync_web_display_data.py
```

## Recommended Sequence

1. `rule_live_account_snapshot.py`
2. `run_rule_after_close_cycle.py`
3. `run_rule_before_open_cycle.py`
4. `sync_web_display_data.py`

## Key Outputs

- [outputs/rule_order_preview.json](/d:/ai/lee_trader/outputs/rule_order_preview.json)
- [outputs/rule_execution_results.json](/d:/ai/lee_trader/outputs/rule_execution_results.json)
- [outputs/rule_account_live_state.json](/d:/ai/lee_trader/outputs/rule_account_live_state.json)
- [outputs/rule_dashboard_summary.json](/d:/ai/lee_trader/outputs/rule_dashboard_summary.json)

## Common Failure Reasons

- `outside_before_open_window_08:55_09:30`
- `calendar_closed_date`
- `order_submitter_requires_pilot_or_live`
- `previous_execution_aborted`
- KIS 계좌/인증 오류
