# Prompt 8 Work Log - 2026-05-01

## Scope

- Prompt 8 only
- preview-only AI/RULE integrated master approval layer
- no submit wiring

## Added File

- `python/master_risk_manager.py`

## Outputs

- `outputs/master_approved_orders.json`
- `outputs/master_blocked_orders.json`
- `outputs/master_risk_summary.json`
- `outputs/master_risk_summary.md`

## Master Risk Criteria

- duplicate BUY code across AI and RULE
- engine daily budget
- total daily BUY budget
- sector exposure limit
- theme exposure limit
- cash ratio floor
- common risk guard pass
- AI entry gate pass
- upstream preview block reason passthrough except submit-only guards

## Default Controls

- `MASTER_RISK_ENGINE_DAILY_BUDGET_AI=500000`
- `MASTER_RISK_ENGINE_DAILY_BUDGET_RULE=500000`
- `MASTER_RISK_TOTAL_DAILY_BUY_BUDGET=1000000`
- `MASTER_RISK_MAX_SECTOR_EXPOSURE_PCT=0.35`
- `MASTER_RISK_MAX_THEME_EXPOSURE_PCT=0.30`
- `MASTER_RISK_MIN_CASH_RATIO=0.20`

## Tests

- `python -m py_compile python/master_risk_manager.py`
- `python python/master_risk_manager.py --self-test`
- `python python/master_risk_manager.py`

## Self-test Results

- AI preview only: approved
- RULE preview only: approved
- AI/RULE same symbol: both blocked as `duplicate_buy_candidate_across_engines`
- common risk blocked row: blocked as `common_risk_blocked`
- entry gate blocked row: blocked as `entry_price_gate_blocked`
- approved/blocked/summary markdown outputs created

## Local Preview Result

- AI BUY candidates: `0`
- RULE BUY candidates: `2`
- approved: `0`
- blocked: `2`
- blocked reason: `common_risk_blocked`

## Notes

- actual order submission remains disconnected
- AI preview entry gate is enforced when present
- RULE preview currently has no dedicated entry gate field; missing gate is treated as neutral, explicit blocked state would be enforced
- holdings sector/theme exposure is only computed when labels are available in inputs

## Next-step Requirements Before Live Wiring

- stable live holdings classification for sector/theme
- agreed source of truth for engine budgets
- explicit rule entry gate if RULE BUY should use same live-price gate
- integration point after preview and before submit with kill switch first

## Status

- Prompt 8: `LOCAL_TESTED`
- Server applied: no
