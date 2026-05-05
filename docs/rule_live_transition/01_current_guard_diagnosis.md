# RULE Live Transition: Current Guard Diagnosis

## Scope
- Reviewed:
  - `python/run_rule_before_open_cycle.py`
  - `python/rule_order_submitter.py`
  - `python/rule_order_preview_builder.py`
  - `python/rule_account_guard.py`
  - `python/rule_execution_simulator.py`
  - `python/rule_live_account_snapshot.py`
  - `python/rule_market_open_snapshot.py`
  - `python/rule_order_fill_sync.py`
  - `python/rule_portfolio_manager.py`
  - `python/rule_signal_builder.py`
  - `config/trading_calendar_kr.json`
  - `.env.example`
  - `docker-compose.yml`

## 1. `RULE_TRADING_RUN_MODE` branch
- Before-open entrypoint:
  - `python/run_rule_before_open_cycle.py:14-15`
  - `paper` -> `python/rule_execution_simulator.py`
  - `pilot` or `live` -> `python/rule_order_submitter.py`
- After-close build path:
  - `python/run_rule_after_close_cycle.py`
  - passes `--run-mode` into `rule_portfolio_manager.py` and `rule_order_preview_builder.py`
- After-open fill sync:
  - `python/run_rule_after_open_cycle.py`
  - `paper` -> skip
  - `pilot/live` -> `python/rule_order_fill_sync.py`

## 2. Where paper mode blocks real orders
- Hard block in order guard:
  - `python/rule_account_guard.py:185-186`
  - any `BUY` or `SELL` in `run_mode=paper` gets `paper_mode_no_order_submission`
- Before-open routing also prevents submitter use:
  - `python/run_rule_before_open_cycle.py:15`
- Submitter rejects non `pilot/live` preview:
  - `python/rule_order_submitter.py:309`
  - abort reason: `order_submitter_requires_pilot_or_live`

## 3. Where pilot/live can still be blocked
- Environment switches:
  - `python/rule_account_guard.py:188-191`
  - `RULE_LIVE_ENABLED != 1` -> `rule_live_disabled`
  - `RULE_ORDER_SUBMIT_ENABLED != 1` -> `rule_order_submit_disabled`
- Rule account env missing:
  - `python/rule_account_guard.py:127-129`
  - `kis_rule_cano_missing`
  - `kis_rule_acnt_prdt_cd_missing`
- Kill switch:
  - `python/rule_account_guard.py:193-194`
  - `RULE_KILL_SWITCH=1` -> `kill_switch_on`
- Order size and amount:
  - `python/rule_account_guard.py:205-216`
  - `final_order_amount_below_min_order_amount`
  - `order_qty_zero`
  - `order_amount_exceeds_limit`
  - `daily_order_amount_exceeds_limit`
  - `pilot_order_amount_exceeds_limit`
  - `pilot_order_qty_exceeds_limit`
- BUY-only strategy/risk filters:
  - `python/rule_account_guard.py:219-231`
  - `buy_requires_strong_entry`
  - `market_defensive_mode`
  - gap risk reasons from preview or open snapshot
  - trading value threshold failure
  - `sector_limit_failed`
  - `cooldown_failed`
  - `cash_limit_failed`
- Common live BUY guard:
  - `python/common_live_risk_guard.py`
  - examples:
    - `global_kill_switch_on`
    - `holdings_sync_missing` / `holdings_sync_stale`
    - `fills_sync_missing` / `fills_sync_stale`
    - `market_status_missing`
    - `same_symbol_buy_already_filled_today`
    - `daily_buy_amount_limit_exceeded`
    - `weekly_buy_amount_limit_exceeded`
    - `daily_loss_pct_unavailable`
    - `daily_loss_limit_reached`
    - `weekly_loss_pct_unavailable`
    - `weekly_loss_limit_reached`

## 4. `RULE_LIVE_ENABLED`
- Checked in `python/rule_account_guard.py:188-189`
- Applies only when `run_mode != paper`
- If unset or `0`, `pilot/live` preview can still be built, but actual order permission becomes `false`

## 5. `RULE_ORDER_SUBMIT_ENABLED`
- Checked in `python/rule_account_guard.py:190-191`
- Same behavior as `RULE_LIVE_ENABLED`
- Required in addition to `RULE_LIVE_ENABLED=1`

## 6. `RULE_BEFORE_OPEN_START_TIME` / `RULE_BEFORE_OPEN_END_TIME`
- Checked in `python/rule_execution_simulator.py:153-160`
- Used by both paper simulator and live submitter via shared `validate_trading_session()`
- Outside window aborts run with:
  - `outside_before_open_window_HH:MM_HH:MM`

## 7. `trading_calendar_kr.json` validation
- Calendar loader:
  - `python/rule_execution_simulator.py:67-81`
- Trading day validation:
  - `python/rule_execution_simulator.py:114-166`
- Closed day abort:
  - reason `calendar_closed_date`
- Current calendar note:
  - `config/trading_calendar_kr.json:11` includes `2026-05-05`
  - current `outputs/rule_execution_results.json` shows `run_mode=paper`, `trading_day_reason=calendar_closed_date`, `order_run_abort_reason=calendar_closed_date`

## 8. Rule-only account env vars
- Rule account resolver:
  - `python/rule_market_open_snapshot.py:46-63`
- Required for `pilot/live`:
  - `KIS_RULE_CANO`
  - `KIS_RULE_ACNT_PRDT_CD`
- KIS client auth still separately requires:
  - `KIS_APP_KEY`
  - `KIS_APP_SECRET`
  - `KIS_BASE_URL`

## 9. All known reasons for `order_allowed=false`
- Base guard:
  - `account_id_mismatch`
  - `strategy_id_mismatch`
  - `engine_type_mismatch`
  - `invalid_run_mode`
  - `no_order_action`
  - `paper_mode_no_order_submission`
  - `rule_live_disabled`
  - `rule_order_submit_disabled`
  - `kill_switch_on`
  - `final_order_amount_below_min_order_amount`
  - `order_qty_zero`
  - `order_amount_exceeds_limit`
  - `daily_order_amount_exceeds_limit`
  - `pilot_order_amount_exceeds_limit`
  - `pilot_order_qty_exceeds_limit`
  - `buy_requires_strong_entry`
  - `market_defensive_mode`
  - any gap risk reason
  - any trading value block reason such as `trading_value_ma20_below_paper_threshold`, `..._pilot_threshold`, `..._live_threshold`
  - `sector_limit_failed`
  - `cooldown_failed`
  - `cash_limit_failed`
- Common BUY guard merge:
  - all reasons returned by `python/common_live_risk_guard.py`

## 10. Last guard before actual KIS order API call
- Function:
  - `python/rule_order_submitter.py:_submit_items()`
- Sequence:
  1. preview-level `common_risk_allowed` re-check shortcut
  2. fresh `evaluate_rule_order_guard()` with actual open snapshot context
  3. only if allowed:
     - logs final approval
     - computes `ord_dvsn` and `ord_unpr`
     - calls `kis_live_account.order_cash()`
- Final decision point:
  - `python/rule_order_submitter.py:200-229`

## 11. Safety status of current defaults
- `.env.example:22-24`
  - `RULE_TRADING_RUN_MODE=paper`
  - `RULE_LIVE_ENABLED=0`
  - `RULE_ORDER_SUBMIT_ENABLED=0`
- `docker-compose.yml:179`, `203`, `227`
  - scheduler rule services default `RULE_TRADING_RUN_MODE` to `paper`

## 12. Changes added in this hardening pass
- Added guard reasons:
  - `daily_order_amount_exceeds_limit`
  - `pilot_order_amount_exceeds_limit`
  - `pilot_order_qty_exceeds_limit`
- Added pilot preview capping support:
  - `RULE_PILOT_MAX_ORDER_AMOUNT`
  - `RULE_PILOT_MAX_ORDER_QTY`
- Added submitter behavior:
  - explicit KIS auth failure abort before submit
  - final approval log before `order_cash()`
  - block log with code/name/side/qty/estimated amount/run mode
- Added additive output fields:
  - preview item:
    - `pilot_order_amount_cap_applied`
    - `pilot_order_qty_cap_applied`
  - submit/execution item:
    - `estimated_amount`
    - `daily_order_amount_used`
    - `final_guard_details`
