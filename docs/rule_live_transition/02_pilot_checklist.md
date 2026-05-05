# RULE Pilot Transition Checklist

## 1. Paper mode evidence that must exist first
- `RULE_TRADING_RUN_MODE=paper`
- `RULE_LIVE_ENABLED=0`
- `RULE_ORDER_SUBMIT_ENABLED=0`
- latest files exist and are readable:
  - `outputs/rule_portfolio_plan.json`
  - `outputs/rule_order_preview.json`
  - `outputs/rule_execution_results.json`
- paper before-open run finishes without unexpected exception
- if the day is a market holiday, abort reason is explicitly `calendar_closed_date`, not an auth or schema error

## 2. `outputs/rule_order_preview.json` checks
- top-level `run_mode` is what you intended
- `account_profile_valid=true`
- at least one candidate has `signal_strength=strong_entry`
- for intended submit targets:
  - `side` is `BUY` or `SELL`
  - `order_qty > 0`
  - `order_amount > 0`
  - `order_allowed=true`
- read blocked rows and understand every `order_block_reason`
- if `pilot_order_amount_cap_applied=true` or `pilot_order_qty_cap_applied=true`, confirm the cap is intentional

## 3. `outputs/rule_portfolio_plan.json` checks
- top-level `run_mode` matches the preview run mode
- `account_state.cash` and `account_state.total_equity` look realistic
- `summary.buy_count` is small enough for pilot
- each `buy` row has:
  - `strong_entry_signal=true`
  - `portfolio_action_reason=strong_entry_selected`
  - `cash_limit_pass=true`
  - `sector_limit_pass=true`
  - `cooldown_pass=true`
- no unintended `reduce` or `exit` action is present

## 4. `outputs/rule_execution_results.json` checks
- paper run:
  - `run_mode=paper`
  - `paper_only=true`
  - `order_run_aborted=false` on normal trading day
- pilot/live submit run:
  - `run_mode=pilot` or `live`
  - `paper_only=false`
  - `order_run_aborted=false`
  - `api_health_status` is not `auth_failed`
- if aborted, stop and fix root cause before the next run

## 5. Strong entry confirmation
- new BUYs should come only from `strong_entry_signal=true`
- if a BUY row is not strong entry, the guard should block it with `buy_requires_strong_entry`

## 6. Quantity and amount confirmation
- do not proceed if target BUY rows have `order_qty=0`
- do not proceed if `order_amount` is below `RULE_MIN_ORDER_AMOUNT`
- do not proceed if `order_amount` exceeds:
  - `RULE_MAX_ORDER_AMOUNT`
  - `RULE_MAX_DAILY_ORDER_AMOUNT`
  - pilot caps if configured

## 7. Conditions required for `order_allowed=true`
- `run_mode` is `pilot` or `live`
- `RULE_LIVE_ENABLED=1`
- `RULE_ORDER_SUBMIT_ENABLED=1`
- `RULE_KILL_SWITCH=0`
- rule account env exists:
  - `KIS_RULE_CANO`
  - `KIS_RULE_ACNT_PRDT_CD`
- KIS auth env exists:
  - `KIS_APP_KEY`
  - `KIS_APP_SECRET`
  - `KIS_BASE_URL`
- BUY-specific:
  - `signal_strength=strong_entry`
  - not market defensive
  - no gap risk block
  - trading value threshold passed
  - sector/cooldown/cash/common risk guards passed

## 8. Before-open execution timing
- confirm current KST time is inside:
  - `RULE_BEFORE_OPEN_START_TIME`
  - `RULE_BEFORE_OPEN_END_TIME`
- default window is `08:55` to `09:30`
- do not pilot outside that window

## 9. KIS auth and account checks
- test KIS auth before pilot day
- confirm `rule_market_open_snapshot.json` can be created without `auth_failed`
- confirm `python/rule_live_account_snapshot.py` can read the rule account
- confirm the account is the dedicated RULE account, not the main discretionary account

## 10. How to minimize pilot order size
- keep `RULE_TRADING_RUN_MODE=pilot`
- keep `RULE_MAX_ORDER_AMOUNT` low
- set `RULE_MAX_DAILY_ORDER_AMOUNT` low
- set `RULE_PILOT_MAX_ORDER_AMOUNT` low
- set `RULE_PILOT_MAX_ORDER_QTY` to `1` or another very small number
- reduce sizing inputs in after-close planning if needed:
  - `RULE_NEW_ENTRY_WEIGHT`
  - `RULE_MAX_POSITIONS`

## 11. Automatic stop criteria
- stop immediately if any of these happens:
  - `auth_failed`
  - `order_run_aborted=true`
  - `calendar_closed_date`
  - `outside_before_open_window_*`
  - `kis_rule_cano_missing`
  - `kis_rule_acnt_prdt_cd_missing`
  - unexpected `order_submit_failed`
  - same-day fill sync does not reconcile submitted orders
- also stop if submitted order count or amount is larger than planned

## 12. Recommended pilot cadence
- run paper for at least several trading days with stable outputs
- then pilot for 1 week with:
  - minimal qty
  - minimal daily amount
  - manual review of preview and execution results every day
- only consider `live` after pilot fills, sync, and account snapshots are consistent
