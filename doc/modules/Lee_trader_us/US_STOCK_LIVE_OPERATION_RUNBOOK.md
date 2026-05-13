# US Stock Live Operation Runbook

> 문서 역할: `상세 참고 문서`
>
> live safety baseline과 operator runbook을 상세하게 설명하는 문서다. 현재 운영 명령은 `OPERATIONS.md`와 함께 본다.

## 1. Document Purpose

This runbook consolidates the Phase 6 live-trading safety outputs into an operator-facing guide.

Current note:

- this document is still valid as the live-safety baseline
- the repository now also has Phase 7 Micro Live operations, reconciliation, and reporting layers
- use this document together with `PHASE7_SUMMARY.md` and the current `OPERATIONS.md`

Phase 6 is the baseline safety-design layer, not the full current state.
Current repository state extends this baseline through Phase 7 review and operations tooling.

## 2. Phase 6 Scope

Phase 6 covers:

- live-order safety policy
- live risk-policy defaults
- pre-trade candidate validation
- kill-switch state management
- manual approval-request lifecycle
- audit and operator review flow

Phase 6 does not cover:

- broker API attachment
- real order submission
- real fill handling
- real account balance reads
- Korean live-trading code changes

## 3. Phase 6 Architecture

```text
recommend.us_stock_rank_daily
    -> order candidate creation
    -> config/us_stock_live_risk_policy.yaml
    -> .env US_LIVE_* overrides
    -> utils/us_live_risk_policy.py
    -> risk.us_stock_live_kill_switch
    -> risk.us_stock_live_daily_risk_usage
    -> utils/us_live_pre_trade_check.py
    -> decision
       - BLOCK -> risk.us_stock_live_order_block_log
       - REQUIRE_APPROVAL -> risk.us_stock_live_order_approval
       - ALLOW -> no order in Phase 6
    -> utils/us_live_order_approval.py
    -> APPROVED / REJECTED / EXPIRED state management
```

Phase 6 results are validation artifacts only.
Even `ALLOW` or `APPROVED` does not create a real order in Phase 6.

## 4. Related Configuration Files

- `config/us_stock_live_risk_policy.yaml`
  - reviewed Micro Live safety defaults
  - all live gates default to disabled
  - manual approval defaults to enabled
  - market orders default to disabled

Required safe-default checks:

- `enabled: false`
- `live_trading_enabled: false`
- `live_order_enabled: false`
- `buy_enabled: false`
- `sell_enabled: false`
- `require_manual_approval: true`
- `real_order_blocked: true`
- `allow_market_order: false`
- `max_order_amount_usd <= 50`
- `max_daily_buy_amount_usd <= 100`
- `max_daily_new_buys <= 1`
- `min_cash_weight >= 0.50`

## 5. Related ENV

Mandatory safety flags:

```env
US_LIVE_TRADING_ENABLED=false
US_LIVE_ORDER_ENABLED=false
US_LIVE_BUY_ENABLED=false
US_LIVE_SELL_ENABLED=false
US_LIVE_REQUIRE_MANUAL_APPROVAL=true
US_LIVE_REAL_ORDER_BLOCKED=true
```

Core risk limits:

```env
US_LIVE_MAX_ORDER_AMOUNT_USD=50
US_LIVE_MIN_ORDER_AMOUNT_USD=10
US_LIVE_MAX_DAILY_BUY_AMOUNT_USD=100
US_LIVE_MAX_DAILY_SELL_AMOUNT_USD=500
US_LIVE_MAX_DAILY_ORDER_COUNT=3
US_LIVE_MAX_DAILY_NEW_BUYS=1
US_LIVE_MAX_POSITION_WEIGHT=0.05
US_LIVE_MAX_SYMBOL_POSITION_AMOUNT_USD=250
US_LIVE_MAX_POSITION_COUNT=5
US_LIVE_MAX_SECTOR_WEIGHT=0.20
US_LIVE_MIN_CASH_WEIGHT=0.50
```

Order-type and session restrictions:

```env
US_LIVE_ALLOW_MARKET_ORDER=false
US_LIVE_DEFAULT_ORDER_TYPE=LIMIT
US_LIVE_BLOCK_LEVERAGED_ETF=true
US_LIVE_BLOCK_INVERSE_ETF=true
US_LIVE_REGULAR_SESSION_ONLY=true
US_LIVE_BLOCK_PREMARKET=true
US_LIVE_BLOCK_AFTERHOURS=true
```

Phase 6 rule:

- never turn `US_LIVE_ORDER_ENABLED` to `true`
- `US_LIVE_REAL_ORDER_BLOCKED` must stay `true`

## 6. Related DB Tables

- `risk.us_stock_live_kill_switch`
  - current kill-switch state
- `risk.us_stock_live_kill_switch_event_log`
  - activate / clear / auto-trigger event history
- `risk.us_stock_live_daily_risk_usage`
  - daily notional, count, failure, and usage summary
- `risk.us_stock_live_order_block_log`
  - blocked candidate audit rows
- `risk.us_stock_live_order_approval`
  - manual approval-request state
- `risk.us_stock_live_order_approval_event_log`
  - request / approve / reject / expire audit history

Representative SQL:

```sql
SELECT
    kill_switch_id,
    scope,
    target_value,
    is_active,
    reason_code,
    activated_at,
    activated_by
FROM risk.us_stock_live_kill_switch
ORDER BY kill_switch_id;
```

```sql
SELECT
    trade_date,
    symbol,
    side,
    block_reason_code,
    block_reason_detail,
    check_stage,
    severity,
    created_at
FROM risk.us_stock_live_order_block_log
WHERE trade_date = '2026-05-15'
ORDER BY created_at DESC;
```

```sql
SELECT
    approval_id,
    trade_date,
    account_id,
    symbol,
    side,
    requested_order_amount_usd,
    requested_order_type,
    approval_status,
    requested_at,
    expires_at
FROM risk.us_stock_live_order_approval
WHERE approval_status = 'PENDING'
ORDER BY requested_at DESC;
```

```sql
SELECT
    event_id,
    approval_id,
    event_type,
    before_status,
    after_status,
    reason_code,
    reason_detail,
    performed_by,
    created_at
FROM risk.us_stock_live_order_approval_event_log
ORDER BY created_at DESC;
```

## 7. Related Utilities And Scripts

Utilities:

- `utils/us_live_risk_policy.py`
- `utils/us_live_pre_trade_check.py`
- `utils/us_live_kill_switch.py`
- `utils/us_live_order_approval.py`

Scripts:

- `scripts/validate_us_live_risk_policy.py`
  - validate safe-default risk policy
- `scripts/init_us_live_risk_state.py`
  - initialize or confirm risk-state rows
- `scripts/run_us_live_pre_trade_check.py`
  - evaluate manual or ranking-based candidates
- `scripts/manage_us_live_kill_switch.py`
  - list / activate / clear kill switches
- `scripts/evaluate_us_live_kill_switch.py`
  - evaluate auto-trigger conditions
- `scripts/manage_us_live_order_approval.py`
  - list / approve / reject / expire approval requests

## 8. Safe-Default Verification

Run:

```powershell
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format console
```

Check:

1. `US_LIVE_TRADING_ENABLED=false`
2. `US_LIVE_ORDER_ENABLED=false`
3. `US_LIVE_BUY_ENABLED=false`
4. `US_LIVE_SELL_ENABLED=false`
5. `US_LIVE_REQUIRE_MANUAL_APPROVAL=true`
6. `US_LIVE_REAL_ORDER_BLOCKED=true`
7. `allow_market_order=false`
8. `max_order_amount_usd <= 50`
9. `max_daily_buy_amount_usd <= 100`
10. `max_daily_new_buys <= 1`
11. `min_cash_weight >= 0.50`
12. result is `SAFE_DEFAULT`

If the result is not `SAFE_DEFAULT`, do not move toward Phase 7.

## 9. Risk Policy Validation Procedure

```powershell
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format console
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format markdown
```

Interpretation:

- `SAFE_DEFAULT`: expected Phase 6 state
- `SAFE_WITH_WARNINGS`: investigate before any next-step review
- `UNSAFE`: stop and fix configuration first

## 10. Kill Switch Operations

Status:

```powershell
python scripts/manage_us_live_kill_switch.py --list
```

Global stop:

```powershell
python scripts/manage_us_live_kill_switch.py --activate --scope GLOBAL --reason-code manual_stop --reason-detail "Operator requested emergency stop" --performed-by lee
```

BUY stop:

```powershell
python scripts/manage_us_live_kill_switch.py --activate --scope BUY --reason-code manual_stop --reason-detail "Block all BUY candidates" --performed-by lee
```

Symbol stop:

```powershell
python scripts/manage_us_live_kill_switch.py --activate --scope SYMBOL --target NVDA --reason-code data_error --reason-detail "NVDA price data abnormal" --performed-by lee
```

Clear:

```powershell
python scripts/manage_us_live_kill_switch.py --clear --scope SYMBOL --target NVDA --clear-reason "Price data verified" --performed-by lee
```

Rules:

- `clear_reason` is mandatory
- `performed_by` is mandatory
- automatic clear is not allowed by default
- global clear must be handled conservatively

## 11. Pre-Trade Check Operations

Single candidate:

```powershell
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --dry-run
```

Ranking batch:

```powershell
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run
```

Approval-request creation:

```powershell
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --create-approval-request --requested-by SYSTEM
```

Decision interpretation:

- `ALLOW`: policy-level pass only; still no real order in Phase 6
- `BLOCK`: inspect `risk.us_stock_live_order_block_log`
- `REQUIRE_APPROVAL`: create or review manual approval request
- `ERROR`: fix the validation/data problem first

## 12. Manual Approval Operations

List pending:

```powershell
python scripts/manage_us_live_order_approval.py --list --status PENDING
```

Detail:

```powershell
python scripts/manage_us_live_order_approval.py --approval-id <APPROVAL_ID>
```

Approve:

```powershell
python scripts/manage_us_live_order_approval.py --approval-id <APPROVAL_ID> --approve --approved-by lee --reason "Micro Live test approved"
```

Reject:

```powershell
python scripts/manage_us_live_order_approval.py --approval-id <APPROVAL_ID> --reject --rejected-by lee --reason "Market volatility too high"
```

Expire:

```powershell
python scripts/manage_us_live_order_approval.py --expire-pending
```

Rules:

- `APPROVED` is not order execution
- `APPROVED` only means the candidate can be reviewed in Phase 7
- Pre-Trade Check must be rerun immediately before any future live-order review

## 13. Block-Log Review Procedure

SQL:

```sql
SELECT
    trade_date,
    symbol,
    side,
    block_reason_code,
    block_reason_detail,
    check_stage,
    severity,
    created_at
FROM risk.us_stock_live_order_block_log
WHERE trade_date = '2026-05-15'
ORDER BY created_at DESC;
```

Interpretation:

- `live_disabled`: expected safe block under Phase 6 defaults
- `*_kill_switch_active`: inspect kill-switch state
- `daily_*`: inspect daily risk usage
- `price_*` or `volatility_*`: inspect price / feature inputs
- `approval_missing`: inspect approval-request state

## 14. Approval-Log Review Procedure

Pending approvals:

```sql
SELECT
    approval_id,
    trade_date,
    account_id,
    symbol,
    side,
    requested_order_amount_usd,
    requested_order_type,
    precheck_decision,
    approval_status,
    requested_at,
    expires_at
FROM risk.us_stock_live_order_approval
WHERE approval_status = 'PENDING'
ORDER BY requested_at DESC;
```

Approval events:

```sql
SELECT
    event_id,
    approval_id,
    event_type,
    before_status,
    after_status,
    reason_code,
    reason_detail,
    performed_by,
    created_at
FROM risk.us_stock_live_order_approval_event_log
ORDER BY created_at DESC;
```

## 15. Event-Log Review Procedure

Kill-switch events:

```sql
SELECT
    event_id,
    kill_switch_id,
    scope,
    target_value,
    event_type,
    reason_code,
    reason_detail,
    trigger_source,
    performed_by,
    before_is_active,
    after_is_active,
    created_at
FROM risk.us_stock_live_kill_switch_event_log
ORDER BY created_at DESC;
```

Use event logs to explain:

- who changed state
- why the state changed
- whether the change was manual or automatic
- when the state changed

## 16. Daily Operating Checks

```powershell
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1
python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1 --trade-date 2026-05-15
python scripts/manage_us_live_kill_switch.py --list
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run
python scripts/manage_us_live_order_approval.py --list --status PENDING
```

Phase 6 daily checks verify that the safety layer works without placing any real order.

## 17. Incident Response Procedure

Pre-Trade Check returns `ERROR`:

1. confirm risk-policy file load
2. confirm ENV presence
3. confirm risk tables exist
4. confirm ranking data exists
5. confirm price / feature data exists
6. confirm kill-switch state
7. inspect block log and console output

Kill switch active:

1. confirm scope
2. confirm reason code
3. confirm `activated_by`
4. inspect event log
5. do not clear before root cause is understood

Approval expired:

1. do not reuse expired approval
2. rerun Pre-Trade Check
3. recreate approval request if still needed

Block log grows abnormally fast:

1. confirm live gates are still false
2. confirm policy is not over-tightened unintentionally
3. inspect data gaps
4. inspect ranking quality
5. inspect repeated market-regime blocks

## 18. Phase 6 Completion Checklist

```text
[Phase 6 Completion Checklist]

Policy / Docs:
- [ ] US_STOCK_LIVE_TRADING_POLICY.md complete
- [ ] US_STOCK_LIVE_RISK_POLICY.md complete
- [ ] US_STOCK_LIVE_OPERATION_RUNBOOK.md complete

Configuration:
- [ ] config/us_stock_live_risk_policy.yaml exists
- [ ] .env.example contains US_LIVE_* settings
- [ ] all live enabled defaults are false
- [ ] manual approval default is true
- [ ] real_order_blocked default is true

DB:
- [ ] risk.us_stock_live_kill_switch exists
- [ ] risk.us_stock_live_kill_switch_event_log exists
- [ ] risk.us_stock_live_daily_risk_usage exists
- [ ] risk.us_stock_live_order_block_log exists
- [ ] risk.us_stock_live_order_approval exists
- [ ] risk.us_stock_live_order_approval_event_log exists

Utilities:
- [ ] us_live_risk_policy.py implemented
- [ ] us_live_pre_trade_check.py implemented
- [ ] us_live_kill_switch.py implemented
- [ ] us_live_order_approval.py implemented

Scripts:
- [ ] validate_us_live_risk_policy.py runnable
- [ ] init_us_live_risk_state.py runnable
- [ ] run_us_live_pre_trade_check.py runnable
- [ ] manage_us_live_kill_switch.py runnable
- [ ] evaluate_us_live_kill_switch.py runnable
- [ ] manage_us_live_order_approval.py runnable

Safety:
- [ ] no real-order API calls
- [ ] no real account reads
- [ ] no live order/fill/position table writes
- [ ] no Korean live-trading impact

Validation:
- [ ] SAFE_DEFAULT verified
- [ ] GLOBAL kill causes Pre-Trade Check BLOCK
- [ ] BLOCK candidates appear in block log
- [ ] REQUIRE_APPROVAL candidates create approval requests
- [ ] approval / reject / expire event logs confirmed
```

## 19. Phase 7 Entry Checklist

```text
[Phase 7 Entry Checklist]

Operational Readiness:
- [ ] Phase 4 backtest evidence exists
- [ ] Phase 4 Forward Test ran for at least 20 to 60 sessions
- [ ] Phase 5 Paper Trading ran stably for at least 20 to 60 sessions
- [ ] Paper Trading integrity issues are not recurring
- [ ] divergence between Paper Trading and Forward Test has been reviewed

Live Safety Layer:
- [ ] Phase 6 policy docs completed
- [ ] Pre-Trade Check works
- [ ] Kill Switch works
- [ ] manual approval flow works
- [ ] block / approval / event logs are recorded correctly
- [ ] SAFE_DEFAULT verification completed

Operating Conditions:
- [ ] Micro Live limits finalized
- [ ] single order amount remains within 10 to 50 USD
- [ ] max one new BUY per day
- [ ] SELL automation disabled or approval-gated
- [ ] market orders disabled
- [ ] limit orders only
- [ ] regular-session middle window only
- [ ] Kill Switch activation / clear procedure understood

Final Conditions:
- [ ] mock or sandbox verification completed before any broker attachment
- [ ] operator can manually approve candidates
- [ ] system can be stopped immediately on incident
```

If any item above is not satisfied, do not move to Phase 7 Micro Live.

## 20. Phase 7 Initial Restrictions

- BUY-only testing at first
- SELL automation disabled
- max single order `10` to `50` USD
- max one new BUY per day
- max three total orders per day
- max five holdings
- max symbol weight `2%` to `5%`
- min cash weight `50%`
- market orders prohibited
- limit orders only
- leveraged and inverse ETF prohibited
- new BUY prohibited on market selloff days
- manual approval required
- Kill Switch must remain available at all times

## 21. Live-Trading Prohibition Notice

Phase 6 is not a live-order implementation phase.

- Phase 6 does not place real orders.
- Phase 6 does not call real-order APIs.
- Phase 6 does not read a real account balance.
- Phase 6 does not modify real-account order, fill, or position tables.
- `ALLOW` or `APPROVED` does not mean execution.
- real ordering remains prohibited until a separate Phase 7 Micro Live implementation and approval path is completed.
