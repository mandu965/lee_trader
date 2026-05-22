# US Stock Live Risk Policy

> 문서 역할: `상세 참고 문서`
>
> live risk policy 구조, YAML/ENV baseline, risk table 의미를 자세히 설명하는 정책 문서다.

> 상태 메모: 2026-05-22 기준 현재 운영 원칙상 US는 `paper-only` 유지다.
> 이 문서는 현재 실전 적용 문서가 아니라, deferred live-risk design reference로 사용한다.

## 1. Phase 6-2 Purpose

Phase 6-2 converts the Phase 6-1 live-trading safety policy into reusable configuration and state structures.

Current note:

- this document remains the baseline risk-policy reference
- the project has since progressed through Phase 7
- Phase 7 Micro Live, reconciliation, and operations reporting build on top of this policy structure

This policy layer still does not by itself authorize real-order execution.
It remains a baseline control document.

## 2. Risk Policy Management Mode

- reviewed baseline policy: YAML file
- operator override layer: ENV
- live operating state: DB tables

Recommended split:

- `config/us_stock_live_risk_policy.yaml`: reviewed Micro Live defaults
- `.env`: conservative override layer
- `risk.*` tables: kill switch state, daily usage, block-log history

## 3. YAML Policy File Structure

Primary file:

- `config/us_stock_live_risk_policy.yaml`

Primary profile:

- `US_LIVE_RULE_V1`

Main sections:

- `safety`
- `account`
- `order`
- `position`
- `sector`
- `instrument`
- `market`
- `time`
- `approval`
- `notification`

Safe-default rules:

- all live enabled flags remain `false`
- manual approval remains `true`
- `real_order_blocked` remains `true`
- `allow_market_order` remains `false`

## 4. ENV Defaults

Key ENV overrides:

- `US_LIVE_RISK_POLICY_ID=US_LIVE_RULE_V1`
- `US_LIVE_RISK_POLICY_FILE=config/us_stock_live_risk_policy.yaml`
- `US_LIVE_REAL_ORDER_BLOCKED=true`
- `US_LIVE_MAX_ORDER_AMOUNT_USD=50`
- `US_LIVE_MIN_ORDER_AMOUNT_USD=10`
- `US_LIVE_MAX_DAILY_BUY_AMOUNT_USD=100`
- `US_LIVE_MAX_DAILY_ORDER_COUNT=3`
- `US_LIVE_MAX_DAILY_NEW_BUYS=1`
- `US_LIVE_MIN_CASH_WEIGHT=0.50`

These values are configuration only in Phase 6-2.

## 5. Kill Switch State Table

Table:

- `risk.us_stock_live_kill_switch`

Purpose:

- store active/inactive kill-switch state
- separate global and scoped switches
- retain activation and clear metadata

Main scopes:

- `GLOBAL`
- `BUY`
- `SELL`
- `SYMBOL`
- `SECTOR`
- `ACCOUNT`

Phase 6-4 additions:

- `target_value` stores the concrete scope target such as `ALL`, `BUY`, `NVDA`, `TECHNOLOGY`, or `US_LIVE_TEST`
- kill-switch rows are now paired with append-only event logs for audit traceability

## 5-A. Kill Switch Event Log Table

Table:

- `risk.us_stock_live_kill_switch_event_log`

Purpose:

- record every activate and clear action
- preserve operator identity and trigger source
- keep `before_is_active` and `after_is_active` for later audit

Event types:

- `ACTIVATE`
- `CLEAR`
- `CHECK`
- `AUTO_TRIGGER`
- `FAILED_ACTIVATE`
- `FAILED_CLEAR`

## 6. Daily Risk Usage Table

Table:

- `risk.us_stock_live_daily_risk_usage`

Purpose:

- daily order-count tracking
- daily buy/sell notional tracking
- failure/rejection/block counter tracking
- daily PnL and exposure snapshot storage for later pre/post-trade checks

## 7. Order Block Log Table

Table:

- `risk.us_stock_live_order_block_log`

Purpose:

- audit blocked live-order candidates
- record reason code and severity
- support later operator review and pre-trade diagnostics

Representative check stages:

- `SYSTEM_FLAG`
- `KILL_SWITCH`
- `RANKING`
- `INSTRUMENT`
- `MARKET`
- `POSITION`
- `SECTOR`
- `DAILY_LIMIT`
- `TIME_WINDOW`
- `APPROVAL`

## 8. Risk Policy Loader

Utility:

- `utils/us_live_risk_policy.py`

Functions:

- `load_us_live_risk_policy()`
- `validate_us_live_risk_policy()`
- `print_us_live_risk_policy_summary()`

Loader behavior:

1. load YAML profile
2. apply ENV overrides
3. validate safe defaults
4. print summary for operator review

## 9. Policy Validation Script

Script:

- `python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format console`

Supported outputs:

- `console`
- `markdown`

Validation result classes:

- `SAFE_DEFAULT`
- `SAFE_WITH_WARNINGS`
- `UNSAFE`

## 10. Risk State Initialization Script

Script:

- `python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1 --dry-run`
- `python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1 --trade-date 2026-05-15`

Initialization behavior:

- ensure risk tables exist
- upsert default kill-switch rows
- upsert one daily usage row for the requested trade date

Default kill switches:

- `US_LIVE_GLOBAL_KILL`
- `US_LIVE_BUY_KILL`
- `US_LIVE_SELL_KILL`

## 10-A. Phase 6-4 Kill Switch Utilities

Utility:

- `utils/us_live_kill_switch.py`

Main functions:

- `build_kill_switch_id()`
- `get_kill_switch_status()`
- `is_kill_switch_active()`
- `activate_kill_switch()`
- `clear_kill_switch()`
- `list_active_kill_switches()`
- `check_kill_switch_for_order_candidate()`
- `evaluate_kill_switch_triggers()`

Activation policy:

1. build `kill_switch_id` from `scope + target_value`
2. upsert kill-switch state with `is_active = true`
3. keep the first `activated_at` when the switch is already active
4. refresh `reason_code`, `reason_detail`, and `updated_at`
5. append an `ACTIVATE` event row

Clear policy:

1. require `clear_reason`
2. require `performed_by`
3. set `is_active = false`
4. write `cleared_at`, `cleared_by`, and `clear_reason`
5. append a `CLEAR` event row

## 11. Basic SQL Checks

Kill switch:

```sql
SELECT
    kill_switch_id,
    scope,
    target_value,
    is_active,
    reason_code,
    reason_detail,
    activated_at,
    activated_by,
    cleared_at,
    cleared_by
FROM risk.us_stock_live_kill_switch
ORDER BY kill_switch_id;
```

Daily usage:

```sql
SELECT
    trade_date,
    policy_id,
    account_id,
    buy_order_count,
    sell_order_count,
    total_order_count,
    buy_amount_usd,
    sell_amount_usd,
    failed_order_count,
    blocked_order_count,
    daily_pnl_usd,
    daily_pnl_pct,
    cash_weight,
    max_position_weight,
    max_sector_weight
FROM risk.us_stock_live_daily_risk_usage
WHERE trade_date = DATE '2026-05-15'
ORDER BY account_id;
```

Block log:

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
WHERE trade_date = DATE '2026-05-15'
ORDER BY created_at DESC;
```

Kill switch event log:

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

## 12. SAFE_DEFAULT Validation Standard

SAFE_DEFAULT requires:

- `US_LIVE_TRADING_ENABLED=false`
- `US_LIVE_ORDER_ENABLED=false`
- `US_LIVE_BUY_ENABLED=false`
- `US_LIVE_SELL_ENABLED=false`
- `US_LIVE_REQUIRE_MANUAL_APPROVAL=true`
- `US_LIVE_REAL_ORDER_BLOCKED=true`
- `allow_market_order=false`
- `max_order_amount_usd <= 50`
- `max_daily_buy_amount_usd <= 100`
- `max_daily_order_count <= 3`
- `max_daily_new_buys <= 1`
- `min_cash_weight >= 0.50`

## 13. Phase 6-3 Integration Path

Planned Phase 6-3 flow:

```text
recommend.us_stock_rank_daily
    ↓
order candidates
    ↓
load_us_live_risk_policy()
    ↓
risk.us_stock_live_kill_switch
    ↓
risk.us_stock_live_daily_risk_usage
    ↓
Pre-Trade Check
    ↓
allow or block
    ↓
blocked candidates recorded in risk.us_stock_live_order_block_log
```

Planned Phase 6-3 checks:

- live enabled flag check
- kill switch check
- rank/grade check
- instrument check
- price/gap/volatility check
- account/cash check
- position/sector exposure check
- daily limit check
- time window check
- manual approval check

## 14. Phase 6-3 Pre-Trade Check

Phase 6-3 introduces a reusable staged validator:

- candidate input structure: `UsLiveOrderCandidate`
- result output structure: `UsLivePreTradeCheckResult`
- decisions:
  - `ALLOW`
  - `BLOCK`
  - `REQUIRE_APPROVAL`
  - `ERROR`

Main check stages:

- `SYSTEM_FLAG`
- `KILL_SWITCH`
- `RANKING`
- `INSTRUMENT`
- `PRICE`
- `MARKET`
- `DAILY_LIMIT`
- `POSITION`
- `SECTOR`
- `TIME_WINDOW`
- `APPROVAL`

Pre-Trade Check rules in Phase 6-3:

- `ALLOW` still does not create or send any real order
- `BLOCK` and `ERROR` candidates can be appended to `risk.us_stock_live_order_block_log`
- missing live account, position, or sector exposure state should bias to `REQUIRE_APPROVAL`
- the validator must not import broker APIs or read a real account balance

Example commands:

```powershell
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --dry-run
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side SELL --amount-usd 50 --dry-run
```

Phase 6-4 kill-switch integration:

- `KILL_SWITCH` stage now uses `utils/us_live_kill_switch.py`
- active `GLOBAL`, `BUY`, `SELL`, `SYMBOL`, `SECTOR`, and `ACCOUNT` switches force `BLOCK`
- block-log rows and kill-switch event-log rows remain separate audit streams
- `ALLOW` from Pre-Trade Check still does not create any real order

## 15. Safe Default Principle

Risk policy defaults must bias toward no execution, not convenience.

That means:

- live gates disabled
- market orders disabled
- manual approval enabled
- cash reserve high
- order limits small
- kill-switch state tracked separately from broker code

## 16. Phase 6-5 Manual Approval Flow

Phase 6-5 adds a manual-approval layer between Pre-Trade Check and any future Micro Live review.

Flow:

```text
order candidate
    -> Pre-Trade Check
    -> ALLOW or REQUIRE_APPROVAL
    -> risk.us_stock_live_order_approval
    -> operator approve / reject / expire
    -> Phase 7 review candidate only
```

Important rules:

- `BLOCK` candidates stay in `risk.us_stock_live_order_block_log`
- `ERROR` candidates do not create approval requests
- `REQUIRE_APPROVAL` candidates create `PENDING` approval requests
- `ALLOW` candidates may also be stored as approval requests in Phase 6 because no real order is allowed yet
- `APPROVED` does not create a real order
- any future live-order review must rerun Pre-Trade Check immediately before execution

Approval tables:

- `risk.us_stock_live_order_approval`
- `risk.us_stock_live_order_approval_event_log`

Approval statuses:

- `PENDING`
- `APPROVED`
- `REJECTED`
- `EXPIRED`
- `CANCELED`
- `ERROR`

Approval event types:

- `REQUEST`
- `APPROVE`
- `REJECT`
- `EXPIRE`
- `CANCEL`
- `CHECK`
- `ERROR`

Required approve inputs:

- `approval_id`
- `approved_by`
- `approval_reason`

Required reject inputs:

- `approval_id`
- `rejected_by`
- `reject_reason`

Expiry policy:

- default expiry comes from `approval.approval_expires_minutes`
- if `expires_at` is passed while status is `PENDING`, the row becomes `EXPIRED`
- expired approvals are not valid for later Micro Live review

Approval commands:

```powershell
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --create-approval-request --requested-by SYSTEM
python scripts/manage_us_live_order_approval.py --list --status PENDING
python scripts/manage_us_live_order_approval.py --approval-id USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000 --approve --approved-by lee --reason "Micro Live test approved"
python scripts/manage_us_live_order_approval.py --approval-id USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000 --reject --rejected-by lee --reason "Rejected for review"
python scripts/manage_us_live_order_approval.py --expire-pending
```

Approval SQL:

```sql
SELECT
    approval_id,
    trade_date,
    account_id,
    symbol,
    side,
    requested_order_amount_usd,
    requested_order_type,
    rank_no,
    recommend_grade,
    total_score,
    precheck_decision,
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
WHERE approval_id = 'USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000'
ORDER BY created_at;
```

Notification options:

- `US_LIVE_APPROVAL_NOTIFY_ENABLED=true`
- `US_LIVE_APPROVAL_EXPIRES_MINUTES=30`

If notifier integration is unavailable, DB persistence remains the primary source of truth.

## 17. Live-Trading Prohibition Note

Phase 6-2 is a risk-structure design step only.

- no broker API call
- no real order creation
- no real account lookup
- no real position synchronization
- no modification of Korean live-trading logic

Phase 6-4 remains inside the same safety-only boundary.

- kill-switch scripts only manage risk state
- they do not submit or route orders
- they do not read a broker account
- they do not modify Korean live-trading code

Phase 6-5 remains inside the same safety-only boundary.

- approval scripts only manage approval-request state
- they do not create broker-order payloads
- they do not read a real account balance
- approved status is not execution
- they do not modify Korean live-trading code
