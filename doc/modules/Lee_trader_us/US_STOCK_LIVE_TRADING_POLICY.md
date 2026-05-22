# US Stock Live Trading Policy

> 문서 역할: `상세 참고 문서`
>
> 현재 운영 명령 문서가 아니라, Micro Live 이전과 이후를 관통하는 live safety baseline policy 문서다.

> 상태 메모: 2026-05-22 기준 현재 운영 원칙상 US는 `paper-only` 유지다.
> 이 문서는 active rollout guide가 아니라, 향후 재검토 시 참조할 live safety baseline 정책 문서다.

## 1. Document Purpose

This document defines the safety-first policy that must exist before any US stock live-trading implementation is considered.

Current note:

- this document is still the baseline policy reference
- the codebase has now progressed through Phase 7
- read this document as the safety-policy baseline that later Phase 7 Micro Live work was built on
- this is not the current operations report document

Phase 6 is not the stage that attaches real broker execution.
Phase 6 is the stage that defines order policy, risk limits, approval flow, blocking rules, and kill-switch behavior.

Phase 6 does not place real orders.
Phase 6 does not call live-trading order APIs.
Phase 6 is the pre-live safety-policy design stage.

## 2. Phase 6 Scope

Phase 6 covers:

- live-order policy definition
- risk-limit definition
- pre-trade blocking rules
- approval and audit policy
- kill-switch policy
- operator reporting requirements

Phase 6 does not cover:

- live broker API execution
- real account balance reads
- real order submission
- real fill handling
- Korean live-trading code changes

## 3. Live-Trading Preconditions

The following conditions must be satisfied before any Phase 7 Micro Live review:

1. Phase 4 backtest evidence exists.
2. Phase 4 Forward Test has accumulated at least 20 to 60 trading days.
3. Phase 5 Paper Trading has operated stably for at least 20 to 60 trading days.
4. Paper Trading order generation -> fill simulation -> snapshot -> report repeats cleanly.
5. Paper Trading integrity errors do not recur.
6. Forward Test and Paper Trading results do not diverge severely.
7. Real-order blocking safeguards remain active.
8. A pre-trade validation module exists.
9. A kill switch exists.
10. Order-failure, partial-fill, and unfilled-order response policy is documented.

If these conditions are not met, the system must not move into any Micro Live review path.

## 4. Order Allow Conditions

All of the following must be true before a live order is allowed.

System conditions:

- `LIVE_TRADING_ENABLED=true`
- `US_LIVE_TRADING_ENABLED=true`
- kill switch is not active
- live account state is known and healthy
- price/data collection finished successfully
- ranking calculation finished successfully
- pre-trade validation passed

Symbol conditions:

- active universe member
- `recommend_grade in ('STRONG_BUY', 'BUY')`
- `rank_no <= 20`
- `data_status in ('OK', 'PARTIAL_DATA')`
- `exclude_reason is null`
- price data is valid
- liquidity threshold is met
- not a leveraged ETF
- not an inverse ETF

Risk conditions:

- single-order limit is not exceeded
- daily order limit is not exceeded
- symbol-weight limit is not exceeded
- sector-weight limit is not exceeded
- minimum cash reserve remains above threshold
- daily loss cap is not breached
- no duplicate order exists

## 5. Order Block Conditions

System blocks:

- `LIVE_TRADING_ENABLED=false`
- `US_LIVE_TRADING_ENABLED=false`
- kill switch active
- account state unknown
- broker API outage
- price/data collection failure
- ranking output missing
- required price data missing
- required FX data missing if needed later
- pre-trade check failure

Symbol blocks:

- `recommend_grade = EXCLUDE`
- rank threshold not met
- `data_status = ERROR`
- `exclude_reason` exists
- leveraged ETF
- inverse ETF
- trading halt or invalid price state
- excessive gap move
- excessive volatility

Risk blocks:

- insufficient cash
- minimum cash weight would be violated
- single-order amount exceeds limit
- daily buy amount exceeds limit
- daily order count exceeds limit
- symbol weight exceeds limit
- sector weight exceeds limit
- daily loss limit exceeded
- pending unfilled order exists
- duplicate same-symbol order exists

## 6. Account / Cash Limits

Micro Live must begin with very small limits.

Recommended initial live limits:

- max single BUY order: `10` to `50` USD
- max daily BUY amount: `100` USD
- max daily SELL amount: only within current holdings
- min cash weight: `50%`
- max position count: `5`

The purpose of early Micro Live is connectivity and policy validation, not return maximization.
Initial order size must stay very small.

## 7. Symbol Limits

- max symbol weight: `2%` to `5%`
- one symbol should not receive multiple new BUY orders on the same day
- symbols with missing or warning-grade data must be blocked or require manual approval
- leveraged and inverse ETFs are blocked

## 8. Sector Limits

- max sector weight: `20%`
- if sector classification is missing, the order should be blocked or require manual approval
- if sector exposure cannot be computed reliably, no fully automatic live BUY should be allowed

## 9. Daily Order Limits

- max daily new BUY symbols: `1`
- max daily total order count: `3`
- max daily order failures before escalation: `3`
- repeated same-symbol retries are not allowed beyond the defined retry cap

## 10. Price / Gap / Volatility Limits

Recommended conservative thresholds:

- symbol gap up above `+5%`: block BUY or require manual approval
- symbol gap down below `-5%`: block BUY or require manual approval
- 20-day volatility above policy threshold: block BUY
- intraday rise above `+7%`: block BUY
- intraday drop below `-7%`: block new BUY
- `SPY` same-day drop below `-2%`: block new BUY
- `QQQ` same-day drop below `-2.5%`: block tech-heavy BUY

On market selloff days, new BUY orders should not be submitted.

## 11. Market-Regime Limits

- `BULL_LOW_VOL`: baseline policy allowed
- `BULL_HIGH_VOL`: smaller order size and fewer new BUYs
- `SIDEWAYS`: only stronger signals allowed
- `BEAR_LOW_VOL`: reduce order size or restrict new BUYs
- `BEAR_HIGH_VOL`: new BUYs blocked, SELLs manual approval only

This regime policy is documented in Phase 6-1 only.
Actual enforcement belongs to later pre-trade validation work.

## 12. BUY Policy

BUY candidate requirements:

- `recommend_grade in ('STRONG_BUY', 'BUY')`
- `rank_no <= 20`
- same symbol has behaved normally in Paper Trading
- data quality is acceptable
- price, gap, and volatility checks pass

BUY blocking examples:

- market selloff condition
- symbol gap-up excess
- volatility excess
- current holding already above target limit
- same-symbol pending BUY exists
- same-symbol BUY already attempted today
- order amount violates min/max policy

Recommended initial Micro Live BUY policy:

- max `1` automatic BUY candidate per day
- choose the highest-ranked eligible symbol only
- order amount stays within `10` to `50` USD

## 13. SELL Policy

SELL candidates:

- current holding exits Top20
- grade falls to `HOLD` or `EXCLUDE`
- data state becomes `ERROR`
- risk limit violation
- manual liquidation request

Recommended initial policy:

- BUY automation may be tested first
- SELL stays manual or separately approved
- auto stop-loss / auto take-profit remains disabled

In early live testing, execution-path stability matters more than SELL automation.

## 14. Order Type Policy

Preferred initial type:

- `LIMIT` first

`MARKET` order policy:

- not preferred for early Micro Live
- may be allowed only for very small manual tests if explicitly approved

Illustrative limit rules:

- BUY `limit_price = latest_price * 0.995`
- SELL limit should also use a conservative buffer and documented rule

Because US equities can move quickly, LIMIT orders are safer than MARKET orders in early Micro Live.

## 15. Order Time Policy

- regular session only
- no pre-market order
- no after-hours order
- no order in the first `5` to `15` minutes after open
- no order in the last `5` to `15` minutes before close
- no holiday order
- no order when data date and intended order date are inconsistent

Phase 6-1 documents the policy only.
Time-window enforcement belongs to later pre-trade checks.

## 16. Duplicate Order Prevention Policy

- block same `account + trade_date + symbol + side` duplication
- block new order when same-symbol unfilled order exists
- block same-symbol add-on order while partial fill remains unresolved
- all duplicate blocks must be logged with explicit reason

## 17. Order Failure Response Policy

- log the failure reason
- no infinite retry loop
- retry count must be capped
- retry spacing must be defined
- after repeated failures, stop the symbol and then stop new BUYs if the issue looks systemic
- send operator alert

Recommended starting policy:

- max retry count: `1` to `2`
- same-symbol intraday retry max: `1`
- if total live-order failures reach `3`, treat as kill-switch candidate

## 18. Partial Fill Response Policy

- record filled quantity
- record remaining quantity
- do not auto-create an additional order immediately
- re-evaluate in the next cycle
- consider unfilled-order cancel review after timeout
- block same-symbol additional order while a partial fill remains unresolved

Initial Micro Live policy:

- partial fill results in alert plus manual review
- no automatic partial-fill follow-up

## 19. Kill Switch Policy

Kill switch trigger examples:

1. daily loss limit exceeded
2. repeated order failures beyond threshold
3. broker API outage detected
4. account balance inconsistency
5. DB position and broker position mismatch
6. negative or abnormal cash balance
7. duplicate order creation detected
8. expected order amount and actual amount mismatch
9. price data abnormality
10. severe market selloff condition
11. manual emergency stop request

Kill switch actions:

- stop all new BUYs immediately
- stop or heavily restrict automated SELL logic
- review cancel of pending orders
- send alert
- write operational log
- require manual release before restart

## 19-A. Phase 6-2 Risk Structure Link

Phase 6-2 translates this live-trading policy into reusable configuration and state structures.

Reference:

- [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md)

Phase 6-2 adds:

- reviewed YAML policy defaults
- ENV override structure
- kill-switch state table
- daily risk-usage table
- blocked-order audit log table

These structures remain safety and validation artifacts only.
They still do not place real orders or call broker APIs.

## 19-B. Phase 6-4 Kill Switch Management Link

Phase 6-4 implements kill-switch state management on top of the Phase 6-2 structures.

Reference:

- [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md)

Phase 6-4 adds:

- scoped kill-switch IDs and `target_value` handling
- activate / clear event logging
- manual kill-switch CLI
- automatic trigger evaluation
- direct Pre-Trade Check integration

This still does not place real orders or call any broker API.

## 20. Manual Approval Policy

Recommended early-live approval flow:

- system prepares candidate order
- human approves or rejects it
- only approved order becomes eligible for Micro Live

Approval-required examples:

- all new BUYs
- all SELLs
- order amount above small-test threshold
- high-volatility names
- market selloff day orders
- symbols with data-quality warnings

Required approval-log fields:

- `approval_id`
- `trade_date`
- `symbol`
- `side`
- `suggested_amount`
- `approved_amount`
- `approver`
- `approval_status`
- `approval_reason`
- `created_at`

Phase 6-5 implements the approval-request storage and audit flow without creating any real order.

Phase 6-5 adds:

- `risk.us_stock_live_order_approval`
- `risk.us_stock_live_order_approval_event_log`
- approval-request creation from `ALLOW` or `REQUIRE_APPROVAL` Pre-Trade Check results
- manual approve / reject / expire CLI flow

Approval-request creation policy:

- `BLOCK`: block log only, no approval request
- `ERROR`: error reporting only, no approval request
- `REQUIRE_APPROVAL`: create `PENDING`
- `ALLOW`: still create approval request in Phase 6 when manual approval remains required

Approval expiry policy:

- default approval lifetime is `30` minutes unless policy override is provided
- expired `PENDING` requests become `EXPIRED`
- approved requests still require a fresh Pre-Trade Check before any later live-order review

Manual approval remains a control gate, not an execution trigger.
`APPROVED` means only that the candidate can be reviewed in Phase 7 Micro Live.

## 21. Alerts / Reporting Policy

Alert classes:

- order candidate created
- approval required
- order blocked
- order failed
- partial fill
- kill switch triggered
- daily loss near limit
- account / DB inconsistency
- daily operating summary

Required reports:

- daily candidate-order report
- pre-trade check report
- blocked-order list
- submitted-order list
- fill result report
- position-change report
- account PnL report
- risk-limit usage report
- kill-switch state report

## 22. Logs / Audit Trail Policy

- log every order candidate
- log every block reason
- log every approval / rejection
- log every order request / response
- log every fill / no-fill / failure
- log every kill-switch trigger / release
- operators must be able to explain later why an order was allowed or blocked

Approval and block-log relationship:

- `BLOCK` -> `risk.us_stock_live_order_block_log`
- `ERROR` -> error log or block-log style audit entry only
- `REQUIRE_APPROVAL` -> `risk.us_stock_live_order_approval` with status `PENDING`
- `ALLOW` -> still approval-gated in Phase 6 if manual-approval policy remains enabled

## 23. Phase 7 Entry Conditions

1. Phase 6 order-policy document completed
2. risk-limit settings completed
3. pre-trade check implementation completed
4. kill-switch implementation completed
5. approval / block / audit logging completed
6. Paper Trading operated stably for at least 20 to 60 trading days
7. Paper Trading integrity errors are not recurring
8. operators can interpret candidate orders and risk blocks
9. sandbox or mock validation completed before real API attachment
10. manual approval process is ready

## 24. Live-Trading Prohibition Notice

Current Phase 6-1 is the live-trading policy documentation stage.
This stage does not place real orders.
This stage does not call real-order APIs.
This stage does not read real account balances.
This stage does not modify real-account order, fill, or position tables.
The project must not move to Phase 7 Micro Live before the full Phase 6 safety layer is completed.
