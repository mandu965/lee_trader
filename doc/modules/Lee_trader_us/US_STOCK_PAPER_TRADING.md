# US Stock Paper Trading

> 문서 역할: `상세 참고 문서`
>
> Phase 5 paper-trading 구조와 운영 검증 흐름을 자세히 설명하는 문서다.

> 상태 메모: 2026-05-22 기준 이 문서는 현재 운영 원칙과 가장 가까운 핵심 참고 문서 중 하나다.  
> US는 `paper-only` 유지가 기본이며, 실제 운영 검증은 이 문서의 paper lifecycle 문맥을 우선 기준으로 본다.

## 1. Purpose

Phase 5 builds a paper-only virtual trading path for the US ranking engine.

- Phase 5-1: account, order, fill, position, and snapshot tables
- Phase 5-2: paper order generation only
- Phase 5-3: paper fill simulation and paper account/position updates

This path is not real trading.
This path is not paper broker integration.
This path must not call any broker order API.

## 2. Boundaries

Paper Trading:

- uses virtual cash
- writes only `paper.us_stock_*`
- tracks simulated orders, fills, positions, and PnL

Forward Test:

- tracks recommendation outcomes only
- has no cash, no position sizing, and no fill lifecycle

Live Trading:

- not implemented here
- must remain separated from the US paper path
- must remain separated from the Korean live-trading path

## 3. Tables

`paper.us_stock_paper_account`

- virtual account state
- cash, equity, realized PnL, unrealized PnL

`paper.us_stock_paper_order`

- virtual order requests
- Phase 5-2 writes `CREATED` or `REJECTED`

`paper.us_stock_paper_fill`

- simulated fill rows
- Phase 5-3 writes `FILLED` rows here

`paper.us_stock_paper_position`

- virtual holdings by `account_id + symbol`
- updated only by fill simulation

`paper.us_stock_paper_account_snapshot`

- daily account snapshots
- reserved for Phase 5-4 reporting

## 4. Config

Config file:

- `config/us_stock_paper_trading.yaml`

Default profile:

- `US_PAPER_RULE_V1`
- base currency: `USD`
- initial cash: `100000`

Important policy fields:

- `selection_rule`
- `buy_grades`
- `max_positions`
- `max_position_weight`
- `max_sector_weight`
- `min_cash_weight`
- `max_daily_new_buys`
- `allow_fractional_shares`
- `commission_per_trade`
- `slippage_bps`
- `real_order_blocked`

## 5. ENV

Key paper ENV values:

- `US_PAPER_TRADING_ENABLED=false`
- `US_PAPER_ACCOUNT_ID=US_PAPER_RULE_V1`
- `US_PAPER_INITIAL_CASH=100000`
- `US_PAPER_BASE_CURRENCY=USD`
- `US_PAPER_MAX_POSITIONS=20`
- `US_PAPER_MAX_POSITION_WEIGHT=0.10`
- `US_PAPER_MAX_SECTOR_WEIGHT=0.30`
- `US_PAPER_MIN_CASH_WEIGHT=0.05`
- `US_PAPER_MAX_DAILY_NEW_BUYS=5`
- `US_PAPER_ALLOW_FRACTIONAL_SHARES=true`
- `US_PAPER_COMMISSION_PER_TRADE=0`
- `US_PAPER_SLIPPAGE_BPS=5`
- `US_PAPER_REAL_ORDER_BLOCKED=true`
- `US_PAPER_OUTPUT_DIR=outputs/us_stock_paper_trading`
- `US_PAPER_MIN_ORDER_AMOUNT=100`

`US_PAPER_REAL_ORDER_BLOCKED=true` must remain the default.

## 6. Account Bootstrap

Commands:

```powershell
python scripts/init_us_stock_paper_account.py --account-id US_PAPER_RULE_V1 --initial-cash 100000 --dry-run
python scripts/init_us_stock_paper_account.py --account-id US_PAPER_RULE_V1 --initial-cash 100000
python scripts/init_us_stock_paper_account.py --account-id US_PAPER_RULE_V1 --reset
```

Rules:

- no DB write in `--dry-run`
- rerun without `--reset` must not wipe the account
- `--reset` touches only `paper.us_stock_*`

## 7. Phase 5-2 Order Generation

Scope:

- read US rank snapshots
- read paper account state
- read paper positions
- create BUY and SELL paper orders only

Phase 5-2 does not:

- create fills
- update positions
- update cash balance
- call broker APIs

BUY rules:

- `recommend_grade in ('STRONG_BUY', 'BUY')`
- `rank_no <= max_rank_no`
- `recommend_grade <> 'EXCLUDE'`
- valid `total_score`
- valid ranking data status

SELL rules:

- open position exits top-N
- grade falls to `HOLD` or `EXCLUDE`
- ranking row disappears
- ranking data becomes invalid

Order statuses used in Phase 5-2:

- `CREATED`
- `REJECTED`
- `ERROR`

Reject codes:

- `already_ordered`
- `already_target_weight`
- `max_positions_reached`
- `insufficient_cash`
- `below_min_order_amount`
- `missing_order_price`
- `invalid_order_price`
- `sector_weight_limit`
- `position_weight_limit`
- `qty_zero`
- `error`

Commands:

```powershell
python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-12 --account-id US_PAPER_RULE_V1 --dry-run
python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-12 --account-id US_PAPER_RULE_V1 --side BUY --dry-run
python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-12 --account-id US_PAPER_RULE_V1 --side SELL --dry-run
python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-12 --account-id US_PAPER_RULE_V1
```

## 8. Phase 5-3 Fill Simulation

Scope:

- read `CREATED` rows from `paper.us_stock_paper_order`
- simulate fills into `paper.us_stock_paper_fill`
- update `paper.us_stock_paper_position`
- update `paper.us_stock_paper_account`
- keep all writes inside the `paper` schema only

### Fill Basis

- `fill_date = next US trading day after order.trade_date`
- `fill_price = next-trading-day close`
- BUY fill price uses `close * (1 + slippage_bps / 10000)`
- SELL fill price uses `close * (1 - slippage_bps / 10000)`
- `commission = US_PAPER_COMMISSION_PER_TRADE`

This avoids same-day look-ahead.

### BUY Handling

- cash check uses `filled_amount + commission`
- position cost basis includes commission
- `avg_price = cost_amount / qty`
- account cash decreases immediately after fill

### SELL Handling

- sell quantity must not exceed open position quantity
- `realized_pnl = (filled_amount - commission) - (avg_price * filled_qty)`
- full sell sets the position to `CLOSED`
- account cash increases by net proceeds

### Order Status Transitions

- `CREATED -> FILLED`
- `CREATED -> REJECTED`
- `CREATED -> ERROR`

### Fill Reject Codes

- `missing_fill_price`
- `invalid_fill_price`
- `insufficient_cash_at_fill`
- `insufficient_position_qty`
- `account_not_active`
- `order_not_created`
- `unsupported_order_type`
- `limit_condition_not_met`
- `duplicate_fill`
- `position_not_found`
- `cash_negative_after_fill`
- `position_negative_after_fill`
- `real_order_blocked_check_failed`
- `error`

### Duplicate Protection

- only `order.status = 'CREATED'` is fillable
- a row with an existing `paper_fill.paper_order_id` is skipped
- a `FILLED` order must not be filled again

### Transaction Rule

One simulated fill must commit or roll back as one unit:

1. insert `paper_fill`
2. update `paper_order`
3. update or upsert `paper_position`
4. update `paper_account`

### Integrity Rules

- `cash_balance >= 0`
- open positions must have `qty > 0`
- closed positions must have `qty = 0`
- filled orders must have matching fill rows
- `equity_value = cash_balance + market_value`
- `total_pnl = realized_pnl + unrealized_pnl`

### Commands

```powershell
python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-13 --account-id US_PAPER_RULE_V1 --dry-run
python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-13 --account-id US_PAPER_RULE_V1 --side BUY --dry-run
python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-13 --account-id US_PAPER_RULE_V1 --side SELL --dry-run
python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-13 --account-id US_PAPER_RULE_V1
```

## 9. Basic SQL

Account state:

```sql
SELECT
    account_id,
    account_name,
    initial_cash,
    cash_balance,
    market_value,
    equity_value,
    realized_pnl,
    unrealized_pnl,
    total_pnl,
    status
FROM paper.us_stock_paper_account
WHERE account_id = 'US_PAPER_RULE_V1';
```

Open positions:

```sql
SELECT
    symbol,
    qty,
    avg_price,
    last_price,
    market_value,
    unrealized_pnl,
    unrealized_pnl_pct,
    status
FROM paper.us_stock_paper_position
WHERE account_id = 'US_PAPER_RULE_V1'
  AND status = 'OPEN'
ORDER BY market_value DESC;
```

Order history:

```sql
SELECT
    trade_date,
    symbol,
    side,
    order_type,
    order_qty,
    order_price,
    status,
    reason,
    reject_reason
FROM paper.us_stock_paper_order
WHERE account_id = 'US_PAPER_RULE_V1'
ORDER BY trade_date DESC, created_at DESC;
```

Fill history:

```sql
SELECT
    trade_date,
    symbol,
    side,
    filled_qty,
    filled_price,
    filled_amount,
    commission,
    fill_status
FROM paper.us_stock_paper_fill
WHERE account_id = 'US_PAPER_RULE_V1'
ORDER BY trade_date DESC, created_at DESC;
```

Snapshots:

```sql
SELECT
    snapshot_date,
    cash_balance,
    market_value,
    equity_value,
    total_pnl,
    total_pnl_pct,
    daily_return_pct,
    excess_return_vs_spy,
    excess_return_vs_qqq,
    position_count
FROM paper.us_stock_paper_account_snapshot
WHERE account_id = 'US_PAPER_RULE_V1'
ORDER BY snapshot_date DESC;
```

## 10. Status Definitions

Account:

- `ACTIVE`
- `PAUSED`
- `CLOSED`
- `ERROR`

Order:

- `CREATED`
- `VALIDATED`
- `REJECTED`
- `FILLED`
- `PARTIALLY_FILLED`
- `CANCELED`
- `ERROR`

Fill:

- `FILLED`
- `PARTIAL`
- `REJECTED`
- `ERROR`

Position:

- `OPEN`
- `CLOSED`
- `ERROR`

## 11. Safety

- Paper Trading must not call real broker order APIs.
- Paper Trading must not import Alpaca/KIS real-order execution modules.
- Paper Trading must not write any real account, real order, or Korean live-trading tables.
- `US_PAPER_REAL_ORDER_BLOCKED=true` is mandatory.
- keep the log line: `[SAFETY] Paper trading only. Real order APIs are blocked.`
- keep the log line: `[SAFETY] Paper fill simulation only. Real order APIs are blocked.`

## 12. Next Step

Phase 5-4 will build:

- daily paper account snapshots
- paper performance summary
- progress and PnL reporting
- review artifacts only, still separate from live trading

## 13. Phase 5-4 Snapshot And Reporting

Scope:

- value OPEN positions on a snapshot date
- refresh paper account valuation fields
- upsert `paper.us_stock_paper_account_snapshot`
- generate console, markdown, and csv review reports

Position valuation:

- `last_price = snapshot-date close`
- optional fallback: latest previous close if `US_PAPER_USE_PREVIOUS_CLOSE_IF_MISSING=true`
- `market_value = qty * last_price`
- `unrealized_pnl = market_value - cost_amount`
- `unrealized_pnl_pct = unrealized_pnl / cost_amount`

Account valuation:

- `market_value = sum(OPEN position market_value)`
- `equity_value = cash_balance + market_value`
- `unrealized_pnl = sum(OPEN position unrealized_pnl)`
- `total_pnl = realized_pnl + unrealized_pnl`
- `total_pnl_pct = total_pnl / initial_cash`

Daily return:

- `daily_return_pct = (today_equity_value - previous_equity_value) / previous_equity_value`
- null when there is no previous snapshot

Benchmark comparison:

- benchmark start uses the earliest existing snapshot date, then first fill date as fallback
- `spy_return_pct` and `qqq_return_pct` are optional
- benchmark failure must not block snapshot creation

Snapshot write key:

- `account_id + snapshot_date`

Commands:

```powershell
python scripts/update_us_stock_paper_snapshot.py --snapshot-date 2026-05-14 --account-id US_PAPER_RULE_V1 --dry-run
python scripts/update_us_stock_paper_snapshot.py --snapshot-date 2026-05-14 --account-id US_PAPER_RULE_V1
python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console
python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format markdown
python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format csv
```

Report outputs:

- console summary
- markdown report under `US_PAPER_REPORT_OUTPUT_DIR`
- snapshot, position, order, and fill csv files under `US_PAPER_REPORT_OUTPUT_DIR`

Scheduler notes:

- `US_PAPER_AUTO_SNAPSHOT_ENABLED=false` by default
- `US_PAPER_AUTO_REPORT_ENABLED=false` by default
- keep paper scheduler wiring optional and fully separated from Korean live trading

Bridge to Phase 5-5:

- Phase 5-5 will use snapshot history and paper account state for rebalancing and operating checks

## 14. Phase 5-5 Rebalancing And Operating Validation

Purpose:

- define a repeatable paper-only rebalance policy
- review SELL-first and BUY-follow-up actions before order generation
- validate account, order, fill, position, and snapshot integrity after repeated runs

Rebalance policy:

- rebalance frequency: `DAILY` by default, optional `WEEKLY`
- BUY universe:
  - `recommend_grade in ('STRONG_BUY', 'BUY')`
  - `rank_no <= 20`
  - valid ranking status only
- SELL universe:
  - rank exit
  - grade downgrade to `HOLD` or `EXCLUDE`
  - invalid ranking data such as `ERROR` or `MISSING_PRICE_FEATURE`
- risk guards:
  - max positions
  - max position weight
  - max sector weight
  - min cash weight
  - min rebalance amount
  - min weight difference
  - optional same-day rebuy block

Config additions:

- `rebalance.enabled`
- `rebalance.frequency`
- `rebalance.sell_first`
- `rebalance.allow_rebuy_same_day`
- `rebalance.min_rebalance_amount`
- `rebalance.min_weight_diff`
- `rebalance.full_sell_on_rank_exit`
- `rebalance.full_sell_on_grade_downgrade`

Environment additions:

- `US_PAPER_REBALANCE_ENABLED=false`
- `US_PAPER_REBALANCE_FREQUENCY=DAILY`
- `US_PAPER_REBALANCE_SELL_FIRST=true`
- `US_PAPER_REBALANCE_ALLOW_REBUY_SAME_DAY=false`
- `US_PAPER_REBALANCE_MIN_AMOUNT=100`
- `US_PAPER_REBALANCE_MIN_WEIGHT_DIFF=0.02`
- `US_PAPER_REBALANCE_FULL_SELL_ON_RANK_EXIT=true`
- `US_PAPER_REBALANCE_FULL_SELL_ON_GRADE_DOWNGRADE=true`
- `US_PAPER_VALIDATION_ENABLED=true`
- `US_PAPER_VALIDATION_FAIL_ON_ERROR=false`
- `US_PAPER_SCHEDULER_ENABLED=false`
- `US_PAPER_SCHEDULER_ACCOUNT_ID=US_PAPER_RULE_V1`
- `US_PAPER_SCHEDULER_RUN_REBALANCE_PLAN=true`
- `US_PAPER_SCHEDULER_GENERATE_ORDERS=false`
- `US_PAPER_SCHEDULER_SIMULATE_FILLS=false`
- `US_PAPER_SCHEDULER_UPDATE_SNAPSHOT=false`
- `US_PAPER_SCHEDULER_VALIDATE=true`
- `US_PAPER_SCHEDULER_REPORT=true`

Rebalance planning:

```powershell
python scripts/plan_us_stock_paper_rebalance.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1 --dry-run
python scripts/plan_us_stock_paper_rebalance.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1 --format markdown
```

This script:

- reads the latest paper account and OPEN positions
- reads the ranking snapshot for the requested trade date
- builds SELL candidates first
- builds BUY candidates after cash/risk review
- does not create any order row

Order-generation linkage:

- `generate_us_stock_paper_orders.py` now uses the same rebalance policy
- `sell_first`, `min_rebalance_amount`, `min_weight_diff`, and same-day rebuy blocking are enforced before writing paper orders
- writes still target `paper.us_stock_paper_order` only

Validation:

```powershell
python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1
python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --snapshot-date 2026-05-15
python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format markdown
```

Validation scope:

- account:
  - existence
  - `status = ACTIVE`
  - non-negative cash
  - `equity_value = cash_balance + market_value`
  - `total_pnl = realized_pnl + unrealized_pnl`
- position:
  - OPEN qty positive
  - CLOSED qty zero
  - `market_value = qty * last_price`
  - `unrealized_pnl = market_value - cost_amount`
  - max position and sector exposure warnings
- order:
  - FILLED orders must have fills
  - stale CREATED orders are warnings
  - REJECTED orders must keep `reject_reason`
- fill:
  - `filled_amount = filled_qty * filled_price`
- snapshot:
  - latest snapshot existence
  - `snapshot.equity_value = snapshot.cash_balance + snapshot.market_value`

Operating-status report:

- `report_us_stock_paper_trading.py` now includes:
  - last rebalance date
  - last order date
  - last fill date
  - last snapshot date
  - created/rejected/error order counts
  - open position count
  - cash weight
  - max position weight
  - max sector weight
  - validation warning/error counts

Daily paper pipeline:

```powershell
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-15
python scripts/plan_us_stock_paper_rebalance.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1
python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1
python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-16 --account-id US_PAPER_RULE_V1
python scripts/update_us_stock_paper_snapshot.py --snapshot-date 2026-05-16 --account-id US_PAPER_RULE_V1
python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1
python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console
```

Scheduler options:

- paper scheduler remains optional only
- default values keep order generation and fill simulation disabled
- scheduler wiring must remain separated from Korean live trading and all broker APIs

Phase 5 completion checklist:

- paper account/order/fill/position/snapshot tables exist
- paper config and env are documented
- ranking-based BUY and SELL paper orders can be generated
- created paper orders can be filled into paper positions and paper account
- daily snapshots and benchmark-relative review reports can be produced
- rebalance plan and operating validation scripts are available
- real-order blocking remains enforced across all paper scripts

Phase 6 entry conditions:

- Paper Trading runs stably for at least 20 to 60 trading days
- order generation -> fill simulation -> snapshot -> report repeats cleanly
- recurring integrity errors are not observed
- paper performance does not diverge severely from backtest and forward-test evidence
- concentration, volatility, and loss behavior are understood well enough for explicit risk-policy design

Safety:

- Phase 5 Paper Trading results are pre-live validation artifacts only.
- Phase 5 results must not trigger real trading or auto-trading by themselves.
- Phase 6 must define separate order policy, risk limits, kill switch, and approval flow before any limited live-trading review.
