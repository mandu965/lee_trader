# Lee_trader_us DB Schema

> 문서 역할: `현재 기준 문서`
>
> 이 문서는 Phase 7까지 실제 운영/검증에 쓰이는 핵심 US 테이블과, Phase 8-1에서 설계만 추가한 BUY 자동화 후보 테이블을 함께 정리한다.

## Purpose

This document summarizes the main US-stock tables that matter operationally after Phase 7 and adds the Phase 8-1 proposed table design for limited BUY automation.

Notes:

- this is a design/reference document
- it is not a migration file
- Phase 8-1 proposed BUY tables are not applied in DB yet

## Baseline Documents Status

- `ARCHITECTURE.md` now exists in `Lee_trader_us`
- this `DB_SCHEMA.md` remains the schema-reference companion for Phase 8 work

## Core Tables By Phase

### Ranking And Recommendation

#### `recommend.us_stock_rank_daily`

Purpose:

- canonical daily US ranking snapshot

Key columns:

- `trade_date`
- `symbol`
- `rank_no`
- `recommend_grade`
- `total_score`
- `momentum_score`
- `relative_strength_score`
- `fundamental_score`
- `growth_score`
- `valuation_score`
- `risk_score`
- `score_detail_json`
- `reason_summary`
- `exclude_reason`
- `source`

PK / uniqueness:

- `(trade_date, symbol)`

Written by:

- Phase 3 ranking calculation

### Paper Trading

#### `paper.us_stock_paper_account`

Purpose:

- virtual account state for paper trading

#### `paper.us_stock_paper_order`

Purpose:

- paper order lifecycle

#### `paper.us_stock_paper_fill`

Purpose:

- paper fill records

#### `paper.us_stock_paper_position`

Purpose:

- open paper positions

#### `paper.us_stock_paper_account_snapshot`

Purpose:

- daily paper account valuation snapshot

### Live Safety

#### `risk.us_stock_live_kill_switch`

Purpose:

- scoped kill-switch state

#### `risk.us_stock_live_kill_switch_event_log`

Purpose:

- append-only kill-switch audit events

#### `risk.us_stock_live_daily_risk_usage`

Purpose:

- daily order/risk usage counters

#### `risk.us_stock_live_order_block_log`

Purpose:

- pre-trade blocked candidate audit log

#### `risk.us_stock_live_order_approval`

Purpose:

- approval request state before Micro Live handling

#### `risk.us_stock_live_order_approval_event_log`

Purpose:

- append-only approval lifecycle log

### Micro Live

#### `live.us_stock_micro_order_request`

Purpose:

- Micro Live order request rows

#### `live.us_stock_micro_order_event_log`

Purpose:

- append-only Micro order lifecycle events

#### `live.us_stock_micro_order_fill`

Purpose:

- normalized fill rows after broker/mock/sandbox sync

#### `live.us_stock_micro_reconciliation_result`

Purpose:

- internal vs broker reconciliation results

#### `live.us_stock_micro_reconciliation_event_log`

Purpose:

- reconciliation run-level events

## Phase 8-1 Proposed BUY Automation Tables

Phase 8-1 is design only. The following tables are proposed for later implementation.

### `trade.us_buy_candidate_log`

Purpose:

- store every symbol entering the BUY evaluation funnel
- preserve the ranking snapshot and early filter context even if the symbol is later blocked

Recommended key columns:

- `candidate_id VARCHAR(120) NOT NULL`
- `trade_date DATE NOT NULL`
- `account_id VARCHAR(100)`
- `automation_mode VARCHAR(20) NOT NULL`
- `ranking_source VARCHAR(50) NOT NULL`
- `symbol VARCHAR(20) NOT NULL`
- `company_name VARCHAR(200)`
- `sector VARCHAR(100)`
- `rank_no INTEGER`
- `recommend_grade VARCHAR(30)`
- `total_score NUMERIC(24,6)`
- `score_detail_json JSONB`
- `price_ref NUMERIC(24,6)`
- `candidate_amount_usd NUMERIC(24,6)`
- `candidate_status VARCHAR(30) NOT NULL`
- `filter_stage VARCHAR(50) NOT NULL`
- `filter_reason_code VARCHAR(100)`
- `filter_reason_detail TEXT`
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

Recommended PK:

- `PRIMARY KEY (candidate_id)`

Recommended uniqueness:

- `UNIQUE (trade_date, automation_mode, symbol, filter_stage)`

Stored when:

- every time the BUY evaluation job includes a symbol in the candidate funnel

Retention:

- keep at least 1 year for audit and threshold tuning

Writer:

- future Phase 8 SHADOW/PAPER BUY evaluation runner

### `trade.us_buy_decision_log`

Purpose:

- store final allow/block outcome for each symbol after all rule checks

Recommended key columns:

- `decision_id VARCHAR(120) NOT NULL`
- `trade_date DATE NOT NULL`
- `account_id VARCHAR(100)`
- `automation_mode VARCHAR(20) NOT NULL`
- `symbol VARCHAR(20) NOT NULL`
- `candidate_id VARCHAR(120)`
- `decision VARCHAR(20) NOT NULL`
- `severity VARCHAR(20) NOT NULL`
- `decision_reason_code VARCHAR(100) NOT NULL`
- `decision_reason_detail TEXT`
- `rule_tags JSONB`
- `rank_no INTEGER`
- `recommend_grade VARCHAR(30)`
- `total_score NUMERIC(24,6)`
- `price_ref NUMERIC(24,6)`
- `planned_order_amount_usd NUMERIC(24,6)`
- `cooldown_until DATE`
- `requires_manual_review BOOLEAN DEFAULT TRUE`
- `report_group VARCHAR(50)`
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

Recommended PK:

- `PRIMARY KEY (decision_id)`

Recommended uniqueness:

- `UNIQUE (trade_date, automation_mode, symbol)`

Stored when:

- final BUY decision is produced

Retention:

- keep at least 2 years because final decision logs are higher-value audit artifacts

Writer:

- future Phase 8 SHADOW/PAPER BUY decision module

### `trade.us_risk_guard_log`

Purpose:

- store market-wide and portfolio-wide guard evaluation so symbol blocks can be explained in context

Recommended key columns:

- `guard_log_id VARCHAR(120) NOT NULL`
- `trade_date DATE NOT NULL`
- `account_id VARCHAR(100)`
- `automation_mode VARCHAR(20) NOT NULL`
- `guard_scope VARCHAR(30) NOT NULL`
- `guard_name VARCHAR(100) NOT NULL`
- `guard_status VARCHAR(20) NOT NULL`
- `severity VARCHAR(20) NOT NULL`
- `metric_value NUMERIC(24,6)`
- `threshold_value NUMERIC(24,6)`
- `reason_code VARCHAR(100)`
- `reason_detail TEXT`
- `raw_payload JSONB`
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

Recommended PK:

- `PRIMARY KEY (guard_log_id)`

Recommended uniqueness:

- `UNIQUE (trade_date, automation_mode, guard_scope, guard_name, account_id)`

Stored when:

- each global risk guard is evaluated

Retention:

- keep at least 1 year

Writer:

- future Phase 8 BUY risk-guard evaluation module

### `trade.us_paper_order`

Purpose:

- store internal Phase 8 PAPER-mode virtual BUY records without touching any broker path

Recommended key columns:

- `paper_order_id VARCHAR(120) NOT NULL`
- `trade_date DATE NOT NULL`
- `account_id VARCHAR(100)`
- `automation_mode VARCHAR(20) NOT NULL`
- `symbol VARCHAR(20) NOT NULL`
- `side VARCHAR(10) NOT NULL`
- `paper_order_qty NUMERIC(24,6)`
- `paper_order_price NUMERIC(24,6)`
- `paper_order_amount NUMERIC(24,6)`
- `assumed_fill_price NUMERIC(24,6)`
- `assumed_fill_status VARCHAR(30)`
- `source_decision_id VARCHAR(120)`
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

Recommended PK:

- `PRIMARY KEY (paper_order_id)`

Recommended uniqueness:

- `UNIQUE (trade_date, automation_mode, symbol, side)`

Stored when:

- Phase 8 PAPER skeleton creates a virtual BUY record

Retention:

- keep at least 1 year for audit and comparison with future paper/live paths

Writer:

- Phase 8 PAPER-only BUY automation skeleton

## Phase 8-3 Proposed Report Tables

### `trade.us_buy_daily_report`

Purpose:

- store daily BUY automation report snapshots after SHADOW/PAPER review

Key columns:

- `report_id`
- `trade_date`
- `automation_mode`
- `report_type`
- `source_json_path`
- `summary_json`
- `created_at`
- `updated_at`

Uniqueness:

- `UNIQUE (trade_date, automation_mode, report_type)`

### `trade.us_paper_performance_snapshot`

Purpose:

- store daily PAPER performance snapshots derived from virtual BUY orders only

Key columns:

- `snapshot_id`
- `trade_date`
- `paper_order_id`
- `symbol`
- `benchmark_symbol`
- `latest_price`
- `current_value`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `benchmark_return_pct`
- `excess_return_pct`
- `status`
- `summary_json`
- `created_at`
- `updated_at`

Uniqueness:

- `UNIQUE (trade_date, paper_order_id)`

## Proposed PostgreSQL DDL Sketches

These are design sketches only. Do not apply them automatically in Phase 8-1.

```sql
CREATE TABLE trade.us_buy_candidate_log (
    candidate_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    account_id VARCHAR(100),
    automation_mode VARCHAR(20) NOT NULL,
    ranking_source VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    company_name VARCHAR(200),
    sector VARCHAR(100),
    rank_no INTEGER,
    recommend_grade VARCHAR(30),
    total_score NUMERIC(24,6),
    score_detail_json JSONB,
    price_ref NUMERIC(24,6),
    candidate_amount_usd NUMERIC(24,6),
    candidate_status VARCHAR(30) NOT NULL,
    filter_stage VARCHAR(50) NOT NULL,
    filter_reason_code VARCHAR(100),
    filter_reason_detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, automation_mode, symbol, filter_stage)
);

CREATE TABLE trade.us_buy_decision_log (
    decision_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    account_id VARCHAR(100),
    automation_mode VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    candidate_id VARCHAR(120),
    decision VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    decision_reason_code VARCHAR(100) NOT NULL,
    decision_reason_detail TEXT,
    rule_tags JSONB,
    rank_no INTEGER,
    recommend_grade VARCHAR(30),
    total_score NUMERIC(24,6),
    price_ref NUMERIC(24,6),
    planned_order_amount_usd NUMERIC(24,6),
    cooldown_until DATE,
    requires_manual_review BOOLEAN DEFAULT TRUE,
    report_group VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, automation_mode, symbol)
);

CREATE TABLE trade.us_risk_guard_log (
    guard_log_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    account_id VARCHAR(100),
    automation_mode VARCHAR(20) NOT NULL,
    guard_scope VARCHAR(30) NOT NULL,
    guard_name VARCHAR(100) NOT NULL,
    guard_status VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric_value NUMERIC(24,6),
    threshold_value NUMERIC(24,6),
    reason_code VARCHAR(100),
    reason_detail TEXT,
    raw_payload JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, automation_mode, guard_scope, guard_name, account_id)
);

CREATE TABLE trade.us_paper_order (
    paper_order_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    account_id VARCHAR(100),
    automation_mode VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    paper_order_qty NUMERIC(24,6),
    paper_order_price NUMERIC(24,6),
    paper_order_amount NUMERIC(24,6),
    assumed_fill_price NUMERIC(24,6),
    assumed_fill_status VARCHAR(30),
    source_decision_id VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, automation_mode, symbol, side)
);

CREATE TABLE trade.us_buy_daily_report (
    report_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    automation_mode VARCHAR(20) NOT NULL,
    report_type VARCHAR(30) NOT NULL,
    source_json_path TEXT,
    summary_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, automation_mode, report_type)
);

CREATE TABLE trade.us_paper_performance_snapshot (
    snapshot_id VARCHAR(120) PRIMARY KEY,
    trade_date DATE NOT NULL,
    paper_order_id VARCHAR(120) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    benchmark_symbol VARCHAR(20) NOT NULL,
    latest_price NUMERIC(24,6),
    current_value NUMERIC(24,6),
    unrealized_pnl NUMERIC(24,6),
    unrealized_pnl_pct NUMERIC(24,6),
    benchmark_return_pct NUMERIC(24,6),
    excess_return_pct NUMERIC(24,6),
    status VARCHAR(40) NOT NULL,
    summary_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (trade_date, paper_order_id)
);
```

## Design Notes

### Why Separate Candidate And Decision Logs

`candidate_log` and `decision_log` should stay separate because:

- the candidate funnel may include multiple filter stages per symbol
- operators need to see where a symbol dropped out
- final decision should stay one-row-per-symbol-per-day-per-mode

### Why Reuse Existing Paper Tables

Phase 5 already has stable paper-trading lifecycle tables. Phase 8 currently keeps a lighter `trade.us_paper_order` skeleton log for BUY automation review, while the older paper-trading lifecycle remains available separately:

- `paper.us_stock_paper_order`
- `paper.us_stock_paper_fill`
- `paper.us_stock_paper_position`
- `paper.us_stock_paper_account_snapshot`

The open design question for later phases is whether to:

- keep `trade.us_paper_order` as a thin intent log, or
- map BUY automation PAPER decisions directly into `paper.us_stock_paper_order`

### Why No Live BUY Table Yet

Phase 8-1 is still pre-LIVE. A separate live BUY decision table can wait until:

- SHADOW logging is stable
- PAPER decision-to-order mapping is validated
- LIVE account state and release control are clearly defined

### Phase 8-3 Reporting Limitation

The Phase 8-3 report and performance layer is for operator review only.

- it does not write to broker state
- it does not read real account balances
- it does not decide LIVE readiness automatically

## Phase 8-5 Proposed Readiness Tables

### `trade.us_buy_readiness_report`

Purpose:

- store the daily readiness evaluation snapshot

Suggested columns:

- `report_id`
- `evaluation_date`
- `evaluation_period_days`
- `benchmark_symbol`
- `live_ready`
- `readiness_score`
- `decision`
- `reasons JSONB`
- `summary_json JSONB`
- `created_at`

### `trade.us_paper_performance_summary`

Purpose:

- store rolled-up PAPER performance windows such as 20d / 60d / 120d / ALL

Suggested columns:

- `summary_id`
- `evaluation_date`
- `period_label`
- `benchmark_symbol`
- `paper_order_count`
- `unique_symbol_count`
- `total_return_pct`
- `benchmark_return_pct`
- `excess_return_pct`
- `win_rate`
- `max_drawdown_pct`
- `data_missing_rate`
- `summary_json`
- `created_at`

### `trade.us_live_promotion_check`

Purpose:

- store the promotion-policy decision snapshot separately from the detailed report body

Suggested columns:

- `check_id`
- `evaluation_date`
- `benchmark_symbol`
- `live_ready`
- `readiness_score`
- `manual_approval_required`
- `reasons JSONB`
- `policy_snapshot JSONB`
- `operational_snapshot JSONB`
- `created_at`

### Phase 8-5 DDL

- see [phase8_5_live_readiness_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_5_live_readiness_tables.sql)

### Phase 8-5 Limitation

- these tables are design / manual migration targets only
- readiness evaluation in the current phase is file/report based
- DB migration is not auto-applied

## Phase 8-6 Proposed SELL / Exit Tables

### `trade.us_paper_position`

Purpose:

- canonical Paper position state used for SELL / Exit evaluation

Suggested columns:

- `paper_position_id`
- `account_id`
- `symbol`
- `entry_trade_date`
- `entry_price`
- `quantity`
- `remaining_quantity`
- `avg_entry_price`
- `latest_price`
- `highest_price_since_entry`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `holding_days`
- `status`
- `exit_reason`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `UNIQUE (account_id, symbol, entry_trade_date)`

### `trade.us_paper_position_snapshot`

Purpose:

- daily Paper position mark-to-market snapshot for exit review and report reconstruction

Suggested columns:

- `snapshot_id`
- `snapshot_date`
- `paper_position_id`
- `symbol`
- `latest_price`
- `remaining_quantity`
- `highest_price_since_entry`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `holding_days`
- `status`
- `created_at`

Suggested uniqueness:

- `UNIQUE (snapshot_date, paper_position_id)`

### `trade.us_sell_decision_log`

Purpose:

- final SELL / PARTIAL_SELL / HOLD / REVIEW_REQUIRED decision log

Suggested columns:

- `sell_decision_id`
- `trade_date`
- `account_id`
- `automation_mode`
- `paper_position_id`
- `symbol`
- `decision`
- `sell_action`
- `sell_ratio`
- `sell_quantity`
- `exit_reason`
- `review_required`
- `applied_rules JSONB`
- `decision_reason_detail`
- `created_at`

Suggested uniqueness:

- `UNIQUE (trade_date, automation_mode, paper_position_id)`

### `trade.us_sell_signal_log`

Purpose:

- lower-level rule trigger evidence before the final sell decision is resolved

Suggested columns:

- `sell_signal_id`
- `trade_date`
- `paper_position_id`
- `symbol`
- `rule_name`
- `rule_result`
- `metric_value`
- `threshold_value`
- `severity`
- `detail`
- `created_at`

### `trade.us_paper_sell_order`

Purpose:

- Paper-only SELL action artifact, separate from any future real SELL path

Suggested columns:

- `paper_sell_order_id`
- `trade_date`
- `paper_position_id`
- `symbol`
- `side`
- `sell_ratio`
- `sell_quantity`
- `sell_price_ref`
- `assumed_fill_status`
- `source_sell_decision_id`
- `created_at`
- `updated_at`

### Phase 8-6 Design Notes

- these tables are design-only in this phase
- no migration is auto-applied
- Paper position state should become the source for stop-loss, trailing-stop, and holding-day evaluation
- real SELL lifecycle tables should wait until Paper SELL validation is mature

## Phase 8-7 SELL Skeleton Logging And Snapshot Contract

Phase 8-7 keeps the same proposed `trade.*` SELL tables and adds an implemented JSON-first logger.

Current behavior:

- JSON output is always written under the SELL output directory
- DB writes are attempted only if the relevant `trade.*` table already exists
- missing tables fall back gracefully to file-only logging

### `trade.us_sell_decision_log`

Current persisted fields include:

- `trade_date`
- `automation_mode`
- `paper_position_id`
- `symbol`
- `decision`
- `sell_action`
- `sell_ratio`
- `sell_quantity`
- `exit_reason`
- `review_required`
- `applied_rules`
- `latest_price`
- `avg_entry_price`
- `unrealized_pnl_pct`
- `realized_paper_pnl`
- `error_message`

### `trade.us_sell_signal_log`

Current persisted fields include one row per applied rule:

- `trade_date`
- `paper_position_id`
- `symbol`
- `rule_name`
- `rule_result`
- `metric_value`
- `threshold_value`
- `severity`
- `detail`

### `trade.us_paper_sell_order`

Phase 8-7 uses this as a Paper-only SELL artifact table.

- no broker order id is created
- `assumed_fill_status` is synthetic
- latest close/reference price is used as the fill assumption

### `trade.us_paper_position_snapshot`

Phase 8-7 snapshot rows store:

- latest marked price
- remaining quantity
- high-water mark
- unrealized pnl
- unrealized pnl pct
- holding days
- status
- data-quality flags

DDL reference:

- see [phase8_7_sell_automation_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_7_sell_automation_tables.sql)

## Phase 8-8 Trade Orchestration Tables

### `trade.us_trade_orchestration_log`

Purpose:

- store one run-level orchestration summary per `trade_date + mode`

Key fields:

- execution time
- sell executed flag
- buy executed flag
- report generated flag
- success
- fail-safe triggered
- conflict summary
- final action summary
- error message

### `trade.us_trade_conflict_log`

Purpose:

- store per-symbol BUY conflict guard results

Key fields:

- trade date
- mode
- symbol
- buy allowed after conflict check
- conflict reasons
- related position id
- related sell signal
- cooldown until

### `trade.us_integrated_daily_report`

Purpose:

- store integrated daily report metadata/body snapshot

Key fields:

- trade date
- mode
- report type
- source json path
- summary json

### BUY Decision Log Phase 8-8 Additions

Phase 8-8 extends the proposed BUY decision log shape with:

- `conflict_checked`
- `conflict_blocked`
- `conflict_reasons JSONB`
- `related_position_id`
- `related_sell_signal JSONB`

DDL reference:

- see [phase8_8_trade_orchestration_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_8_trade_orchestration_tables.sql)

## Phase 8-9 Scheduler Stability Tables

### `trade.us_trade_scheduler_run_log`

Purpose:

- store one scheduler-run summary per `trade_date + mode + job_name`

Suggested fields:

- job status
- guard result
- health result
- warnings
- errors
- pipeline should fail

### `trade.us_trade_scheduler_health_check`

Purpose:

- persist health-check snapshots separately from the run summary

### `trade.us_trade_scheduler_lock_log`

Purpose:

- preserve duplicate-run / stale-lock audit visibility

DDL reference:

- see [phase8_9_scheduler_stability_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_9_scheduler_stability_tables.sql)

## Phase 8-10 Dashboard Read Model Design

Phase 8-10 is dashboard-design only. No dashboard-specific migration is required in this phase.

Preferred rule:

- reuse existing `trade.*` logs and snapshots first
- add dashboard-specific persistence only if file-based assembly becomes too slow or inconsistent

### Existing Tables Used By The Dashboard

Primary dashboard inputs:

- `trade.us_buy_decision_log`
- `trade.us_risk_guard_log`
- `trade.us_paper_order`
- `trade.us_sell_decision_log`
- `trade.us_sell_signal_log`
- `trade.us_paper_sell_order`
- `trade.us_paper_position`
- `trade.us_paper_position_snapshot`
- `trade.us_trade_orchestration_log`
- `trade.us_trade_conflict_log`
- `trade.us_integrated_daily_report`
- `trade.us_trade_scheduler_run_log`
- `trade.us_trade_scheduler_health_check`
- `trade.us_buy_readiness_report`
- `trade.us_paper_performance_summary`

### Optional Future Read Models

#### `trade.us_dashboard_daily_snapshot`

Purpose:

- persist one assembled dashboard body per `trade_date`

Suggested columns:

- `snapshot_id`
- `trade_date`
- `mode`
- `summary_json`
- `created_at`
- `updated_at`

#### `trade.us_dashboard_section_status`

Purpose:

- persist section-level assembly health and missing-data status

Suggested columns:

- `status_id`
- `trade_date`
- `section_name`
- `status`
- `warning_count`
- `error_count`
- `detail_json`
- `created_at`

### Dashboard Schema Notes

- these are optional future read models, not required for Phase 8-10
- file-based dashboard output can be the first implementation target
- dashboard persistence must remain read-only relative to trading decisions

## Phase 8-11 Dashboard Report Table

Phase 8-11 keeps file output as the primary artifact and adds an optional DDL target for later DB persistence.

### `trade.us_paper_dashboard_report`

Purpose:

- store one assembled dashboard payload per `trade_date`

Suggested fields:

- `dashboard_report_id`
- `trade_date`
- `report_type`
- `report_status`
- `report_json`
- `generated_at`
- `created_at`

DDL reference:

- see [phase8_11_dashboard_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_11_dashboard_tables.sql)

Current note:

- Phase 8-11 implementation writes files only
- DB migration is still manual and is not auto-applied

## Phase 8-12 Dashboard Scheduler / Notification Tables

### `trade.us_dashboard_scheduler_log`

Purpose:

- optional future persistence for dashboard scheduler step status

Suggested fields:

- `dashboard_scheduler_log_id`
- `trade_date`
- `dashboard_status`
- `report_paths JSONB`
- `warnings JSONB`
- `errors JSONB`
- `created_at`

### `trade.us_dashboard_notification_payload`

Purpose:

- optional future persistence for generated notification payload artifacts

Suggested fields:

- `notification_payload_id`
- `trade_date`
- `notification_format`
- `notification_payload JSONB`
- `notification_text`
- `created_at`

DDL reference:

- see [phase8_12_dashboard_scheduler_notification_tables.sql](/d:/ai/lee_trader/sql/lee_trader_us/phase8_12_dashboard_scheduler_notification_tables.sql)

## Phase 8-13 Notification Adapter Tables

Phase 8-13 is design-only. The following tables describe future notification-adapter auditability and approval tracking.

### `trade.us_notification_event_log`

Purpose:

- store one normalized notification event per trade date and payload type

Suggested fields:

- `notification_event_id`
- `trade_date`
- `message_type`
- `severity`
- `mode`
- `paper_trading_only`
- `approval_required`
- `approval_status`
- `payload_json`
- `message_text`
- `error_message`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(trade_date, message_type, mode)`

### `trade.us_notification_delivery_log`

Purpose:

- store one channel-level dry-run or future delivery result per notification event

Suggested fields:

- `delivery_log_id`
- `notification_event_id`
- `trade_date`
- `channel`
- `delivery_mode`
- `delivery_status`
- `severity`
- `payload_json`
- `message_text`
- `error_message`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(notification_event_id, channel, delivery_mode)`

### `trade.us_notification_approval_log`

Purpose:

- store manual-approval lifecycle records for notification delivery

Suggested fields:

- `approval_log_id`
- `notification_event_id`
- `trade_date`
- `approval_required`
- `approval_status`
- `approver`
- `approved_at`
- `comment`
- `expires_at`
- `created_at`
- `updated_at`

Notes:

- manual approval here is for notification delivery only
- it must not be confused with LIVE trading approval
- no migration or runtime writer is added in this phase
