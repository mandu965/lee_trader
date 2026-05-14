# Paper Trading Dashboard Design

## Purpose

Phase 8-10 defines a read-only dashboard and monitoring design for the US Paper Trading operating flow.

The goal is to let an operator review, on one screen or one daily artifact:

- why a symbol was bought
- why a symbol was sold
- why a symbol was blocked
- what the current Paper portfolio looks like
- how Paper performance compares with `SPY` and `QQQ`
- whether scheduler, health, and data quality are stable enough for continued operation

Important boundary:

- this phase is design only
- no web UI is implemented
- no API server is implemented
- no broker API is called
- no real account balance or position is read
- all performance must be labeled as `Paper`

## Core Principles

- Paper-only metrics must be clearly separated from real-account metrics.
- Missing data must stay visible as `data_missing`, `unknown`, or `not_enough_sample`.
- Dashboard output is an operator-review surface, not an execution surface.
- `live_ready=true` must never mean automatic LIVE release.
- SELL / HOLD / REVIEW_REQUIRED state must be visible alongside BUY decisions.

## Target Dashboard Sections

### 1. Daily Overview

Purpose:

- summarize the full daily orchestration result in one compact header section

Key metrics:

- `trade_date`
- `orchestration_executed`
- `mode`
- `final_status`
- `buy_candidate_count`
- `final_buy_allowed_count`
- `sell_signal_count`
- `hold_count`
- `review_required_count`
- `conflict_blocked_count`
- `paper_buy_order_count`
- `paper_sell_order_count`
- `fail_safe_triggered`
- `top_warning_reason`

Data sources:

- `trade.us_trade_orchestration_log`
- `trade.us_integrated_daily_report`
- `trade.us_sell_decision_log`
- `trade.us_buy_decision_log`

Refresh cadence:

- once per daily orchestration run

Normal state:

- orchestration executed
- `final_status=SUCCESS`
- warning count limited

Warning state:

- `final_status=WARNING` or `ERROR`
- fail-safe triggered
- repeated data-missing or review-required growth

Operator action:

- verify that the run completed
- inspect the top warning reason before reviewing performance

### 2. Paper Portfolio Summary

Purpose:

- show current Paper portfolio exposure and aggregate PnL

Key metrics:

- `open_position_count`
- `closed_position_count`
- `total_invested_amount`
- `current_paper_value`
- `unrealized_paper_pnl`
- `unrealized_paper_pnl_pct`
- `realized_paper_pnl`
- `realized_paper_pnl_pct`
- `total_paper_pnl`
- `cash_simulation_available`
- `average_holding_days`
- `largest_position_weight`
- `sector_concentration`

Data sources:

- `trade.us_paper_position`
- `trade.us_paper_position_snapshot`
- `trade.us_paper_sell_order`
- optional Phase 5 `paper.us_stock_paper_account_snapshot`

Refresh cadence:

- once per daily orchestration run

Normal state:

- position snapshot aligns with order history
- concentration is within configured policy

Warning state:

- `PORTFOLIO_STATE_INCONSISTENT`
- sector concentration unknown
- cash simulation unavailable

Operator action:

- review large positions
- inspect symbols with stale or missing marks

### 3. BUY Decision Monitor

Purpose:

- inspect how new-entry candidates moved from ranking to final BUY allow/block

Key fields:

- `symbol`
- `rank_no`
- `total_score`
- `probability`
- `risk_guard_result`
- `conflict_guard_result`
- `final_buy_decision`
- `block_reasons`
- `conflict_reasons`
- `allocated_paper_amount`
- `paper_buy_order_created`

Filters:

- `final_buy_decision`
- `block_reason`
- `conflict_reason`
- `sector`
- `score range`
- `rank range`

Data sources:

- `trade.us_buy_candidate_log`
- `trade.us_buy_decision_log`
- `trade.us_risk_guard_log`
- `trade.us_paper_order`

Refresh cadence:

- once per daily orchestration run

Normal state:

- candidate count is within expected range
- blocked reasons are explainable and consistent

Warning state:

- repeated `DATA_MISSING`
- repeated `CONFLICT_FAILSAFE`
- final allowed count is persistently zero without clear market reason

Operator action:

- inspect whether risk thresholds are too strict
- inspect whether conflict rules are overly suppressing good candidates

### 4. SELL Decision Monitor

Purpose:

- inspect how open Paper positions move to `HOLD`, `SELL`, `PARTIAL_SELL`, or `REVIEW_REQUIRED`

Key fields:

- `symbol`
- `paper_position_id`
- `entry_trade_date`
- `avg_entry_price`
- `latest_price`
- `unrealized_pnl_pct`
- `highest_price_since_entry`
- `drawdown_from_high_pct`
- `holding_days`
- `sell_decision`
- `sell_action`
- `sell_ratio`
- `exit_reason`
- `review_required`
- `applied_rules`
- `paper_sell_order_created`

Filters:

- `sell_decision`
- `exit_reason`
- `pnl range`
- `holding_days range`
- `data_quality_flags`

Data sources:

- `trade.us_sell_decision_log`
- `trade.us_sell_signal_log`
- `trade.us_paper_position`
- `trade.us_paper_position_snapshot`
- `trade.us_paper_sell_order`

Refresh cadence:

- once per daily orchestration run

Normal state:

- applied rules exist for every symbol
- review-required count stays manageable

Warning state:

- missing applied rules
- repeated `PRICE_DATA_MISSING`
- review-required backlog grows

Operator action:

- inspect stop-loss and trailing-stop sensitivity
- inspect high-water-mark integrity when trailing-stop looks suspicious

### 5. Conflict Guard Monitor

Purpose:

- explain why a BUY candidate was blocked by current portfolio or SELL state

Key fields:

- `symbol`
- `buy_candidate`
- `open_position_exists`
- `sell_signal_exists`
- `review_required`
- `cooldown_active`
- `duplicate_buy`
- `conflict_reasons`
- `final_action`

Conflict reasons:

- `OPEN_POSITION_EXISTS`
- `SELL_SIGNAL_EXISTS`
- `REVIEW_REQUIRED_SYMBOL`
- `COOLDOWN_ACTIVE`
- `DUPLICATE_BUY`
- `PORTFOLIO_STATE_INCONSISTENT`

Data sources:

- `trade.us_trade_conflict_log`
- `trade.us_buy_decision_log`
- `trade.us_sell_decision_log`
- `trade.us_paper_position`

Refresh cadence:

- once per daily orchestration run

Normal state:

- conflicts are explainable and low-noise

Warning state:

- repeated `PORTFOLIO_STATE_INCONSISTENT`
- same symbol repeatedly blocked by stale cooldown or duplicate rows

Operator action:

- review whether re-entry cooldown is too long
- inspect portfolio-state reconstruction if inconsistency repeats

### 6. Paper Performance Monitor

Purpose:

- track cumulative and rolling Paper performance over time

Key metrics:

- `cumulative_paper_return_pct`
- `daily_paper_return_pct`
- `weekly_paper_return_pct`
- `monthly_paper_return_pct`
- `win_rate`
- `loss_rate`
- `average_trade_return_pct`
- `median_trade_return_pct`
- `best_trade`
- `worst_trade`
- `max_drawdown_pct`
- `profit_factor`
- `average_holding_days`
- `trade_count`
- `active_position_count`

Periods:

- `20 trading days`
- `60 trading days`
- `120 trading days`
- `all`

Data sources:

- `trade.us_paper_performance_summary`
- `trade.us_paper_sell_order`
- `trade.us_paper_position_snapshot`
- `trade.us_buy_readiness_report`

Refresh cadence:

- daily after orchestration or via separate dashboard batch

Normal state:

- enough sample size
- performance metrics are internally consistent

Warning state:

- `NOT_ENOUGH_SAMPLE`
- realized trade sample too small
- persistent negative excess return

Operator action:

- interpret realized and unrealized metrics separately
- avoid over-weighting short sample windows

### 7. Benchmark Comparison

Purpose:

- compare Paper performance against `SPY` and `QQQ`

Key metrics:

- `paper_return_pct`
- `SPY_return_pct`
- `QQQ_return_pct`
- `excess_return_vs_SPY`
- `excess_return_vs_QQQ`
- `benchmark_win`
- `rolling_excess_return`
- `benchmark_data_missing`

Data sources:

- `trade.us_paper_performance_summary`
- `trade.us_buy_readiness_report`
- benchmark rows from market / feature tables

Refresh cadence:

- daily

Normal state:

- benchmark data exists
- comparison windows align with Paper window

Warning state:

- `BENCHMARK_DATA_MISSING`
- benchmark window mismatch

Operator action:

- treat benchmark comparison as a review metric, not an execution trigger

### 8. Risk / Data Quality Monitor

Purpose:

- expose reliability and data-trust signals that affect interpretation of BUY/SELL outcomes

Key metrics:

- `data_missing_count`
- `data_missing_rate`
- `price_data_missing_count`
- `benchmark_data_missing_count`
- `financial_data_missing_count`
- `invalid_decision_log_count`
- `block_reason_missing_count`
- `portfolio_state_inconsistent_count`
- `fail_safe_triggered_count`
- `review_required_count`

State thresholds:

- `NORMAL`: `data_missing_rate <= 5%`
- `WARNING`: `5% < data_missing_rate <= 20%`
- `CRITICAL`: `data_missing_rate > 20%`

Data sources:

- `trade.us_buy_decision_log`
- `trade.us_sell_decision_log`
- `trade.us_trade_orchestration_log`
- `trade.us_trade_scheduler_health_check`

Refresh cadence:

- daily

Normal state:

- missing rate low
- invalid decision logs absent

Warning state:

- repeated missing data
- fail-safe grows while market conditions are normal

Operator action:

- inspect upstream ranking / feature / benchmark ingestion
- inspect position reconstruction quality

### 9. Scheduler / Health Check Monitor

Purpose:

- show whether daily orchestration is operationally stable

Key metrics:

- `scheduler_run_status`
- `last_run_at`
- `last_success_at`
- `duplicate_run_detected`
- `stale_lock_removed`
- `scheduler_success_rate`
- `health_check_status`
- `report_generated`
- `json_report_exists`
- `markdown_report_exists`
- `pipeline_should_fail`
- `warning_count`
- `error_count`

State rules:

- `PASS`: scheduler success and health pass
- `WARNING`: report missing or warnings exist
- `ERROR`: orchestration failed or lock issue persists

Data sources:

- `trade.us_trade_scheduler_run_log`
- `trade.us_trade_scheduler_health_check`
- `trade.us_trade_scheduler_lock_log`
- `trade.us_trade_orchestration_log`

Refresh cadence:

- every orchestration scheduler run

Operator action:

- inspect duplicate-run and stale-lock events
- inspect missing report artifacts before trusting daily overview

### 10. LIVE Readiness Monitor

Purpose:

- expose review-only indicators that help decide whether Paper operation is mature enough for later LIVE planning

Key metrics:

- `live_ready`
- `readiness_score`
- `manual_approval_required`
- `min_shadow_days_met`
- `min_paper_days_met`
- `min_paper_orders_met`
- `win_rate_threshold_met`
- `max_drawdown_threshold_met`
- `excess_return_threshold_met`
- `data_missing_rate_threshold_met`
- `scheduler_success_rate_threshold_met`
- `not_ready_reasons`

Required banner:

- `live_ready=true does not enable LIVE trading`
- `LIVE transition requires a later phase and explicit manual approval`

Data sources:

- `trade.us_buy_readiness_report`
- `trade.us_live_promotion_check`
- `trade.us_paper_performance_summary`
- `trade.us_trade_scheduler_run_log`
- `trade.us_trade_scheduler_health_check`

Refresh cadence:

- daily or weekly review

Normal state:

- readiness thresholds met and warnings limited

Warning state:

- readiness score falling
- excess return negative
- scheduler success unstable

Operator action:

- treat this as a governance view only
- never use it as direct release authorization

## Data Source And Table Mapping

| Table / Source | Usage | Key fields | Dashboard sections | Required | Missing-data handling |
| --- | --- | --- | --- | --- | --- |
| `trade.us_buy_decision_log` | final BUY outcome | `trade_date`, `symbol`, `decision`, `decision_reason_code`, `conflict_reasons` | Daily Overview, BUY Decision, Risk | Required | section shows `buy_decision_missing` |
| `trade.us_risk_guard_log` | BUY guard evidence | `guard_name`, `guard_status`, `reason_code` | BUY Decision, Risk | Optional | show partial BUY monitor |
| `trade.us_paper_order` | Paper BUY artifact | `trade_date`, `symbol`, `side`, `paper_order_amount` | Daily Overview, BUY Decision, Portfolio | Optional | count stays `0` or `unknown` |
| `trade.us_sell_decision_log` | final SELL outcome | `decision`, `sell_action`, `exit_reason`, `applied_rules` | Daily Overview, SELL Decision, Conflict, Risk | Required | section warns `sell_decision_missing` |
| `trade.us_sell_signal_log` | rule-level SELL evidence | `rule_name`, `rule_result`, `metric_value` | SELL Decision | Optional | applied-rule detail degrades |
| `trade.us_paper_sell_order` | Paper SELL artifact | `sell_quantity`, `sell_price_ref`, `realized_paper_pnl` | Daily Overview, SELL Decision, Portfolio, Performance | Optional early, required later | realized metrics become `unknown` |
| `trade.us_paper_position` | current Paper position state | `paper_position_id`, `symbol`, `remaining_quantity`, `avg_entry_price` | Portfolio, SELL Decision, Conflict | Required | `portfolio_state_inconsistent` |
| `trade.us_paper_position_snapshot` | daily mark-to-market snapshot | `snapshot_date`, `latest_price`, `unrealized_pnl` | Portfolio, Performance | Required | unrealized metrics become `unknown` |
| `trade.us_trade_orchestration_log` | run-level summary | `success`, `fail_safe_triggered`, `conflict_summary` | Daily Overview, Risk, Scheduler | Required | dashboard status becomes `warning` |
| `trade.us_trade_conflict_log` | per-symbol conflict result | `symbol`, `conflict_reasons`, `buy_allowed_after_conflict_check` | Conflict, BUY Decision | Required | conflict monitor shows `incomplete` |
| `trade.us_integrated_daily_report` | persisted report body | `trade_date`, `summary_json` | Daily Overview, Risk | Optional | file fallback allowed |
| `trade.us_trade_scheduler_run_log` | scheduler stability trend | `job_status`, `warnings`, `errors` | Scheduler / Health, LIVE Readiness | Optional until DB persistence lands | file-based scheduler log fallback |
| `trade.us_trade_scheduler_health_check` | post-run health result | `health_result`, `warnings`, `errors` | Scheduler / Health, Risk | Optional until DB persistence lands | report artifact validation fallback |
| `trade.us_buy_readiness_report` | readiness evaluation | `live_ready`, `readiness_score`, `reasons` | Performance, Benchmark, LIVE Readiness | Optional | readiness section marked `not_evaluated` |
| `trade.us_paper_performance_summary` | rolling Paper performance | `period_label`, `total_return_pct`, `benchmark_return_pct`, `max_drawdown_pct` | Performance, Benchmark, LIVE Readiness | Optional early, preferred target | derive from files if absent |

## API And Query Design

### `GET /api/us-trade/dashboard/daily?trade_date=YYYY-MM-DD`

Purpose:

- return the full daily dashboard snapshot for one trade date

Parameters:

- `trade_date` required

Response fields:

- `daily_overview`
- `portfolio_summary`
- `buy_summary`
- `sell_summary`
- `conflict_summary`
- `risk_summary`
- `scheduler_summary`

Data sources:

- integrated daily report first
- fall back to underlying decision logs

Errors:

- `404 DASHBOARD_NOT_FOUND`
- `409 INCOMPLETE_DAILY_ARTIFACT`
- `500 DASHBOARD_ASSEMBLY_ERROR`

Caching:

- recommended once the daily run is complete

### `GET /api/us-trade/dashboard/portfolio`

Purpose:

- return current Paper portfolio state

Parameters:

- optional `trade_date`

Response fields:

- open positions
- closed positions
- aggregate pnl
- concentration summary

Data sources:

- `trade.us_paper_position`
- `trade.us_paper_position_snapshot`
- `trade.us_paper_sell_order`

Errors:

- `409 PORTFOLIO_STATE_INCONSISTENT`

Caching:

- short cache acceptable for file/dashboard views

### `GET /api/us-trade/dashboard/buy-decisions?trade_date=YYYY-MM-DD`

Purpose:

- return symbol-level BUY decision monitor rows

Parameters:

- `trade_date`
- optional `decision`
- optional `block_reason`
- optional `sector`
- optional `min_score`
- optional `max_rank`

Response fields:

- symbol-level BUY monitor rows

Data sources:

- `trade.us_buy_candidate_log`
- `trade.us_buy_decision_log`
- `trade.us_risk_guard_log`

Errors:

- `404 BUY_DECISION_NOT_FOUND`

Caching:

- daily cache recommended

### `GET /api/us-trade/dashboard/sell-decisions?trade_date=YYYY-MM-DD`

Purpose:

- return symbol-level SELL decision monitor rows

Parameters:

- `trade_date`
- optional `decision`
- optional `exit_reason`
- optional `min_pnl_pct`
- optional `max_holding_days`

Response fields:

- symbol-level SELL monitor rows

Data sources:

- `trade.us_sell_decision_log`
- `trade.us_sell_signal_log`
- `trade.us_paper_position_snapshot`

Errors:

- `404 SELL_DECISION_NOT_FOUND`

Caching:

- daily cache recommended

### `GET /api/us-trade/dashboard/conflicts?trade_date=YYYY-MM-DD`

Purpose:

- return symbol-level conflict rows

Parameters:

- `trade_date`
- optional `conflict_reason`

Response fields:

- conflict rows and summary counts

Data sources:

- `trade.us_trade_conflict_log`

Errors:

- `404 CONFLICT_LOG_NOT_FOUND`

Caching:

- daily cache recommended

### `GET /api/us-trade/dashboard/performance?days=60`

Purpose:

- return rolling Paper performance summary

Parameters:

- `days`
- optional `benchmark`

Response fields:

- cumulative return
- rolling returns
- win rate
- drawdown
- benchmark comparison

Data sources:

- `trade.us_paper_performance_summary`
- `trade.us_buy_readiness_report`

Errors:

- `404 PERFORMANCE_SUMMARY_NOT_FOUND`
- `409 NOT_ENOUGH_SAMPLE`

Caching:

- cacheable per day

### `GET /api/us-trade/dashboard/readiness`

Purpose:

- return LIVE-readiness review snapshot

Parameters:

- optional `trade_date`

Response fields:

- readiness score
- threshold checks
- reasons

Data sources:

- `trade.us_buy_readiness_report`
- `trade.us_live_promotion_check`

Errors:

- `404 READINESS_NOT_EVALUATED`

Caching:

- daily cache recommended

### `GET /api/us-trade/dashboard/health`

Purpose:

- return scheduler and health monitor state

Parameters:

- optional `days`

Response fields:

- latest run state
- success rate
- duplicate lock events
- report existence
- warnings and errors

Data sources:

- `trade.us_trade_scheduler_run_log`
- `trade.us_trade_scheduler_health_check`
- `trade.us_trade_orchestration_log`

Errors:

- `404 HEALTH_HISTORY_NOT_FOUND`

Caching:

- short-lived cache allowed

## File-Based Dashboard Output Design

Recommended output paths:

- `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.json`
- `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.md`
- `reports/lee_trader_us/dashboard/latest_dashboard.json`
- `reports/lee_trader_us/dashboard/latest_dashboard.md`

Recommended file payload:

- `daily_overview`
- `portfolio_summary`
- `buy_monitor`
- `sell_monitor`
- `conflict_monitor`
- `performance_summary`
- `benchmark_comparison`
- `risk_data_quality`
- `scheduler_health`
- `live_readiness`

Markdown file intent:

- human daily review

JSON file intent:

- CLI, API, or future UI data source

## ENV Design

Recommended dashboard ENV:

```env
US_DASHBOARD_ENABLED=0
US_DASHBOARD_OUTPUT_DIR=reports/lee_trader_us/dashboard
US_DASHBOARD_FORMAT=json,markdown
US_DASHBOARD_INCLUDE_BUY_MONITOR=1
US_DASHBOARD_INCLUDE_SELL_MONITOR=1
US_DASHBOARD_INCLUDE_CONFLICT_MONITOR=1
US_DASHBOARD_INCLUDE_PERFORMANCE=1
US_DASHBOARD_INCLUDE_HEALTH=1
US_DASHBOARD_INCLUDE_READINESS=1
US_DASHBOARD_DEFAULT_LOOKBACK_DAYS=60
US_DASHBOARD_DATA_MISSING_WARNING_PCT=5
US_DASHBOARD_DATA_MISSING_CRITICAL_PCT=20
```

Policy:

- disabled by default
- output is read-only
- missing-data thresholds are operator-warning thresholds, not execution thresholds

## Proposed Future Module Structure

Actual repository-aligned recommendation:

```text
python/us/dashboard/
  - __init__.py
  - config.py
  - dashboard_data_loader.py
  - dashboard_summary.py
  - dashboard_report_generator.py
  - dashboard_api_schema.py
  - run_us_dashboard_report.py
```

Prompt-aligned generic reference:

```text
src/modules/lee_trader_us/dashboard/
  - __init__.py
  - config.py
  - dashboard_data_loader.py
  - dashboard_summary.py
  - dashboard_report_generator.py
  - dashboard_api_schema.py
  - run_us_dashboard_report.py
```

Suggested module roles:

- `config.py`: dashboard ENV loading and output policy
- `dashboard_data_loader.py`: assemble dashboard-ready data from DB tables and file artifacts
- `dashboard_summary.py`: compute summary cards, rates, pnl, benchmark deltas, readiness flags
- `dashboard_report_generator.py`: generate JSON and Markdown dashboard artifacts
- `dashboard_api_schema.py`: response schema contracts for future API or CLI consumers
- `run_us_dashboard_report.py`: CLI entrypoint for daily dashboard build

## Normalization And Query Notes

- prefer one canonical `trade_date` across all sections
- treat `trade.us_integrated_daily_report` as the first-choice assembled daily source when present
- treat `trade.us_paper_position_snapshot` as the preferred daily unrealized PnL source
- treat `trade.us_paper_sell_order` as the preferred realized PnL source
- if DB persistence is incomplete, allow file-based fallback from orchestration and readiness report artifacts

## Known Limitations

- cash simulation is not yet reliable enough for required dashboard display
- realized trade history may be too short for stable performance statistics
- benchmark comparison depends on clean daily benchmark data
- scheduler DB persistence is still a design target in some environments
- sector concentration depends on symbol-to-sector mapping quality
- portfolio NAV history may remain approximate until a dedicated daily equity series is persisted

## Phase 8-11 Recommendation

Implementation-focused next step:

- add `python/us/dashboard/` read-only modules
- generate daily JSON / Markdown dashboard artifacts
- reuse existing orchestration and readiness outputs before building any web UI
- keep the first implementation file-based and operator-review oriented

## Phase 8-11 Implemented File-Based Dashboard

Phase 8-11 now implements the first read-only file dashboard layer.

Implemented modules in the actual repository:

- `python/us/dashboard/config.py`
- `python/us/dashboard/dashboard_data_loader.py`
- `python/us/dashboard/dashboard_summary.py`
- `python/us/dashboard/dashboard_report_generator.py`
- `python/us/dashboard/dashboard_markdown_renderer.py`
- `python/us/dashboard/dashboard_json_writer.py`
- `python/us/dashboard/run_us_dashboard_report.py`
- `python/us/run_us_dashboard_report.py`
- `scripts/run_us_dashboard_report.py`

### Execution

```powershell
python -m python.us.dashboard.run_us_dashboard_report --force
python -m python.us.dashboard.run_us_dashboard_report --trade-date 2026-05-14 --force
python -m python.us.dashboard.run_us_dashboard_report --format json --force
python -m python.us.dashboard.run_us_dashboard_report --format markdown --force
python scripts/run_us_dashboard_report.py --force
```

### Output Files

- `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.json`
- `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.md`
- `reports/lee_trader_us/dashboard/latest_dashboard.json`
- `reports/lee_trader_us/dashboard/latest_dashboard.md`

Meaning of `latest_dashboard.*`:

- a rolling pointer to the most recently generated dashboard artifact
- useful for operators, scripts, and future UI readers that do not want to resolve the latest trade date manually

### Current Behavior

- DB-first load when the related `trade.*` tables exist
- graceful file fallback when DB tables are absent or unavailable
- dashboard still renders when some sources are missing
- missing inputs are shown as `unknown`, `data_missing`, or `not_available`
- all output is explicitly Paper-only

### Important Boundary

- no web UI
- no API server
- no broker API call
- no real account balance lookup
- no real account position lookup
- no real BUY or SELL execution

### Current Known Limitations

- dashboard performance uses existing Paper logs and current summaries, not a dedicated NAV time series
- scheduler health detail may degrade to partial file-based visibility when DB persistence is incomplete
- realized performance remains limited when Paper SELL history is still sparse
- sector concentration remains weak when symbol-to-sector mapping is missing in position data

## Phase 8-12 Scheduler Integration And Notification

Phase 8-12 adds optional orchestration-scheduler integration and notification payload generation.

Implemented modules:

- `python/us/dashboard/scheduler_integration.py`
- `python/us/dashboard/dashboard_health_adapter.py`
- `python/us/dashboard/dashboard_notification_formatter.py`
- `python/us/dashboard/dashboard_notification_payload.py`

### Runtime Order

1. trade orchestration runs first
2. integrated trade report is written
3. dashboard report is generated
4. health check validates integrated report and dashboard artifacts together
5. notification payload is generated as text/json file only

### Dashboard Health Checks

Current checks include:

- dashboard JSON exists
- dashboard Markdown exists
- latest dashboard files exist
- dashboard JSON payload is parseable
- Paper Trading notice exists in Markdown
- `paper_trading_only=true`
- `live_trading_enabled=false`
- `generated_at` exists
- dashboard `trade_date` matches the scheduler trade date

### Notification Payload Behavior

- text summary is generated
- JSON payload is generated
- both are written to local files only
- no email send
- no Slack/webhook send
- no external notification API call

### Current Limitation

- scheduler integration does not yet persist dashboard/notification state into DB by default
- notification payload is a local operator artifact only

## Phase 8-13 Notification Adapter Relationship

Phase 8-13 does not change dashboard generation itself. It defines how dashboard notification payloads can later be routed safely.

Relationship to dashboard artifacts:

- dashboard JSON / Markdown remain the source artifacts
- notification payloads remain summary artifacts derived from the dashboard layer
- a future notification adapter must run after dashboard health validation
- notification delivery approval must not be confused with trading approval

Required dashboard-to-notification guarantees:

- dashboard payload must keep `paper_trading_only=true`
- dashboard payload must keep `live_trading_enabled=false`
- notification summaries must include a Paper-only notice
- notification summaries must never imply real account holdings or live execution

Current limitation:

- Phase 8-13 is still design-only
- no actual email or Slack delivery is implemented
