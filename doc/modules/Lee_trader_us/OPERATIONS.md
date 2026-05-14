# Lee_trader_us Operations

> 문서 역할: `현재 기준 문서`
>
> 현재 수동 실행 명령과 운영 확인 절차를 모아 둔 문서다. 실제 실행 시 가장 직접적으로 참고한다.

## Manual Commands

```powershell
python scripts/init_us_stock_universe.py --dry-run
python scripts/init_us_stock_universe.py
python scripts/init_us_stock_universe.py --refresh
python scripts/verify_us_stock_rank_table.py
python scripts/verify_us_stock_rank_table.py --write-sample --cleanup
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --dry-run
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --symbols AAPL,MSFT,NVDA --dry-run
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --top-n 20
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --grade BUY
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format markdown
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format csv
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --auto-calculate
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --show-excluded --limit 50
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --top-n 20
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --fail-on-error
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --output markdown
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --holding-days 5,20,60 --dry-run
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --holding-days 5,20,60
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --strategy TOP20
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --backtest-id US_RANK_RULE_V1_TEST
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format console
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format markdown
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format csv
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --strategy US_RANK_TOP20 --holding-days 20
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --symbol NVDA
python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12 --dry-run
python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format console
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format markdown
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format csv
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --strategy US_RANK_TOP20 --holding-days 20
python scripts/experiment_us_stock_rule_weights.py --start-date 2026-01-01 --end-date 2026-05-12 --weight-configs RULE_V1_BASELINE,RULE_V1_MOMENTUM_PLUS,RULE_V1_QUALITY_PLUS --holding-days 20 --dry-run
python scripts/experiment_us_stock_rule_weights.py --start-date 2026-01-01 --end-date 2026-05-12 --weight-configs ALL --holding-days 5,20,60 --experiment-id US_RULE_WEIGHT_EXP_001
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format console
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format markdown
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format csv
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format console
python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format markdown
python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1 --dry-run
python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1 --trade-date 2026-05-15
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --dry-run
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --create-approval-request --requested-by SYSTEM
python scripts/manage_us_live_kill_switch.py --list
python scripts/manage_us_live_kill_switch.py --activate --scope GLOBAL --reason-code manual_stop --reason-detail "test global stop" --performed-by lee --dry-run
python scripts/manage_us_live_kill_switch.py --clear --scope GLOBAL --clear-reason "test clear" --performed-by lee --dry-run
python scripts/evaluate_us_live_kill_switch.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --policy-id US_LIVE_RULE_V1 --dry-run
python scripts/manage_us_live_order_approval.py --list --status PENDING
python scripts/manage_us_live_order_approval.py --approval-id USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000
python scripts/manage_us_live_order_approval.py --approval-id USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000 --approve --approved-by lee --reason "Micro Live test approved"
python scripts/manage_us_live_order_approval.py --approval-id USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515123000 --reject --rejected-by lee --reason "Rejected for review"
python scripts/manage_us_live_order_approval.py --expire-pending
python -m python.us.load_us_universe --universe NASDAQ100
python -m python.us.download_us_prices --universe NASDAQ100 --backfill
python -m python.us.download_us_prices --universe NASDAQ100 --incremental
python -m python.us.validate_us_price_data --universe NASDAQ100 --as-of-date 2026-05-11
python -m python.us.build_us_features --universe NASDAQ100
python -m python.us.build_us_features --universe NASDAQ100 --ticker AAPL --verbose
python -m python.us.build_us_features --universe NASDAQ100 --limit 5
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --backfill
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --incremental
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --skip-prices
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --skip-quality
```

## Integrated Guide

- Use [US_STOCK_RANKING_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_RANKING_V1.md) as the primary Phase 3 operator guide.
- Use [PHASE3_CHECKLIST.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/PHASE3_CHECKLIST.md) for short-form completion tracking.
- Use [US_STOCK_BACKTEST_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md) for Phase 4-1 rank backtest setup and SQL checks.
- Use [US_STOCK_PAPER_TRADING.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_PAPER_TRADING.md) for Phase 5 paper operations.
- Use [US_STOCK_LIVE_OPERATION_RUNBOOK.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_OPERATION_RUNBOOK.md) as the primary Phase 6 operations guide.
- Use [US_STOCK_LIVE_TRADING_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_TRADING_POLICY.md) for Phase 6 live-order safety policy before any Micro Live review.
- Use [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md) for Phase 6-2 risk-state tables, YAML policy structure, and safe-default validation flow.
- Use [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md) for Phase 6-4 kill-switch scope, event log, and manual-management flow.
- Use [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md) for Phase 6-5 approval-request tables, approval lifecycle, and audit flow.
- Use [PHASE7_SUMMARY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/PHASE7_SUMMARY.md) for the current Phase 7 document map and next-step handoff.

## Phase 7 Micro Live Commands

```powershell
python scripts/create_us_micro_order_request.py --approval-id <APPROVAL_ID> --execution-mode MOCK --dry-run
python scripts/manage_us_micro_order.py --list --account-id US_LIVE_TEST
python scripts/send_us_micro_order_mock.py --micro-order-id <MICRO_ORDER_ID> --dry-run
python scripts/send_us_micro_order_sandbox.py --micro-order-id <MICRO_ORDER_ID> --dry-run
python scripts/send_us_micro_order_live.py --micro-order-id <MICRO_ORDER_ID> --dry-run
python scripts/sync_us_micro_order_status.py --micro-order-id <MICRO_ORDER_ID> --dry-run
python scripts/report_us_micro_order_status.py --account-id US_LIVE_TEST --trade-date 2026-05-16
python scripts/reconcile_us_micro_live.py --account-id US_LIVE_TEST --recon-date 2026-05-16 --execution-mode MOCK --dry-run
python scripts/report_us_micro_live_operations.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --format console
python scripts/run_us_micro_live_daily_check.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --execution-mode MOCK --dry-run
```

## Phase 7 Notes

- Phase 7 is still not unrestricted live trading.
- all default gates remain closed
- reconciliation is record-and-review only
- operations report is read/report only
- no automatic correction order is allowed

## Phase 8-1 Limited BUY Automation Design Notes

Phase 8-1 is still a design stage.

- no BUY execution script is added in this phase
- no broker call is added in this phase
- no scheduler step is added in this phase
- no live BUY release is allowed in this phase

### Phase 8-1 Design Review Inputs

Use the following existing commands as design-review inputs when shaping limited BUY automation policy.

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --show-excluded --limit 50
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12
python scripts/run_us_live_pre_trade_check.py --trade-date 2026-05-15 --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run
python scripts/manage_us_live_kill_switch.py --list
python scripts/report_us_micro_live_operations.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --format console
python scripts/run_us_micro_live_daily_check.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --execution-mode MOCK --dry-run
```

### Phase 8-1 Operator Review Questions

When preparing a limited BUY design or later SHADOW rollout, answer these before any implementation:

1. Are there enough `BUY` or better ranking rows to justify automation at all?
2. Are excluded rows dominated by data-quality weakness or by intentionally conservative score thresholds?
3. Does the current pre-trade check block reasons align with the proposed BUY automation blocks?
4. Is kill-switch state visible enough to be treated as a first-class hard block?
5. Are Micro Live reconciliation and operations-report results stable enough to act as future LIVE prerequisites?
6. Is repeat-buy history available enough to enforce a cooldown rule safely?
7. If any of the above is uncertain, should the answer default to `BLOCK`?

### Proposed Future Phase 8 Command Families

These are design placeholders only and are not implemented in Phase 8-1.

- `evaluate_us_buy_candidates.py`: SHADOW candidate funnel and decision logging
- `report_us_buy_automation.py`: BUY candidate and block report
- `run_us_buy_shadow_daily.py`: wrapper for daily SHADOW review
- `release_us_buy_paper_orders.py`: future PAPER-only decision-to-order bridge

### Phase 8-1 Safety Reminder

- ranking output is not an order
- pre-trade `ALLOW` is not a release instruction
- approval state is not an execution instruction
- SHADOW mode is evaluation only
- PAPER mode is virtual only
- LIVE mode remains reserved for a later gated phase

## Phase 8-2 BUY Automation Skeleton Commands

```powershell
python -m python.us.buy_automation.run_us_buy_automation
python scripts/run_us_buy_automation.py
python scripts/run_us_buy_automation.py --trade-date 2026-05-14 --account-id US_BUY_SHADOW
python scripts/run_us_buy_automation.py --trade-date 2026-05-14 --account-id US_BUY_PAPER
```

### Phase 8-2 What The Skeleton Does

1. reads the latest available `recommend.us_stock_rank_daily` snapshot
2. loads Top-N candidates
3. enriches them with available price / financial / relative-strength snapshots
4. applies fail-safe BUY guard checks
5. writes JSON logs always
6. writes DB logs only if the proposed `trade.*` tables already exist
7. never calls any broker API

### Phase 8-2 Log Check

- inspect the console block summary first
- inspect the JSON artifact under `US_BUY_REPORT_OUTPUT_DIR`
- if DDL was applied manually, inspect:
  - `trade.us_buy_candidate_log`
  - `trade.us_buy_decision_log`
  - `trade.us_risk_guard_log`
  - `trade.us_paper_order`

### Phase 8-2 Known Operational Limits

- current ranking data may produce zero allowed candidates
- if probability is required but not present, fail-safe blocking is expected
- `LIVE` mode remains non-executable even if selected in ENV

## Phase 8-3 BUY Report Commands

```powershell
python scripts/run_us_buy_automation.py
python scripts/run_us_buy_report.py --format console
python scripts/run_us_buy_report.py --trade-date 2026-05-11 --format json
python scripts/run_us_buy_report.py --trade-date 2026-05-11 --format markdown
```

### Phase 8-3 Output Locations

- raw execution log: `output/us_stock_buy_automation/`
- final JSON/Markdown report: `reports/lee_trader_us/buy_automation/`

### Phase 8-3 Reading Order

1. run the BUY automation skeleton
2. inspect console block summary
3. generate the daily report
4. inspect:
   - block summary
   - rule summary
   - invalid decision logs
   - PAPER performance section

### Phase 8-3 Interpretation Notes

- `AUTOMATION_DISABLED` means the evaluation ran but no candidate was allowed for operational release
- `DATA_MISSING` and related missing-data codes are fail-safe blocks, not bugs by themselves
- `INVALID_DECISION_LOG` means a blocked candidate had no block reason and should be investigated before any tighter automation
- `PRICE_DATA_MISSING` in PAPER performance means the report could not compute current PnL safely
- `LIVE` transition readiness remains `NOT_EVALUATED`

## Phase 8-4 BUY Scheduler Commands

```powershell
python -m python.us.buy_automation.scheduler_job
python scripts/run_us_buy_scheduler_job.py
python scripts/run_us_buy_scheduler_job.py --trade-date 2026-05-14
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force
```

### Phase 8-4 Scheduler Reading Order

1. confirm ranking / score data is already available or understand that the job may return `SOURCE_DATA_MISSING`
2. run the scheduler job directly once
3. inspect console summary:
   - `enabled`
   - `mode`
   - `automation_executed`
   - `report_executed`
   - `success`
4. inspect the generated BUY report
5. if integrated through `run_us_daily_pipeline.py`, inspect the pipeline summary line for `buy_scheduler`

### Phase 8-4 Scheduler Safety Notes

- `LIVE` mode is blocked with `LIVE_DISABLED_IN_SCHEDULER`
- scheduler failures are isolated by default
- `US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR=1` is the only mode that lets scheduler errors fail the parent pipeline
- scheduler integration is SHADOW / PAPER review only
- no broker API, account query, or real BUY order path is added

## Phase 8-5 Readiness Commands

```powershell
python -m python.us.buy_automation.run_us_buy_readiness
python scripts/run_us_buy_readiness.py
python scripts/run_us_buy_readiness.py --days 60 --benchmark SPY --format console
python scripts/run_us_buy_readiness.py --days 60 --benchmark SPY --format json
python scripts/run_us_buy_readiness.py --days 60 --benchmark SPY --format markdown
```

### Phase 8-5 Reading Order

1. confirm SHADOW/PAPER raw logs exist under `output/us_stock_buy_automation/`
2. confirm scheduler summary artifacts exist
3. run readiness evaluation
4. inspect:
   - paper order count
   - total return vs benchmark
   - excess return
   - win rate
   - max drawdown
   - data missing rate
   - scheduler success rate
   - NOT_READY reasons

### Phase 8-5 Interpretation Notes

- `live_ready=true` means only that the system is eligible for manual review
- `manual_approval_required=true` remains mandatory
- `BENCHMARK_DATA_MISSING` means fail-safe `NOT_READY`
- `PAPER_DAYS_BELOW_MINIMUM` or `PAPER_ORDERS_BELOW_MINIMUM` means the sample is still too small
- readiness output must never be treated as permission to activate LIVE automatically

## Phase 8-6 SELL Design Review Inputs

Phase 8-6 is design-only. No SELL execution command is added yet.

Use the following existing commands as design-review inputs:

```powershell
python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console
python scripts/run_us_buy_report.py --format console
python scripts/run_us_buy_readiness.py --format console
python scripts/report_us_micro_live_operations.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --format console
```

### Phase 8-6 Operator Review Questions

1. Which exits should force full Paper liquidation versus review-only state?
2. Is stop-loss more important than benchmark-relative underperformance for the first version?
3. Should data errors become `REVIEW_REQUIRED` rather than automatic Paper SELL?
4. Is same-day BUY re-entry after a SELL decision allowed or blocked?
5. Is partial take-profit worth adding before Paper position persistence is more mature?
6. Which position state is authoritative for `highest_price_since_entry`?
7. If price / rank / benchmark inputs disagree, should the system default to hold, sell, or review?

## Phase 8-7 SELL Automation Commands

```powershell
python -m python.us.sell_automation.run_us_sell_automation
python scripts/run_us_sell_automation.py
python scripts/run_us_sell_automation.py --trade-date 2026-05-14 --account-id US_SELL_SHADOW
python scripts/run_us_sell_automation.py --trade-date 2026-05-14 --account-id US_SELL_PAPER
```

### Phase 8-7 What The Skeleton Does

1. reads Paper BUY/SSELL history only
2. reconstructs open Paper positions
3. loads latest price, ranking, and benchmark context
4. evaluates SELL rules and writes a decision log
5. creates Paper SELL artifacts only in `PAPER` mode
6. never calls a broker API
7. never reads real account position or balance

### Phase 8-7 Output Review

- inspect console summary first:
  - `mode`
  - `enabled`
  - `loaded_positions`
  - `hold_positions`
  - `sell_signals`
  - `partial_sell_signals`
  - `review_required`
  - `paper_sell_orders`
- inspect the JSON artifact under `output/us_stock_sell_automation/` or `US_SELL_REPORT_OUTPUT_DIR`
- if DDL was applied manually, inspect:
  - `trade.us_sell_decision_log`
  - `trade.us_sell_signal_log`
  - `trade.us_paper_sell_order`
  - `trade.us_paper_position_snapshot`

### Phase 8-7 Safety Notes

- `LIVE` mode remains blocked with `LIVE_NOT_IMPLEMENTED`
- missing price, ranking, probability, or benchmark context can result in `REVIEW_REQUIRED`
- data uncertainty must not trigger automatic real SELL

## Phase 8-8 Trade Orchestration Commands

```powershell
python -m python.us.trade_orchestration.run_us_trade_orchestration
python scripts/run_us_trade_orchestration.py
python scripts/run_us_trade_orchestration.py --trade-date 2026-05-14
python scripts/run_us_trade_orchestration.py --trade-date 2026-05-14 --mode SHADOW
```

### Phase 8-8 Execution Order

1. run SELL automation first
2. reconstruct Paper portfolio state
3. run BUY automation
4. apply conflict guard
5. generate integrated report

### Phase 8-8 Conflict Guard Meaning

BUY is blocked when any of the following applies:

- `OPEN_POSITION_EXISTS`
- `SELL_SIGNAL_EXISTS`
- `REVIEW_REQUIRED_SYMBOL`
- `COOLDOWN_ACTIVE`
- `DUPLICATE_BUY`
- `PORTFOLIO_STATE_INCONSISTENT`

### Phase 8-8 Output Locations

- integrated report JSON / Markdown:
  - `reports/lee_trader_us/trade_orchestration/`
- raw orchestration JSON log:
  - same report directory by default

### Phase 8-8 Scheduler Note

- BUY-only scheduler and orchestration scheduler must not be enabled together
- current BUY scheduler records `SCHEDULER_CONFIGURATION_CONFLICT` when orchestration scheduler flags are also enabled

## Phase 8-9 Trade Scheduler Commands

```powershell
python -m python.us.trade_orchestration.scheduler_job
python scripts/run_us_trade_scheduler_job.py
python -m python.us.trade_orchestration.run_us_trade_orchestration
```

### Phase 8-9 Scheduler Flow

1. scheduler guard
2. run lock
3. daily trade orchestrator
4. integrated report check
5. health check
6. operations checklist write
7. lock release

### Phase 8-9 Daily Operator Checklist

1. confirm scheduler job executed
2. review SELL summary
3. review BUY final candidates
4. review conflict block reasons
5. review `REVIEW_REQUIRED` symbols
6. review data-missing rate
7. review Paper portfolio pnl
8. confirm integrated report exists
9. confirm duplicate run was not detected
10. confirm `LIVE` mode remained blocked

### Phase 8-9 Pipeline Integration

Actual current pipeline hook:

- `python/us/run_us_daily_pipeline.py`

Current policy:

- trade scheduler runs after upstream feature/ranking-related stages
- if orchestration scheduler is enabled and disable-buy-only flag is on, BUY-only scheduler is skipped
- if orchestration scheduler is off, legacy BUY-only scheduler can still run independently

## Phase 8-10 Dashboard Design Commands

Phase 8-10 is design-only. No dashboard builder or API server is implemented yet.

Future intended command family:

```powershell
python -m python.us.dashboard.run_us_dashboard_report
python scripts/run_us_dashboard_report.py --trade-date 2026-05-14
python scripts/run_us_dashboard_report.py --trade-date 2026-05-14 --format markdown
```

Current operator review inputs for the future dashboard:

```powershell
python -m python.us.trade_orchestration.scheduler_job
python -m python.us.trade_orchestration.run_us_trade_orchestration
python scripts/run_us_buy_report.py --trade-date 2026-05-14 --format json
python scripts/run_us_sell_automation.py --trade-date 2026-05-14
python scripts/run_us_buy_readiness.py --days 60 --format json
```

### Phase 8-10 Intended Daily Reading Order

1. confirm orchestration scheduler executed
2. confirm integrated report exists
3. review Daily Overview
4. review BUY and SELL monitors
5. review conflict summary
6. review Paper portfolio and performance
7. review health and readiness

### Phase 8-10 Dashboard Safety Notes

- dashboard is Paper-only
- dashboard is read-only
- dashboard output must clearly label all performance as `Paper`
- `live_ready=true` must never be interpreted as automatic LIVE release

## Phase 8-11 Dashboard Report Commands

```powershell
python -m python.us.dashboard.run_us_dashboard_report --force
python -m python.us.dashboard.run_us_dashboard_report --trade-date 2026-05-14 --force
python -m python.us.dashboard.run_us_dashboard_report --format json --force
python -m python.us.dashboard.run_us_dashboard_report --format markdown --force
python scripts/run_us_dashboard_report.py --force
```

### Phase 8-11 Output Locations

- date-based dashboard files:
  - `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.json`
  - `reports/lee_trader_us/dashboard/YYYY-MM-DD_dashboard.md`
- latest rolling files:
  - `reports/lee_trader_us/dashboard/latest_dashboard.json`
  - `reports/lee_trader_us/dashboard/latest_dashboard.md`

### Phase 8-11 Reading Order

1. review `Daily Overview`
2. review `Paper Portfolio`
3. review `BUY Decision Monitor`
4. review `SELL Decision Monitor`
5. review `Conflict Guard Monitor`
6. review `Paper Performance`
7. review `Risk / Data Quality`
8. review `Scheduler / Health Check`
9. review `LIVE Readiness`

### Phase 8-11 Safety Notes

- dashboard report is Paper-only
- dashboard report is read-only
- missing data is surfaced, not backfilled by guesswork
- `latest_dashboard.*` is a convenience pointer only
- no real order, broker, or real-account integration exists in this phase

## Phase 8-12 Dashboard Scheduler / Notification Commands

```powershell
python -m python.us.trade_orchestration.scheduler_job
python -m python.us.dashboard.run_us_dashboard_report --force
```

### Phase 8-12 Execution Order

1. scheduler guard
2. run lock
3. trade orchestration
4. integrated report
5. dashboard report
6. health check with dashboard validation
7. notification payload generation
8. operations checklist
9. lock release

### Phase 8-12 Notification Output

- `reports/lee_trader_us/dashboard/notifications/YYYY-MM-DD_notification.txt`
- `reports/lee_trader_us/dashboard/notifications/YYYY-MM-DD_notification.json`
- `reports/lee_trader_us/dashboard/notifications/latest_notification.txt`
- `reports/lee_trader_us/dashboard/notifications/latest_notification.json`

### Phase 8-12 Safety Notes

- notification payload generation is file-only
- no SMTP send
- no Slack webhook send
- no external API call
- dashboard failure is isolated by default unless explicit fail-fast ENV is enabled

## Phase 8-13 Notification Adapter Design Notes

Phase 8-13 is design-only.

- no notification adapter runner is implemented yet
- no actual email delivery is implemented
- no actual Slack delivery is implemented
- no approval UI is implemented
- no external API call is allowed

### Phase 8-13 Intended Future Execution Order

1. trade orchestration
2. integrated report
3. dashboard report
4. dashboard-aware health check
5. notification payload generation
6. notification adapter routing
7. scheduler final result logging

### Phase 8-13 Operator Review Checklist

1. confirm notification payload still says `Paper Trading only`
2. confirm `live_orders_executed=false`
3. confirm FILE / CONSOLE remain the only default channels
4. confirm `US_NOTIFICATION_ADAPTER_MODE` is not treated as trading approval
5. confirm `LIVE` notification mode stays blocked until a future implementation phase
6. confirm no sensitive fields appear in dry-run text or JSON

## Phase 8-14 Notification Adapter Commands

```powershell
python -m python.us.notification.run_us_notification_adapter --force
python -m python.us.notification.run_us_notification_adapter --trade-date 2026-05-15 --force
python -m python.us.notification.run_us_notification_adapter --channels FILE,CONSOLE --force
python scripts/run_us_notification_adapter.py --force
```

### Phase 8-14 Output

- `reports/lee_trader_us/notification/YYYY-MM-DD_notification_adapter.txt`
- `reports/lee_trader_us/notification/YYYY-MM-DD_notification_adapter.json`
- `reports/lee_trader_us/notification/latest_notification_adapter.txt`
- `reports/lee_trader_us/notification/latest_notification_adapter.json`
- `reports/lee_trader_us/notification/approvals/YYYY-MM-DD_approval_pending.json`
- `reports/lee_trader_us/notification/approvals/latest_approval_pending.json`

### Phase 8-14 Safety Notes

- all channels are dry-run only
- no SMTP send
- no Slack webhook send
- no external API call
- `EMAIL_LIVE` and `SLACK_LIVE` are explicitly blocked
- approval pending is for notification delivery review only, not for trading approval

## Phase 8-15 Quality Gate Design Notes

Phase 8-15 is design-only.

- no quality-gate evaluator is implemented yet
- no Go-Live approval automation is implemented
- no LIVE enablement is allowed
- no broker or real-account integration is added

### Phase 8-15 Intended Future Execution Order

1. trade orchestration
2. dashboard report
3. notification adapter dry-run
4. quality-gate evaluation
5. scheduler final result logging

### Phase 8-15 Operator Review Checklist

1. confirm data-quality gate inputs are complete
2. confirm BUY / SELL / conflict reasons remain explainable
3. confirm integrated report, dashboard, and notification counts agree
4. confirm scheduler success and health-check pass rates are stable
5. confirm notification artifacts remain Paper-only and non-sensitive
6. confirm LIVE safety signals remain fully blocked
7. confirm Go-Live checklist is treated as review-only, not approval

## Backfill Method

- use `US_STOCK_PRICE_BACKFILL_YEARS` as the default historical range
- allow explicit override with `US_STOCK_PRICE_START_DATE` and `US_STOCK_PRICE_END_DATE`
- run backfill as a US-only standalone operation

## Incremental Collection Method

- run the US pipeline separately from Korean scheduling
- collect only the latest missing US trading dates
- write collection status to `market.us_stock_data_collect_log`

## Data Quality Checks

- verify missing rows by universe member
- verify stale latest trade date
- verify invalid OHLC relationships
- verify invalid volume values
- remember stale checks are calendar-day based in Phase 1

## Feature Build Notes

- `build_us_features.py` creates only baseline price features
- it is not AI/ML model training
- it does not create final ranking scores
- it does not connect to Korean auto-trading

## Pipeline Notes

- `run_us_daily_pipeline.py` is the standalone US-only Phase 1 runner
- if `US_STOCK_ENABLED=false`, the pipeline is skipped unless `--force` is provided
- `--skip-universe`, `--skip-prices`, `--skip-quality`, `--skip-features` are for controlled manual runs
- if quality status is `FAILED`, feature generation is skipped by default
- `--force-features` can override the default feature skip after a quality failure

## Feature Failure Check Order

1. confirm `DATABASE_URL`
2. confirm US migrations applied
3. confirm active universe rows exist
4. confirm latest price rows exist in `market.us_stock_daily_price`
5. confirm quality report status if needed
6. confirm feature rows exist in `feature.us_stock_feature_daily`
7. inspect pipeline summary logs and ticker-level logs from manual execution

## Safety Principle

The US stock pipeline must not affect Korean auto-trading.

- no connection to `run_daily_scheduler.py` in this phase
- no connection to KIS order files
- no paper trading code
- no live trading code
- US pipeline failure must not break Korean trading operations

## Phase 3-2 Ranking Table Checks

1. confirm `migrations/us_stock_phase3_2_rank_table.sql` is applied
2. run `python scripts/verify_us_stock_rank_table.py`
3. if DB validation is needed, run `python scripts/verify_us_stock_rank_table.py --write-sample --cleanup`
4. confirm `recommend.us_stock_rank_daily` primary key is `(trade_date, symbol)`
5. confirm ranking indexes exist for date/rank, date/grade, and symbol/date

## Ranking Query Examples

Top 20:

```sql
SELECT
    trade_date,
    rank_no,
    symbol,
    company_name,
    recommend_grade,
    total_score,
    momentum_score,
    relative_strength_score,
    fundamental_score,
    growth_score,
    valuation_score,
    risk_score,
    reason_summary
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
ORDER BY rank_no
LIMIT 20;
```

History:

```sql
SELECT
    trade_date,
    symbol,
    rank_no,
    recommend_grade,
    total_score
FROM recommend.us_stock_rank_daily
WHERE symbol = 'AAPL'
ORDER BY trade_date DESC;
```

## Phase 3-3 Rule Ranking Checks

1. confirm `recommend.us_stock_rank_daily` exists
2. run `python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --dry-run`
3. run `python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --symbols AAPL,MSFT,NVDA --dry-run`
4. review the logged effective US trade date if the requested date is not a US session date
5. confirm `risk_score` stays between `-10` and `0`
6. confirm `score_detail_json` contains missing field lists instead of mutating source tables

## Rule Ranking Safety Notes

- Phase 3-3 calculates ranking scores only.
- It does not touch Korean order logic, KIS execution, or live-trading schedulers.
- `--dry-run` performs no DB writes.
- A full-universe non-dry-run execution refreshes the same `trade_date + source` snapshot before upserting rows.

## Phase 3-4 Report Checks

1. run `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20`
2. confirm rows are ordered by `rank_no ASC`
3. confirm `recommend_grade = EXCLUDE` rows are absent from the default Top N result
4. run `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --format markdown`
5. confirm markdown is written under `outputs/us_stock_top_rank/`
6. run `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --format csv`
7. confirm UTF-8 csv is written under `outputs/us_stock_top_rank/`
8. run `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA`
9. confirm score breakdown and `score_detail_json` preview are visible
10. if the date has no ranking rows, confirm the script prints `calculate_us_stock_rule_scores.py` guidance or use `--auto-calculate`

## Report Safety Notes

- Phase 3-4 is still a reporting layer only.
- The report must not be interpreted as an order submission command.
- No Korean trading scheduler, KIS execution flow, or live-order module is touched by this script.
- Optional notifier integration is disabled by default and must remain best-effort only.

## Phase 3-5 Validation Checks

1. confirm `reason_summary` is populated for all ranked rows
2. confirm `score_detail_json.meta.reason_category` exists
3. confirm `score_detail_json.meta.reason_tags` exists as a list
4. confirm `recommend_grade = EXCLUDE` rows have `exclude_reason`
5. run `python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12`
6. confirm score range errors stay at `0`
7. review anomaly warnings separately from exclusion policy
8. use `--show-excluded` when Top N eligible output is empty

## Phase 4-1 Backtest Checks

1. confirm `migrations/us_stock_phase4_1_rank_backtest.sql` is applied
2. confirm `recommend.us_stock_rank_daily` has historical rows across the requested period
3. run `python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --holding-days 5,20,60 --dry-run`
4. confirm `entry_date > trade_date`
5. confirm `exit_date > entry_date`
6. confirm `return_pct` uses next-session entry and forward exit
7. confirm `spy_return_pct` / `qqq_return_pct` populate when benchmark prices exist
8. confirm same `backtest_id` re-run does not create duplicate PK rows
9. review `data_status` for `NOT_ENOUGH_FORWARD_DATA` near the latest dates
10. treat results as research only, not as trade instructions

## Phase 4-2 Backtest Report Checks

1. run `python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format console`
2. confirm strategy rows are grouped by `strategy_name + holding_days`
3. confirm `%` formatting is human-readable
4. confirm `Best Candidate` is shown only as a review candidate
5. run markdown and csv output once and confirm files land under `outputs/us_stock_backtest/`
6. run `--strategy ... --holding-days ...` and confirm best/worst day sections filter correctly
7. run `--symbol NVDA` and confirm symbol-level holding-day summary output
8. if `backtest_id` has no rows, confirm a clear guidance message is printed

## Phase 4-3 Regime Analysis Checks

1. confirm `migrations/us_stock_phase4_3_market_regime.sql` is applied
2. run `python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12 --dry-run`
3. run `python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12`
4. confirm `research.us_market_regime_daily` rows are upserted without duplicates
5. confirm `spy_regime`, `qqq_regime`, `vol_regime`, and `market_regime` are populated from same-day benchmark data only
6. run `python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format console`
7. confirm monthly and quarterly sections render in markdown/csv outputs
8. if benchmark data is stale relative to backtest `trade_date`, confirm the report shows `UNKNOWN` regime and a guidance message
9. treat all regime findings as research diagnostics only

## Phase 4-4 Weight Experiment Checks

1. confirm `migrations/us_stock_phase4_4_weight_experiment.sql` is applied
2. run the experiment once with `--dry-run` and confirm no experiment rows are written
3. confirm `RULE_V1_BASELINE` reproduces the stored baseline `total_score`
4. confirm experiment rows are written to `research.us_stock_rank_weight_experiment_result`
5. confirm aggregate comparison rows are written to `research.us_stock_weight_experiment_backtest_summary`
6. confirm `recommend.us_stock_rank_daily` is unchanged by the experiment run
7. confirm `PROMOTE_CANDIDATE` / `WATCH_CANDIDATE` / `REJECT_CANDIDATE` wording is treated as review-only
8. if all weight configs still produce `EXCLUDE`, treat the output as insufficient-data evidence rather than a winning candidate

## Phase 4-5 Forward Test Checks

1. confirm `migrations/us_stock_phase4_5_forward_test.sql` is applied
2. run `python scripts/register_us_stock_forward_test.py --trade-date 2026-05-12 --forward-test-id US_RANK_FORWARD_RULE_V1 --holding-days 5,20,60 --dry-run`
3. confirm the same `trade_date + strategy + symbol + holding_days` re-run does not duplicate rows
4. run `python scripts/update_us_stock_forward_entry.py --as-of-date 2026-05-13 --forward-test-id US_RANK_FORWARD_RULE_V1 --dry-run`
5. confirm `entry_date` is the next US trading day after `trade_date`
6. run `python scripts/update_us_stock_forward_exit.py --as-of-date 2026-06-12 --forward-test-id US_RANK_FORWARD_RULE_V1 --dry-run`
7. confirm incomplete horizons remain `ACTIVE` or `PENDING_EXIT`
8. run `python scripts/update_us_stock_forward_summary.py --forward-test-id US_RANK_FORWARD_RULE_V1 --dry-run`
9. run `python scripts/report_us_stock_forward_test.py --forward-test-id US_RANK_FORWARD_RULE_V1 --format console`
10. treat all Forward Test output as post-recommendation diagnostics only, not as a paper/live trading signal

## Phase 5-1 Paper Trading Checks

1. confirm `migrations/us_stock_phase5_1_paper_trading.sql` is applied
2. run `python scripts/init_us_stock_paper_account.py --account-id US_PAPER_RULE_V1 --initial-cash 100000 --dry-run`
3. confirm dry-run performs no DB write
4. run `python scripts/init_us_stock_paper_account.py --account-id US_PAPER_RULE_V1 --initial-cash 100000`
5. confirm `paper.us_stock_paper_account` has one row with matching `initial_cash` and `cash_balance`
6. confirm `paper.us_stock_paper_order`, `paper.us_stock_paper_fill`, `paper.us_stock_paper_position`, and `paper.us_stock_paper_account_snapshot` all exist
7. confirm rerun without `--reset` does not wipe the account
8. confirm `--reset` warning clearly states only `paper.us_stock_*` rows are touched
9. confirm no broker API or real-order module is called anywhere in the init path

## Phase 5-2 Paper Order Generation Checks

1. run `python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-12 --account-id US_PAPER_RULE_V1 --dry-run`
2. run BUY-only and SELL-only dry-runs once each
3. confirm writes target only `paper.us_stock_paper_order`
4. confirm no `paper_fill`, `paper_position`, `paper_account` balance mutation occurs
5. confirm inactive paper accounts are blocked
6. confirm duplicate `account_id + trade_date + symbol + side + strategy_name` orders are skipped by default
7. confirm `--replace-existing` updates only unfilled paper orders
8. confirm reject reasons use standardized codes such as `insufficient_cash`, `missing_order_price`, and `already_target_weight`

## Phase 5-3 Paper Fill Simulation Checks

1. run `python scripts/simulate_us_stock_paper_fills.py --as-of-date 2026-05-13 --account-id US_PAPER_RULE_V1 --dry-run`
2. confirm only `paper.us_stock_paper_order.status = 'CREATED'` rows are treated as fill candidates
3. confirm the simulated `fill_date` is the next US trading day after `order.trade_date`
4. confirm BUY fills apply positive slippage and SELL fills apply negative slippage
5. confirm commission reduces cash on BUY and reduces proceeds on SELL
6. confirm live execution writes only `paper.us_stock_paper_fill`, `paper.us_stock_paper_position`, and `paper.us_stock_paper_account`
7. confirm already `FILLED` orders are not reprocessed on rerun
8. confirm `validate_paper_account_integrity()` reports no issues after a successful simulation run

## Phase 5-4 Paper Snapshot And Report Checks

1. run `python scripts/update_us_stock_paper_snapshot.py --snapshot-date 2026-05-14 --account-id US_PAPER_RULE_V1 --dry-run`
2. confirm only `OPEN` positions are valuation targets
3. confirm `market_value = qty * last_price`
4. confirm `unrealized_pnl = market_value - cost_amount`
5. confirm `equity_value = cash_balance + market_value`
6. confirm `total_pnl = realized_pnl + unrealized_pnl`
7. run the non-dry snapshot update once and confirm `paper.us_stock_paper_account_snapshot` upserts by `account_id + snapshot_date`
8. rerun the same `snapshot_date` and confirm no duplicate snapshot row is created
9. run `python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console`
10. confirm markdown/csv report outputs are written under `US_PAPER_REPORT_OUTPUT_DIR`

## Phase 5-5 Rebalance And Validation Checks

1. run `python scripts/plan_us_stock_paper_rebalance.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1 --dry-run`
2. confirm the plan shows SELL candidates first and BUY candidates after current-position review
3. confirm BUY candidates respect `US_PAPER_REBALANCE_MIN_AMOUNT` and `US_PAPER_REBALANCE_MIN_WEIGHT_DIFF`
4. confirm same-day rebuy is skipped when `US_PAPER_REBALANCE_ALLOW_REBUY_SAME_DAY=false`
5. run `python scripts/generate_us_stock_paper_orders.py --trade-date 2026-05-15 --account-id US_PAPER_RULE_V1 --dry-run`
6. confirm the generated paper orders are consistent with the rebalance plan and still write only to `paper.us_stock_paper_order`
7. run `python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1`
8. confirm validation output separates warnings from errors and does not call any broker API
9. run `python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console`
10. confirm the report includes operation-status fields such as last order date, last fill date, created/rejected order counts, and validation warning counts

## Phase 5 Daily Paper Pipeline

1. collect US prices and rebuild required features
2. calculate the latest Rule ranking snapshot
3. review `python scripts/plan_us_stock_paper_rebalance.py --trade-date ... --account-id US_PAPER_RULE_V1`
4. generate paper orders with `python scripts/generate_us_stock_paper_orders.py --trade-date ... --account-id US_PAPER_RULE_V1`
5. simulate fills on the next US trading day with `python scripts/simulate_us_stock_paper_fills.py --as-of-date ... --account-id US_PAPER_RULE_V1`
6. refresh the paper snapshot with `python scripts/update_us_stock_paper_snapshot.py --snapshot-date ... --account-id US_PAPER_RULE_V1`
7. run `python scripts/validate_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1`
8. publish the review report with `python scripts/report_us_stock_paper_trading.py --account-id US_PAPER_RULE_V1 --format console`

## Phase 6-3 Pre-Trade Check

1. validate the reviewed live risk policy with `python scripts/validate_us_live_risk_policy.py --policy-id US_LIVE_RULE_V1 --format console`
2. ensure default kill-switch and daily usage rows exist with `python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1`
3. run a manual dry-run candidate with `python scripts/run_us_live_pre_trade_check.py --trade-date ... --account-id US_LIVE_TEST --symbol NVDA --side BUY --amount-usd 50 --dry-run`
4. run a ranking-driven dry-run batch with `python scripts/run_us_live_pre_trade_check.py --trade-date ... --account-id US_LIVE_TEST --from-ranking --top-n 20 --side BUY --dry-run`
5. treat `ALLOW` as policy eligibility only, not as an order instruction
6. if `BLOCK` or `ERROR` is persisted, review `risk.us_stock_live_order_block_log`

## Phase 6-4 Kill Switch

1. ensure default state rows exist with `python scripts/init_us_live_risk_state.py --policy-id US_LIVE_RULE_V1`
2. inspect current state with `python scripts/manage_us_live_kill_switch.py --list`
3. use `--activate --scope GLOBAL|BUY|SELL|SYMBOL|SECTOR|ACCOUNT` for emergency stops only
4. require `--reason-code`, `--reason-detail`, and `--performed-by` on activation
5. require `--clear-reason` and `--performed-by` on clear
6. run `python scripts/evaluate_us_live_kill_switch.py --trade-date ... --account-id ... --policy-id US_LIVE_RULE_V1 --dry-run` before any auto-activation review
7. treat kill-switch activation as a safety-state change only, not as a broker or execution action

## Phase 3-1 Universe Rules

- `meta.us_stock_universe` is the recommendation-universe master table
- the initial seed merges:
  - S&P500
  - NASDAQ100
  - selected major ETFs
- leveraged and inverse ETFs are inserted with inactive status and exclusion reasons
- filtered active candidates are resolved by:
  1. active meta-universe flag
  2. ETF inclusion policy
  3. leveraged / inverse exclusion flags
  4. price availability
  5. recent average volume threshold
  6. market cap threshold
  7. derived feature-quality threshold

## Universe Validation Checks

1. confirm `meta.us_stock_universe` migration is applied
2. run `python scripts/init_us_stock_universe.py --dry-run`
3. run `python scripts/init_us_stock_universe.py`
4. confirm duplicate-free symbols in `meta.us_stock_universe`
5. confirm leveraged / inverse ETF rows are inactive
6. confirm filtered active candidate count is logged
