# Lee_trader_us File Index

> 문서 역할: `현재 기준 문서`
>
> 구현 파일 위치를 찾기 위한 인덱스 문서다. 코드 수정이나 스크립트 추적 시 사용한다.

## Python Files

- `python/us/__init__.py`: package marker for the US module
- `python/us/us_config.py`: US environment configuration loader
- `python/us/us_db.py`: shared DB access and upsert/query helpers for US-only modules
- `python/us/load_us_universe.py`: NASDAQ100 universe seed loader
- `python/us/download_us_prices.py`: US OHLCV collector using `yfinance`
- `python/us/validate_us_price_data.py`: daily US price quality validator and quality report writer
- `python/us/build_us_features.py`: baseline price feature builder for `feature.us_stock_feature_daily`
- `python/us/build_us_ranking_placeholder.py`: placeholder entry point for future ranking stage
- `python/us/calculate_us_stock_rule_scores.py`: Phase 3-3 Rule-based ranking calculator and DB writer
- `python/us/report_us_stock_top_rank.py`: Phase 3-4 Top N ranking report reader and formatter
- `python/us/validate_us_stock_rank_daily.py`: Phase 3-5 ranking validation and anomaly checker
- `python/us/backtest_us_stock_rank_strategy.py`: Phase 4-1 rank backtest engine for stored ranking snapshots
- `python/us/report_us_stock_rank_backtest.py`: Phase 4-2 strategy performance report builder for stored backtest outputs
- `python/us/build_us_market_regime_daily.py`: Phase 4-3 benchmark regime snapshot builder using SPY/QQQ price history
- `python/us/analyze_us_stock_backtest_by_regime.py`: Phase 4-3 regime and period analysis report builder for stored backtest summaries
- `python/us/experiment_us_stock_rule_weights.py`: Phase 4-4 weight-candidate experiment runner using stored Rule component scores
- `python/us/report_us_stock_rule_weight_experiment.py`: Phase 4-4 weight-candidate comparison report builder
- `python/us/forward_test_us_stock.py`: Phase 4-5 forward-test registration, update, summary, and report helpers
- `python/us/init_us_stock_paper_account.py`: Phase 5-1 paper account bootstrap and safety checks
- `python/us/generate_us_stock_paper_orders.py`: Phase 5-2 paper-only virtual order generation from ranking snapshots
- `python/us/simulate_us_stock_paper_fills.py`: Phase 5-3 paper-only virtual fill simulation and account/position updates
- `python/us/update_us_stock_paper_snapshot.py`: Phase 5-4 paper account valuation and daily snapshot builder
- `python/us/report_us_stock_paper_trading.py`: Phase 5-4 paper account and performance report builder
- `python/us/paper_rebalance.py`: Phase 5-5 shared paper rebalance policy and order-planning helpers
- `python/us/plan_us_stock_paper_rebalance.py`: Phase 5-5 paper rebalance planning report builder
- `python/us/validate_us_stock_paper_trading.py`: Phase 5-5 paper trading operating validation and integrity summary
- `python/us/validate_us_live_risk_policy.py`: Phase 6-2 live risk-policy default validation report builder
- `python/us/init_us_live_risk_state.py`: Phase 6-2 live risk kill-switch and daily-usage state initializer
- `python/us/run_us_live_pre_trade_check.py`: Phase 6-3 live pre-trade candidate validation runner
- `python/us/manage_us_live_kill_switch.py`: Phase 6-4 manual kill-switch state management runner
- `python/us/evaluate_us_live_kill_switch.py`: Phase 6-4 automatic kill-switch trigger evaluation runner
- `python/us/manage_us_live_order_approval.py`: Phase 6-5 live order-approval state management runner
- `python/us/create_us_micro_order_request.py`: Phase 7 Micro order-request creation entry point
- `python/us/manage_us_micro_order.py`: Phase 7 Micro order detail/list/status management helper runner
- `python/us/send_us_micro_order_mock.py`: Phase 7 mock Micro order send runner
- `python/us/send_us_micro_order_sandbox.py`: Phase 7 sandbox Micro order send runner
- `python/us/send_us_micro_order_live.py`: Phase 7 gated live Micro order send runner
- `python/us/sync_us_micro_order_status.py`: Phase 7-4 broker-status and fill sync runner
- `python/us/report_us_micro_order_status.py`: Phase 7-4 Micro order status/fill report runner
- `python/us/reconcile_us_micro_live.py`: Phase 7-5 reconciliation runner
- `python/us/report_us_micro_live_operations.py`: Phase 7-6 integrated daily operations report runner
- `python/us/run_us_micro_live_daily_check.py`: Phase 7-6 wrapper that combines ops report and reconciliation check
- `python/us/init_us_stock_universe.py`: recommendation universe master initializer for `meta.us_stock_universe`
- `python/us/us_rank_design.py`: Phase 3-2 ranking table constants and validation sample payloads
- `python/us/verify_us_stock_rank_table.py`: ranking table validation helper for dry-run/sample insert checks
- `python/us/run_us_daily_pipeline.py`: standalone Phase 1 daily pipeline orchestrator for universe, prices, quality, and features
- `python/us/run_us_buy_automation.py`: Phase 8-2 wrapper entry for limited BUY automation skeleton
- `python/us/run_us_buy_scheduler_job.py`: Phase 8-4 wrapper entry for BUY scheduler job
- `python/us/run_us_buy_readiness.py`: Phase 8-5 wrapper entry for readiness evaluation

## Python Package Files

- `python/us/buy_automation/__init__.py`: package exports for Phase 8 BUY automation
- `python/us/buy_automation/config.py`: Phase 8-2 BUY automation ENV loader and safe defaults
- `python/us/buy_automation/candidate_loader.py`: Phase 8-2 latest ranking candidate loader and snapshot enrichment
- `python/us/buy_automation/risk_guard.py`: Phase 8-2 fail-safe BUY rule evaluation
- `python/us/buy_automation/decision_engine.py`: Phase 8-2 SHADOW/PAPER/LIVE-blocked decision pipeline
- `python/us/buy_automation/paper_order.py`: Phase 8-2 internal PAPER order skeleton builder
- `python/us/buy_automation/logger.py`: Phase 8-2 DB-or-JSON logging helper
- `python/us/buy_automation/run_us_buy_automation.py`: Phase 8-2 executable entry module
- `python/us/buy_automation/report_generator.py`: Phase 8-3 BUY automation daily report builder
- `python/us/buy_automation/validation_summary.py`: Phase 8-3 block/rule summary helper
- `python/us/buy_automation/paper_performance.py`: Phase 8-3 PAPER performance tracker
- `python/us/buy_automation/notification_formatter.py`: Phase 8-3 notification text formatter
- `python/us/buy_automation/run_us_buy_report.py`: Phase 8-3 report entry module
- `python/us/buy_automation/scheduler_job.py`: Phase 8-4 scheduler-safe BUY automation wrapper and failure isolation layer
- `python/us/buy_automation/performance_metrics.py`: Phase 8-5 pure performance-metric helpers
- `python/us/buy_automation/paper_backtest_summary.py`: Phase 8-5 cumulative PAPER performance window summary builder
- `python/us/buy_automation/promotion_policy.py`: Phase 8-5 readiness policy / threshold loader
- `python/us/buy_automation/live_readiness_evaluator.py`: Phase 8-5 LIVE readiness evaluator
- `python/us/buy_automation/run_us_buy_readiness.py`: Phase 8-5 readiness-report entry module

## Documents

- `doc/modules/Lee_trader_us/README.md`: module purpose and current phase scope
- `doc/modules/Lee_trader_us/CONTEXT.md`: architecture boundaries and long-term direction
- `doc/modules/Lee_trader_us/ARCHITECTURE.md`: current runtime architecture and Phase 8-4 scheduler integration boundary
- `doc/modules/Lee_trader_us/FLOW.md`: target pipeline flow and current phase boundary
- `doc/modules/Lee_trader_us/ENV.md`: US_STOCK environment variable reference
- `doc/modules/Lee_trader_us/OPERATIONS.md`: manual operations and failure checks
- `doc/modules/Lee_trader_us/DB_SCHEMA.md`: current US schema summary and Phase 8-1 proposed BUY tables
- `doc/modules/Lee_trader_us/BUY_AUTOMATION_DESIGN.md`: Phase 8-1 limited BUY automation design
- `doc/modules/Lee_trader_us/RANKING.md`: Phase 3-2 ranking result table design
- `doc/modules/Lee_trader_us/US_STOCK_RANKING_V1.md`: integrated Phase 3 operations guide
- `doc/modules/Lee_trader_us/PHASE3_CHECKLIST.md`: short-form Phase 3 completion checklist
- `doc/modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md`: Phase 4-1 rank backtest design and execution guide
- `doc/modules/Lee_trader_us/US_STOCK_PAPER_TRADING.md`: Phase 5-1 paper trading structure and account bootstrap guide
- `doc/modules/Lee_trader_us/US_STOCK_LIVE_TRADING_POLICY.md`: Phase 6-1 live-trading safety policy and Micro Live entry criteria
- `doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md`: Phase 6-2 live risk-policy structure, risk tables, and safe-default validation guide
- `doc/modules/Lee_trader_us/US_STOCK_LIVE_OPERATION_RUNBOOK.md`: Phase 6-6 integrated live-safety runbook, operating steps, and Phase 7 entry checklist
- `doc/modules/Lee_trader_us/PHASE7_SUMMARY.md`: Phase 7 completion summary, document map, and next-step handoff

## Script Files

- `scripts/calculate_us_stock_rule_scores.py`: wrapper entry point for Phase 3-3 ranking runs
- `scripts/report_us_stock_top_rank.py`: wrapper entry point for Phase 3-4 ranking reports
- `scripts/validate_us_stock_rank_daily.py`: wrapper entry point for Phase 3-5 ranking validation
- `scripts/backtest_us_stock_rank_strategy.py`: wrapper entry point for Phase 4-1 rank backtest runs
- `scripts/report_us_stock_rank_backtest.py`: wrapper entry point for Phase 4-2 backtest performance reports
- `scripts/build_us_market_regime_daily.py`: wrapper entry point for Phase 4-3 market regime snapshot builds
- `scripts/analyze_us_stock_backtest_by_regime.py`: wrapper entry point for Phase 4-3 regime analysis reports
- `scripts/experiment_us_stock_rule_weights.py`: wrapper entry point for Phase 4-4 weight experiments
- `scripts/report_us_stock_rule_weight_experiment.py`: wrapper entry point for Phase 4-4 weight experiment reports
- `scripts/register_us_stock_forward_test.py`: wrapper entry point for Phase 4-5 forward-test registration
- `scripts/update_us_stock_forward_entry.py`: wrapper entry point for Phase 4-5 forward-test entry updates
- `scripts/update_us_stock_forward_exit.py`: wrapper entry point for Phase 4-5 forward-test exit updates
- `scripts/update_us_stock_forward_summary.py`: wrapper entry point for Phase 4-5 forward-test summary refresh
- `scripts/report_us_stock_forward_test.py`: wrapper entry point for Phase 4-5 forward-test reporting
- `scripts/init_us_stock_paper_account.py`: wrapper entry point for Phase 5-1 paper account initialization
- `scripts/generate_us_stock_paper_orders.py`: wrapper entry point for Phase 5-2 paper order generation
- `scripts/simulate_us_stock_paper_fills.py`: wrapper entry point for Phase 5-3 paper fill simulation
- `scripts/update_us_stock_paper_snapshot.py`: wrapper entry point for Phase 5-4 paper snapshot refresh
- `scripts/report_us_stock_paper_trading.py`: wrapper entry point for Phase 5-4 paper trading report
- `scripts/plan_us_stock_paper_rebalance.py`: wrapper entry point for Phase 5-5 paper rebalance planning
- `scripts/validate_us_stock_paper_trading.py`: wrapper entry point for Phase 5-5 paper operating validation
- `scripts/validate_us_live_risk_policy.py`: wrapper entry point for Phase 6-2 live risk-policy validation
- `scripts/init_us_live_risk_state.py`: wrapper entry point for Phase 6-2 live risk-state initialization
- `scripts/run_us_live_pre_trade_check.py`: wrapper entry point for Phase 6-3 pre-trade candidate checks
- `scripts/manage_us_live_kill_switch.py`: wrapper entry point for Phase 6-4 kill-switch listing, activation, and clear
- `scripts/evaluate_us_live_kill_switch.py`: wrapper entry point for Phase 6-4 auto-trigger evaluation
- `scripts/manage_us_live_order_approval.py`: wrapper entry point for Phase 6-5 approval listing, detail, approve, reject, and expire actions
- `scripts/create_us_micro_order_request.py`: wrapper entry point for Phase 7 Micro order-request creation
- `scripts/manage_us_micro_order.py`: wrapper entry point for Phase 7 Micro order inspection and state actions
- `scripts/send_us_micro_order_mock.py`: wrapper entry point for Phase 7 mock send path
- `scripts/send_us_micro_order_sandbox.py`: wrapper entry point for Phase 7 sandbox send path
- `scripts/send_us_micro_order_live.py`: wrapper entry point for Phase 7 gated live send path
- `scripts/sync_us_micro_order_status.py`: wrapper entry point for Phase 7-4 status/fill sync
- `scripts/report_us_micro_order_status.py`: wrapper entry point for Phase 7-4 status report
- `scripts/reconcile_us_micro_live.py`: wrapper entry point for Phase 7-5 reconciliation
- `scripts/report_us_micro_live_operations.py`: wrapper entry point for Phase 7-6 operations report
- `scripts/run_us_micro_live_daily_check.py`: wrapper entry point for Phase 7-6 daily operational check
- `scripts/run_us_buy_automation.py`: wrapper entry point for Phase 8-2 limited BUY automation skeleton
- `scripts/run_us_buy_report.py`: wrapper entry point for Phase 8-3 BUY automation report
- `scripts/run_us_buy_scheduler_job.py`: wrapper entry point for Phase 8-4 BUY scheduler job
- `scripts/run_us_buy_readiness.py`: wrapper entry point for Phase 8-5 readiness evaluation
- `scripts/verify_us_stock_rank_table.py`: wrapper entry point for Phase 3-2 table validation

## DDL

- `migrations/us_stock_phase1.sql`: Project C Phase 1 US stock schema bootstrap
- `migrations/us_stock_phase1_2_universe.sql`: Phase 1-2 universe table adjustment migration
- `migrations/us_stock_phase1_3_price_collect.sql`: Phase 1-3 OHLCV price and collect-log alignment
- `migrations/us_stock_phase1_4_quality_report.sql`: Phase 1-4 quality report table alignment
- `migrations/us_stock_phase1_5_feature_daily.sql`: Phase 1-5 feature table alignment
- `migrations/us_stock_phase3_1_meta_universe.sql`: Phase 3-1 recommendation universe master schema
- `migrations/us_stock_phase3_2_rank_table.sql`: Phase 3-2 ranking result table schema
- `migrations/us_stock_phase4_1_rank_backtest.sql`: Phase 4-1 backtest result and summary schema
- `migrations/us_stock_phase4_3_market_regime.sql`: Phase 4-3 market regime and regime-summary schema
- `migrations/us_stock_phase4_4_weight_experiment.sql`: Phase 4-4 weight config and experiment result schema
- `migrations/us_stock_phase4_5_forward_test.sql`: Phase 4-5 forward-test detail and summary schema
- `migrations/us_stock_phase5_1_paper_trading.sql`: Phase 5-1 paper account/order/fill/position/snapshot schema
- `migrations/us_stock_phase6_2_live_risk.sql`: Phase 6-2 live risk policy state schema
- `migrations/us_stock_phase6_4_kill_switch.sql`: Phase 6-4 kill-switch target/event-log schema extension
- `migrations/us_stock_phase6_5_live_order_approval.sql`: Phase 6-5 approval request and approval event-log schema
- `migrations/us_stock_phase7_1_micro_live_mock.sql`: Phase 7-1 Micro order request/event-log schema
- `migrations/us_stock_phase7_4_micro_order_sync.sql`: Phase 7-4 sync/fill schema extension
- `migrations/us_stock_phase7_5_micro_reconciliation.sql`: Phase 7-5 reconciliation result/event-log schema
- `sql/lee_trader_us/phase8_2_buy_automation_tables.sql`: Phase 8-2 proposed BUY automation log/paper-order DDL sketch
- `sql/lee_trader_us/phase8_3_buy_report_tables.sql`: Phase 8-3 proposed report/performance snapshot DDL sketch
- `sql/lee_trader_us/phase8_5_live_readiness_tables.sql`: Phase 8-5 proposed readiness/performance/promotion-check DDL sketch

## Config And Utils

- `config/us_stock_paper_trading.yaml`: Phase 5 paper-only trading policy defaults
- `config/us_stock_live_risk_policy.yaml`: Phase 6-2 Micro Live risk policy defaults with safe gates disabled
- `utils/paper_trading_safety.py`: paper-only safety assertion helper
- `utils/us_live_risk_policy.py`: Phase 6-2 YAML + ENV live risk-policy loader and safe-default validator
- `utils/us_live_trading_safety.py`: shared live pre-trade safety assertion helper that still blocks real-order paths
- `utils/us_live_pre_trade_check.py`: Phase 6-3 staged pre-trade validation and block-log writer
- `utils/us_live_kill_switch.py`: Phase 6-4 scoped kill-switch management and auto-trigger evaluation helper
- `utils/us_live_order_approval.py`: Phase 6-5 approval-request lifecycle and approval validation helper
- `utils/us_micro_order_request.py`: Phase 7 Micro order lifecycle helper
- `utils/us_order_client_interface.py`: Phase 7 broker order-client interface
- `utils/us_mock_order_client.py`: Phase 7 mock order client
- `utils/us_sandbox_order_client.py`: Phase 7 sandbox order client
- `utils/us_live_order_client.py`: Phase 7 gated live order client
- `utils/us_order_status_mapper.py`: Phase 7-4 raw broker-status to internal-status mapper
- `utils/us_micro_order_sync.py`: Phase 7-4 status/fill sync helper
- `utils/us_broker_account_interface.py`: Phase 7-5 broker account/position lookup interface
- `utils/us_mock_account_client.py`: Phase 7-5 mock account-state adapter
- `utils/us_sandbox_account_client.py`: Phase 7-5 sandbox account-state adapter
- `utils/us_live_account_client.py`: Phase 7-5 gated live account adapter placeholder
- `utils/us_micro_reconciliation.py`: Phase 7-5 reconciliation helper
- `utils/us_micro_live_operations.py`: Phase 7-6 integrated operations reporting helper

## Data Files

- `data/us/nasdaq100_universe.csv`: static NASDAQ100 universe seed file
