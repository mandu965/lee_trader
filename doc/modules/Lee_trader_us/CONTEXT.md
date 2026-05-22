# Lee_trader_us Context

> 문서 역할: `현재 기준 문서`
>
> 왜 이런 구조가 되었는지, 한국 시스템과 어떤 경계를 가지는지, 각 phase가 어떤 의미였는지를 설명한다.

## Relationship To Korean System

- Project C is separated from the Korean auto-trading pipeline.
- It does not modify Korean score calculation, order creation, or KIS execution flow.
- Failure in the US pipeline must not propagate to the Korean daily trading pipeline.

## Relationship To Project B

- Project B handled US macro overlay data for the Korean system.
- Project C is a separate long-term track for direct US stock recommendation.
- Project B and Project C may share some operational ideas, but their runtime paths remain separate.

## Long-Term Goal

The long-term goal is a US stock recommendation system that can evolve through:

- universe management
- market data collection
- feature generation
- score and ranking
- backtest
- paper trading
- live trading review

## Current Operating Position (2026-05-22)

The long-term design history includes live-safety and Micro Live review layers.
But the **current operating position** is narrower:

- US is treated as a `paper-only` track
- live transition is not part of the active short-term operating plan
- current success criteria are paper lifecycle stability, backtest interpretability, and policy validation
- live / Micro Live documents remain useful as reference and design history, but not as immediate rollout instructions

## Phase 1 Limits

Phase 1 is limited to data foundation and standalone validation.

- universe loading is implemented
- yfinance OHLCV collection is implemented
- quality validation is implemented
- baseline feature computation is implemented
- standalone US daily pipeline is implemented
- ranking logic is not implemented
- paper trading is not implemented
- live trading is not implemented
- AI/ML training is not implemented

## Phase 3-1 Extension

Phase 3-1 introduces the first recommendation-universe master for US stocks.

- `market.us_stock_universe` is still used for raw data collection membership
- `meta.us_stock_universe` is used for recommendation candidate management
- the universe master keeps ETF / leveraged / inverse flags and exclusion reasons
- active recommendation candidates are filtered with market cap, liquidity, and feature-availability rules
- ranking scores are still out of scope in this phase

## Phase 3-2 Extension

Phase 3-2 adds the ranking-result storage contract.

- `recommend.us_stock_rank_daily` is the canonical daily result table for US recommendations
- Phase 3-2 stores ranking outputs only and still does not calculate scores
- `risk_score` is stored as a negative penalty so `total_score` stays additive
- `score_detail_json` uses PostgreSQL `jsonb` for future audit/debug usage

## Phase 3-3 Extension

Phase 3-3 implements the first deterministic Rule scorer.

- ranking uses universe metadata plus price/fundamental/relative-strength evidence when available
- missing optional layers produce `0` section scores rather than source-table mutation
- requested Korea-local dates may resolve to the previous US trade date for scoring
- the output remains a recommendation artifact only and does not trigger trade execution

## Phase 3-4 Extension

Phase 3-4 adds operator-facing Top N ranking reports.

- `recommend.us_stock_rank_daily` is now consumed by a standalone reporting script
- Top N, grade-filtered, and symbol-detail outputs are separated from any trade path
- markdown/csv outputs are stored for review and audit, not for order execution
- missing ranking data prompts ranking calculation guidance or optional manual auto-calculate
- report generation failure must not propagate to the Korean trading pipeline

## Phase 3-5 Extension

Phase 3-5 adds explainability and validation metadata for the ranking snapshot.

- recommendation rows now carry Rule-based reason categories and tags through `score_detail_json`
- operator validation focuses on score sanity, exclusion reasons, data completeness, and anomaly warnings
- excluded rows are now reviewable as a first-class report output
- validation warnings are diagnostic artifacts only and must not be treated as trade instructions

## Phase 4-1 Extension

Phase 4-1 adds the first historical performance validation path for stored ranking snapshots.

- `recommend.us_stock_rank_daily` becomes the input for research-only backtest selection
- `research.us_stock_rank_backtest_result` stores symbol-level forward-return rows
- `research.us_stock_rank_backtest_summary` stores date-level strategy summaries
- entry uses the next US trading day after `trade_date` to avoid look-ahead bias
- backtest outputs remain separate from any live or paper trading path

## Phase 4-5 Extension

Phase 4-5 adds a recommendation-tracking Forward Test path for newly created ranking snapshots.

- `research.us_stock_rank_forward_test` stores strategy-level recommendation tracking rows
- `research.us_stock_rank_forward_test_summary` stores progress and completed-return summaries
- the snapshot is registered first and performance is filled only after future dates arrive
- Forward Test remains separate from paper trading, live trading, and broker order APIs

## Phase 5-1 Extension

Phase 5-1 adds the first US paper-trading data model.

- `paper.us_stock_paper_account` stores virtual account state
- `paper.us_stock_paper_order`, `paper.us_stock_paper_fill`, and `paper.us_stock_paper_position` separate paper order lifecycle from any real broker flow
- `paper.us_stock_paper_account_snapshot` stores daily virtual account snapshots
- Phase 5-1 still does not generate virtual buy/sell orders or simulate fills
- paper-trading structure remains separated from Korean real-trading code and broker APIs

## Phase 5-2 Extension

Phase 5-2 adds paper-only virtual order generation on top of the paper account structure.

- `paper.us_stock_paper_order` can now receive `CREATED` or `REJECTED` virtual orders
- BUY candidates come from US ranking snapshots and SELL candidates come from existing paper positions
- this phase still does not create fills, modify positions, or update cash balances
- order generation remains fully separated from broker APIs and Korean real-trading logic

## Phase 5-3 Extension

Phase 5-3 adds paper-only virtual fill simulation for created US paper orders.

- `paper.us_stock_paper_fill` now stores simulated fills for `CREATED` paper orders only
- BUY fills update paper cash and paper positions without touching any real-order path
- SELL fills realize paper PnL and can close paper positions entirely
- fill date uses the next US trading day after `order.trade_date` to avoid same-day look-ahead
- the simulation remains isolated to `paper.us_stock_*` tables and never calls broker APIs

## Phase 5-4 Extension

Phase 5-4 adds paper-only daily valuation snapshots and report artifacts.

- `paper.us_stock_paper_account_snapshot` now stores daily paper account valuation rows
- OPEN positions are marked to the snapshot-date close or an allowed previous close fallback
- paper account equity, unrealized PnL, and benchmark-relative performance are recalculated from paper data only
- markdown/csv/console reports remain review artifacts and do not trigger any trading path

## Phase 5-5 Extension

Phase 5-5 adds paper-only rebalancing planning and operating validation.

- rebalance policy is now explicitly configured through paper-only strategy, rebalance, and safety settings
- `plan_us_stock_paper_rebalance.py` shows SELL-first and BUY follow-up actions without placing any real order
- `validate_us_stock_paper_trading.py` checks account, position, order, fill, and snapshot consistency for repeatable operation
- paper trading reports now include operating-status sections such as last order/fill/snapshot dates and validation warning counts
- scheduler flags remain optional and disabled by default, and still do not connect to any broker or Korean trading path

## Phase 6-1 Extension

Phase 6-1 adds the first US live-trading safety policy document.

- `US_STOCK_LIVE_TRADING_POLICY.md` defines pre-live order policy, blocking rules, risk limits, approval flow, and kill-switch requirements
- live-trading ENV values are added as conservative drafts only and remain disabled by default
- this phase does not implement broker execution, real-balance reads, or live-order submission
- Phase 7 Micro Live remains blocked until the Phase 6 safety layer is fully designed and validated

## Phase 6-2 Extension

Phase 6-2 adds the reusable risk-policy data structure for later pre-trade validation.

- `config/us_stock_live_risk_policy.yaml` stores the reviewed Micro Live policy defaults with all live gates disabled
- `risk.us_stock_live_kill_switch` stores live kill-switch state separately from any broker integration
- `risk.us_stock_live_daily_risk_usage` stores daily policy-usage counters for future pre/post-trade checks
- `risk.us_stock_live_order_block_log` stores blocked live-order candidates and their reason codes
- `utils/us_live_risk_policy.py` loads YAML + ENV overrides and validates safe defaults without calling any broker API
- this phase still does not implement pre-trade blocking logic, broker execution, or real-balance reads

## Phase 6-3 Extension

Phase 6-3 adds the first US live pre-trade validation module.

- `utils/us_live_pre_trade_check.py` defines the live order-candidate structure, staged validation checks, and decision results
- `scripts/run_us_live_pre_trade_check.py` runs dry-run candidate checks from manual input or ranking-based candidates
- blocked or error candidates can be appended into `risk.us_stock_live_order_block_log` for audit review
- `ALLOW` in this phase still means policy-level eligibility only and does not create or send any real order
- account, position, and sector live-state validation remain approval-gated until a dedicated live account state model exists

## Phase 6-4 Extension

Phase 6-4 adds scoped Kill Switch management for the US live safety layer.

- `risk.us_stock_live_kill_switch` now supports `target_value` for symbol, sector, and account-level stops
- `risk.us_stock_live_kill_switch_event_log` stores activate/clear audit events
- manual and auto-evaluated Kill Switch paths remain safety controls only and do not submit any order
- Pre-Trade Check must force `BLOCK` whenever a matching kill switch is active
- this phase still does not call broker APIs, read real balances, or attach live execution

## Phase 6-5 Extension

Phase 6-5 adds approval-request state management between Pre-Trade Check and any future Micro Live review.

- `risk.us_stock_live_order_approval` stores `PENDING`, `APPROVED`, `REJECTED`, `EXPIRED`, `CANCELED`, and `ERROR` approval rows
- `risk.us_stock_live_order_approval_event_log` stores append-only approval lifecycle events
- `ALLOW` and `REQUIRE_APPROVAL` candidates can become approval requests, but still do not create any real order
- approval expiry is policy-driven and expired requests are invalid for later use
- Phase 7 must rerun Pre-Trade Check before considering any approved candidate for Micro Live review
- this phase still does not call broker APIs, read real balances, or attach live execution

## Phase 6-6 Extension

Phase 6-6 consolidates the full live-safety layer into an operator runbook and Phase 7 readiness checklist.

- `US_STOCK_LIVE_OPERATION_RUNBOOK.md` becomes the primary integrated operations document for Phase 6
- the runbook now lists related config, ENV, DB tables, utilities, scripts, and SQL review patterns
- Phase 6 completion and Phase 7 entry conditions are explicitly documented as operational checklists
- this phase remains documentation and operator-guidance only and still does not call broker APIs or create orders

## Phase 7-1 To 7-3 Extension

Phase 7 introduces the first tightly gated Micro Live order-review path.

- `live.us_stock_micro_order_request` stores Micro order-request rows
- `live.us_stock_micro_order_event_log` stores append-only Micro order lifecycle events
- execution modes are separated into `MOCK`, `SANDBOX`, and `LIVE`
- `LIVE` remains blocked by default and requires explicit safety gates
- manual approval remains mandatory
- no automatic unrestricted live execution is introduced

## Phase 7-4 Extension

Phase 7-4 adds broker-status and fill synchronization for already-created Micro orders.

- `utils/us_micro_order_sync.py` synchronizes broker order status into internal standardized states
- `live.us_stock_micro_order_fill` stores normalized fill rows
- `ORDER_FILLED` is still not enough to finalize internal position truth by itself
- sync remains separate from any automatic new-order or retry logic

## Phase 7-5 Extension

Phase 7-5 adds reconciliation between internal Micro order/fill state and broker-facing account/position state.

- `live.us_stock_micro_reconciliation_result` stores comparison rows
- `live.us_stock_micro_reconciliation_event_log` stores run-level reconciliation events
- expected internal positions are reconstructed from fills
- mismatches are recorded, not auto-corrected
- critical mismatches become kill-switch candidates

## Phase 7-6 Extension

Phase 7-6 adds the operator-facing daily Micro Live operations layer.

- ranking, block logs, approvals, micro orders, fills, reconciliation, kill switch, and risk usage are aggregated into one report
- action-required items are derived by severity
- system health is normalized into `HEALTHY`, `ATTENTION`, `DEGRADED`, and `CRITICAL`
- an incident runbook defines what to check, what to stop, and what not to do
- this phase remains review/reporting oriented and still does not authorize auto-trading

## Current Boundary After Phase 7

After Phase 7, the project has a validated safety and review path, but not a production auto-trading path.

- manual approval is still a control gate
- kill switch remains a hard stop mechanism
- status sync and reconciliation are operator-review tools
- operations reporting is informational and escalation-oriented
- Phase 8 must be treated as limited automation design, not unrestricted go-live

## Phase 4-2 Extension

Phase 4-2 adds operator-facing strategy performance reporting on top of the stored backtest outputs.

- backtest summary rows are aggregated into strategy/holding-day comparison views
- markdown/csv outputs are review artifacts only and remain outside any trade path
- Best Candidate wording is limited to review priority and must not be treated as a live-trading recommendation

## Phase 4-3 Extension

Phase 4-3 adds period and market-regime analysis for stored backtest summaries.

- `research.us_market_regime_daily` stores benchmark-based daily regime labels from SPY/QQQ inputs
- `research.us_stock_rank_backtest_regime_summary` stores aggregated regime/period performance views
- regime labels are derived from same-day benchmark trend and volatility state only
- monthly/quarterly/regime analysis remains a research artifact and must not trigger any trading path

## Phase 4-4 Extension

Phase 4-4 adds weight-candidate experimentation on top of stored Rule component scores.

- baseline Rule weights remain unchanged in the operational ranking path
- candidate weight sets are stored separately for research comparison
- experiment ranking outputs are written under `research` and never overwrite `recommend.us_stock_rank_daily`
- candidate promotion labels are review artifacts only and require forward testing before any operational consideration
