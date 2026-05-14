# Paper Trading Quality Gate Design

## Purpose

Phase 8-15 defines a quality-gate framework for the US Paper Trading operating stack before any future LIVE review discussion.

This phase does not approve LIVE trading.

The purpose is to determine whether the existing Paper Trading operating artifacts are reliable enough for human review of a future Go-Live discussion.

Important boundary:

- this phase is design only
- no broker API is called
- no real account balance or position is read
- no real BUY or SELL order is implemented
- no actual email or Slack delivery is implemented
- no automatic LIVE transition is allowed

## Core Principles

- Paper Trading performance alone is not enough for Go-Live review
- data quality, decision traceability, report consistency, scheduler stability, and LIVE safety must all be checked
- a gate `PASS` does not mean LIVE is automatically approved
- `go_live_review_allowed=true` means only that a manual review meeting may proceed
- `manual_review_required` must remain true

## Quality Gate Model

Recommended gate set:

1. `DATA_QUALITY_GATE`
2. `DECISION_LOGIC_GATE`
3. `REPORT_CONSISTENCY_GATE`
4. `SCHEDULER_STABILITY_GATE`
5. `NOTIFICATION_SAFETY_GATE`
6. `PERFORMANCE_VALIDATION_GATE`
7. `LIVE_SAFETY_GATE`
8. `MANUAL_REVIEW_GATE`

Status values:

- `PASS`
- `WARNING`
- `FAIL`
- `NOT_EVALUATED`
- `DATA_MISSING`

## 1. DATA_QUALITY_GATE

Purpose:

- verify that the daily operating stack is built on usable, date-aligned Paper-only source data

Validation targets:

- price data
- benchmark data
- ranking / score data
- financial data
- feature outputs
- Paper position reconstruction inputs
- trade-date alignment
- duplicate rows
- stale data
- missing-rate calculation

Pass criteria:

- required data sources exist
- `data_missing_rate <= 5%`
- ranking / score snapshot exists
- price and benchmark data exist for the evaluated trade date window
- no trade-date mismatch across critical artifacts

Warning criteria:

- `5% < data_missing_rate <= 20%`
- some optional financial or sector fields are missing
- stale but still reviewable data exists

Fail criteria:

- ranking / score data missing
- price data missing
- benchmark data missing
- `data_missing_rate > 20%`
- critical trade-date mismatch
- Paper position cannot be reconstructed safely

Data sources:

- `market.us_stock_daily_price`
- benchmark feature or market tables
- `recommend.us_stock_rank_daily`
- `feature.us_stock_feature_daily`
- `trade.us_paper_position_snapshot`
- `trade.us_integrated_daily_report`
- dashboard risk/data-quality sections

Operator action on failure:

- inspect upstream collection and feature jobs
- inspect date mismatch root cause
- suspend Go-Live review discussion until the source gap is fixed

LIVE transition impact:

- `FAIL` blocks Go-Live review

## 2. DECISION_LOGIC_GATE

Purpose:

- verify that BUY / SELL / conflict results are logically consistent and fully explainable

Validation targets:

- BUY decision reason completeness
- SELL exit reason completeness
- REVIEW_REQUIRED reason completeness
- conflict reason completeness
- disabled automation behavior
- position-vs-sell consistency
- quantity validity
- same-symbol BUY/SELL final-state conflict

Pass criteria:

- blocked BUY decisions always have block reasons
- SELL decisions always have exit reasons
- conflict-blocked decisions always have conflict reasons
- no negative remaining quantity
- disabled automation does not create unintended Paper orders

Warning criteria:

- repeated REVIEW_REQUIRED clusters
- repeated but still explainable conflict blocks
- low volume of isolated invalid rows pending cleanup

Fail criteria:

- `INVALID_DECISION_LOG`
- `BLOCK_REASON_MISSING`
- `EXIT_REASON_MISSING`
- `POSITION_QUANTITY_NEGATIVE`
- `LIVE_SAFETY_BYPASS`
- open-position absence with generated SELL decision
- same symbol treated as both final BUY and final SELL on the same effective run

Data sources:

- `trade.us_buy_decision_log`
- `trade.us_sell_decision_log`
- `trade.us_trade_conflict_log`
- `trade.us_paper_order`
- `trade.us_paper_sell_order`
- `trade.us_paper_position`

Operator action on failure:

- inspect decision-engine and logger outputs
- identify whether the issue is rule logic, logging loss, or state-reconstruction drift

LIVE transition impact:

- `FAIL` blocks Go-Live review

## 3. REPORT_CONSISTENCY_GATE

Purpose:

- verify that the same operational facts are reported consistently across BUY, SELL, integrated, dashboard, readiness, health, and notification artifacts

Validation targets:

- BUY automation report
- SELL automation report
- integrated trade report
- dashboard report
- notification payload
- readiness report
- health check result

Pass criteria:

- BUY candidate counts match
- final BUY allowed counts match
- SELL signal counts match
- REVIEW_REQUIRED counts match
- conflict blocked counts match
- Paper buy/sell order counts match
- `paper_trading_only=true` is preserved
- `live_orders_executed=false` is preserved
- `live_ready` does not conflict across readiness/dashboard/notification summaries

Warning criteria:

- one optional derived number differs but the primary source remains intact
- markdown-only artifact lags while JSON is correct

Fail criteria:

- a critical count differs between integrated report and dashboard
- notification payload contradicts readiness or dashboard live state
- Paper-only markers are missing or inconsistent

Data sources:

- `reports/lee_trader_us/buy_automation/`
- `output/us_stock_sell_automation/` or future sell report dir
- `reports/lee_trader_us/trade_orchestration/`
- `reports/lee_trader_us/dashboard/`
- `reports/lee_trader_us/dashboard/notifications/`
- `reports/lee_trader_us/buy_automation/readiness/`

Operator action on failure:

- identify the canonical source first
- fix report assembly or payload projection before any Go-Live discussion

LIVE transition impact:

- `FAIL` blocks Go-Live review

## 4. SCHEDULER_STABILITY_GATE

Purpose:

- verify that the Paper Trading operating pipeline runs reliably enough to support repeated review

Validation targets:

- scheduler success rate
- duplicate-run detection
- stale lock events
- lock release failures
- health-check pass rate
- report generation success rate
- dashboard generation success rate
- notification adapter dry-run success rate
- consecutive failures
- `pipeline_should_fail` frequency

Pass criteria:

- last `20` trading-day scheduler success rate `>= 95%`
- duplicate runs `= 0`
- health-check pass rate `>= 95%`
- dashboard and notification artifacts are produced consistently

Warning criteria:

- scheduler success rate between `90%` and `95%`
- stale lock seen at least once
- some report artifacts missing intermittently

Fail criteria:

- scheduler success rate `< 90%`
- duplicate runs are repeated
- health check fails repeatedly
- lock release failures repeat

Data sources:

- `trade.us_trade_scheduler_run_log`
- `trade.us_trade_scheduler_health_check`
- orchestration scheduler artifacts
- dashboard scheduler artifacts
- notification adapter logs

Operator action on failure:

- inspect lock policy and scheduler configuration
- separate infra/runtime failures from business-logic failures

LIVE transition impact:

- `FAIL` blocks Go-Live review

## 5. NOTIFICATION_SAFETY_GATE

Purpose:

- verify that notifications remain clearly Paper-only, non-sensitive, and non-executing

Validation targets:

- notification payload existence
- notification adapter dry-run success
- Paper Trading notice presence
- `live_orders_executed=false`
- `paper_trading_only=true`
- sensitive-field redaction
- `EMAIL_LIVE` / `SLACK_LIVE` blocking
- manual approval record existence
- CRITICAL severity conditions

Pass criteria:

- payload exists
- dry-run adapter succeeds for enabled channels
- Paper-only markers are preserved
- live channels remain blocked
- approval pending artifact exists when required

Warning criteria:

- markdown or optional channel output missing
- recipient/channel placeholders are empty in dry-run mode
- manual-approval pending artifact missing while no live send path exists

Fail criteria:

- `live_orders_executed=true`
- `paper_trading_only=false`
- sensitive data exposed
- live-delivery channel not blocked

Data sources:

- dashboard notification payload files
- notification adapter logs
- notification approval artifacts
- `trade.us_notification_event_log`
- `trade.us_notification_delivery_log`
- `trade.us_notification_approval_log`

Operator action on failure:

- treat as a safety issue, not a formatting issue
- stop any discussion of external delivery until fixed

LIVE transition impact:

- `FAIL` blocks Go-Live review

## 6. PERFORMANCE_VALIDATION_GATE

Purpose:

- verify that Paper Trading performance history is sufficiently long, interpretable, and benchmark-aware for review

Validation targets:

- Paper run duration
- minimum Paper order count
- minimum completed SELL count
- win rate
- average return
- median return
- max drawdown
- excess return vs `SPY`
- excess return vs `QQQ`
- realized vs unrealized mix
- sample sufficiency

Pass criteria:

- Paper period `>= 60` trading days
- Paper order count `>= 20`
- SELL order count `>= 5`
- scheduler success rate `>= 95%`
- excess return vs `SPY >= 0`
- max drawdown `<= 15%`
- data missing rate `<= 5%`

Warning criteria:

- sample still small
- realized SELL sample limited
- unrealized exposure dominates performance interpretation

Fail criteria:

- sustained underperformance vs benchmark
- drawdown exceeds policy
- missing-rate is too high to trust the performance numbers

Data sources:

- `trade.us_paper_performance_summary`
- `trade.us_paper_sell_order`
- `trade.us_buy_readiness_report`
- dashboard performance and benchmark sections

Operator action on failure:

- continue Paper Trading and collect a larger sample
- review whether performance weakness is strategy, execution policy, or data quality

LIVE transition impact:

- `FAIL` blocks Go-Live review
- `WARNING` keeps Go-Live review manual and cautious

## 7. LIVE_SAFETY_GATE

Purpose:

- verify that all Paper-only safety boundaries still hold and no LIVE pathway was accidentally opened

Validation targets:

- LIVE mode block behavior
- `LIVE_NOT_IMPLEMENTED` and `LIVE_DISABLED_IN_SCHEDULER` traces
- absence of real order API activity
- absence of broker API imports/calls in Paper path
- absence of real account balance queries
- absence of real account position queries
- absence of automatic ENV change
- manual approval requirement continuity

Pass criteria:

- LIVE mode attempts are blocked
- Paper artifacts still show `live_trading_enabled=false`
- no evidence of broker/live order path execution

Warning criteria:

- a reserved LIVE setting appears in config or artifacts but remains blocked correctly

Fail criteria:

- real order API call evidence
- LIVE mode without a block record
- `live_trading_enabled=true`
- `live_orders_executed=true`
- automatic ENV promotion toward LIVE

Data sources:

- scheduler logs
- dashboard payload
- notification payload
- code review results
- operations logs

Operator action on failure:

- treat as an immediate safety breach
- block any Go-Live discussion until resolved

LIVE transition impact:

- `FAIL` is an absolute blocker

## 8. MANUAL_REVIEW_GATE

Purpose:

- require a human operator to review longer-horizon trends and ambiguous risk items that automatic metrics cannot fully judge

Validation targets:

- 20-day and 60-day trend review
- BUY candidate quality
- SELL exit-rule reasonableness
- conflict guard over-blocking risk
- repeated REVIEW_REQUIRED cases
- data-missing root causes
- dashboard clarity
- notification clarity
- readiness-score trustworthiness
- real-order disabling state

Pass criteria:

- operator review completed
- key exceptions are documented
- no unresolved critical concern remains

Warning criteria:

- recurring but understood review items remain open

Fail criteria:

- review not performed
- major unresolved operator concern exists
- notification approval is confused with LIVE approval

Data sources:

- dashboard artifacts
- notification artifacts
- integrated reports
- readiness reports
- operations checklist

Operator action on failure:

- schedule manual review before continuing any Go-Live discussion

LIVE transition impact:

- `FAIL` blocks Go-Live review

## Quality Gate Result Structure

Recommended standard payload:

```json
{
  "evaluation_date": "2026-05-14",
  "lookback_days": 60,
  "overall_status": "WARNING",
  "go_live_review_allowed": false,
  "manual_review_required": true,
  "paper_trading_only": true,
  "live_orders_executed": false,
  "gates": {
    "DATA_QUALITY_GATE": {
      "status": "PASS",
      "score": 92,
      "warnings": [],
      "errors": []
    },
    "DECISION_LOGIC_GATE": {
      "status": "WARNING",
      "score": 80,
      "warnings": ["REVIEW_REQUIRED_REPEATED"],
      "errors": []
    },
    "LIVE_SAFETY_GATE": {
      "status": "PASS",
      "score": 100,
      "warnings": [],
      "errors": []
    }
  },
  "blocking_reasons": [
    "PAPER_DAYS_BELOW_MINIMUM",
    "SELL_SAMPLE_TOO_SMALL"
  ],
  "next_actions": [
    "Continue Paper Trading until 60 trading days",
    "Review repeated DATA_MISSING source"
  ]
}
```

Recommended interpretation:

- `overall_status=PASS` still does not enable LIVE
- `go_live_review_allowed=true` means only that a manual Go-Live review may be scheduled
- `manual_review_required` should stay `true` by policy

## Go-Live Pre-Check Checklist

```markdown
# US Paper Trading Go-Live Pre-Check

## Data
- [ ] Price data completeness verified
- [ ] Benchmark data completeness verified
- [ ] Ranking / score data available
- [ ] Data missing rate within threshold

## Decision Logic
- [ ] BUY block reasons are always recorded
- [ ] SELL exit reasons are always recorded
- [ ] Conflict reasons are always recorded
- [ ] REVIEW_REQUIRED reasons are clear

## Operations
- [ ] Scheduler success rate meets threshold
- [ ] Duplicate run protection verified
- [ ] Health check pass rate meets threshold
- [ ] Dashboard generated daily
- [ ] Notification dry-run generated daily

## Performance
- [ ] Minimum Paper days satisfied
- [ ] Minimum Paper order count satisfied
- [ ] Benchmark comparison reviewed
- [ ] Max drawdown reviewed
- [ ] Win rate reviewed

## Safety
- [ ] LIVE mode remains disabled
- [ ] No broker order API call exists
- [ ] No real account balance query exists
- [ ] No real account position query exists
- [ ] Manual approval required

## Final
- [ ] Go-Live review meeting completed
- [ ] Risk accepted manually
- [ ] Separate LIVE implementation Phase approved
```

Required statement:

- this checklist passing does not mean automatic LIVE transition
- LIVE requires a separate implementation phase and separate manual approval

## ENV Design

Recommended ENV:

```env
# US Paper Trading Quality Gate
US_QUALITY_GATE_ENABLED=0
US_QUALITY_GATE_LOOKBACK_DAYS=60
US_QUALITY_GATE_OUTPUT_DIR=reports/lee_trader_us/quality_gate

US_QUALITY_GATE_MIN_PAPER_DAYS=60
US_QUALITY_GATE_MIN_PAPER_ORDERS=20
US_QUALITY_GATE_MIN_SELL_ORDERS=5
US_QUALITY_GATE_MAX_DATA_MISSING_RATE_PCT=5
US_QUALITY_GATE_MIN_SCHEDULER_SUCCESS_RATE_PCT=95
US_QUALITY_GATE_MAX_DRAWDOWN_PCT=15
US_QUALITY_GATE_REQUIRE_POSITIVE_EXCESS_RETURN=1

US_QUALITY_GATE_FAIL_ON_LIVE_SAFETY_ERROR=1
US_QUALITY_GATE_REQUIRE_MANUAL_REVIEW=1
US_QUALITY_GATE_ALLOW_GO_LIVE_REVIEW=0
```

## Future Module Structure Proposal

Actual repository-aligned proposal:

```text
python/us/quality_gate/
  __init__.py
  config.py
  quality_data_loader.py
  data_quality_gate.py
  decision_logic_gate.py
  report_consistency_gate.py
  scheduler_stability_gate.py
  notification_safety_gate.py
  performance_validation_gate.py
  live_safety_gate.py
  manual_review_gate.py
  quality_gate_evaluator.py
  quality_gate_report.py
  run_us_quality_gate.py
```

Prompt-aligned alternative path:

```text
src/modules/lee_trader_us/quality_gate/
```

Suggested responsibilities:

- `config.py`: load quality-gate thresholds and output policy
- `quality_data_loader.py`: collect report, scheduler, dashboard, notification, and performance inputs
- `data_quality_gate.py`: evaluate source completeness, staleness, duplicates, and missing rate
- `decision_logic_gate.py`: validate BUY/SSELL/conflict logical consistency
- `report_consistency_gate.py`: compare counts and Paper-only markers across artifacts
- `scheduler_stability_gate.py`: evaluate success rate, lock stability, report generation rate
- `notification_safety_gate.py`: verify notification safety markers and live-channel blocking
- `performance_validation_gate.py`: evaluate sample size, performance, drawdown, benchmark excess return
- `live_safety_gate.py`: verify that no LIVE path or real-order signal leaked through
- `manual_review_gate.py`: capture operator-review obligations and unresolved concerns
- `quality_gate_evaluator.py`: combine gate results and compute `overall_status`
- `quality_gate_report.py`: render JSON/Markdown report artifacts
- `run_us_quality_gate.py`: CLI entrypoint for manual review-mode evaluation

## Report Output Design

Recommended output files:

- `reports/lee_trader_us/quality_gate/YYYY-MM-DD_quality_gate.json`
- `reports/lee_trader_us/quality_gate/YYYY-MM-DD_quality_gate.md`
- `reports/lee_trader_us/quality_gate/latest_quality_gate.json`
- `reports/lee_trader_us/quality_gate/latest_quality_gate.md`

Recommended Markdown sections:

1. Executive Summary
2. Gate Results
3. Blocking Reasons
4. Data Quality
5. Decision Logic
6. Report Consistency
7. Scheduler Stability
8. Notification Safety
9. Performance Validation
10. LIVE Safety
11. Manual Review Checklist
12. Next Actions

## Table Design

### `trade.us_quality_gate_report`

Purpose:

- store one assembled quality-gate report per evaluation date

Suggested fields:

- `quality_gate_report_id`
- `evaluation_date`
- `lookback_days`
- `overall_status`
- `go_live_review_allowed`
- `manual_review_required`
- `paper_trading_only`
- `live_orders_executed`
- `report_json`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(evaluation_date, lookback_days)`

Stored when:

- final report is assembled

Retention:

- at least 2 years

Related modules:

- `quality_gate_evaluator.py`
- `quality_gate_report.py`

### `trade.us_quality_gate_result`

Purpose:

- store one row per gate result for queryability

Suggested fields:

- `gate_result_id`
- `evaluation_date`
- `gate_name`
- `status`
- `score`
- `warnings JSONB`
- `errors JSONB`
- `detail_json`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(evaluation_date, gate_name)`

Stored when:

- each gate is evaluated

Retention:

- at least 2 years

Related modules:

- each `*_gate.py`

### `trade.us_go_live_precheck_log`

Purpose:

- store manual Go-Live pre-check review artifacts and meeting outcomes

Suggested fields:

- `precheck_log_id`
- `evaluation_date`
- `checklist_version`
- `review_status`
- `reviewed_by`
- `reviewed_at`
- `comment`
- `blocking_reasons JSONB`
- `next_actions JSONB`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(evaluation_date, checklist_version, reviewed_by)`

Stored when:

- a manual Go-Live review is recorded

Retention:

- at least 2 years

Related modules:

- future manual-review workflow only

## Scheduler Integration Design

Future scheduler ENV:

```env
US_TRADE_SCHEDULER_RUN_QUALITY_GATE=0
```

Recommended future order:

1. Trade Orchestration execution
2. Dashboard report generation
3. Notification adapter dry-run execution
4. Quality Gate evaluation
5. Scheduler final result recording

Rules:

- disabled by default
- quality-gate output is review-only
- `FAIL` does not rewrite trading decisions
- `FAIL` does not enable LIVE or change ENV

## Operator Actions By Overall Status

### PASS

- continue Paper Trading
- optionally schedule formal Go-Live review

### WARNING

- continue Paper Trading
- create follow-up actions for weak gates
- do not interpret warning as live approval

### FAIL

- block Go-Live review
- fix blocking issues first

### DATA_MISSING / NOT_EVALUATED

- treat as incomplete review input
- gather missing artifacts before reassessment

## Known Limitations

- this phase does not implement the evaluator
- no automated DB write path is added here
- quality scores are policy placeholders and may need tuning after longer Paper history
- manual review remains essential even if every automatic gate passes
