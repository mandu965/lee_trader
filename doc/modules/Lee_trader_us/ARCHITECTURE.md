# Lee_trader_us Architecture

> 문서 역할: `현재 기준 문서`
>
> 이 문서는 `Lee_trader_us`의 현재 실행 경계와 단계별 연결 위치를 설명합니다. Phase 8-4 기준으로 SHADOW/PAPER BUY 자동화 스케줄러 연결까지 반영합니다.

## Purpose

This document explains the current architecture boundary of the US module and the runtime order that operators and future implementers should assume.

## Core Boundary

The US module now includes:

- data collection
- feature generation
- ranking
- backtest / forward test / paper research flows
- live safety and Micro Live validation flows
- limited BUY automation in SHADOW / PAPER review mode
- limited SELL / Exit automation in SHADOW / PAPER review mode
- BUY / SELL orchestration and integrated daily review reporting
- Paper Trading dashboard and monitoring design

The US module still does not include:

- unrestricted real BUY execution
- unrestricted real SELL execution
- broker order submission from BUY automation
- broker order submission from SELL automation
- real account cash lookup for BUY automation
- real account cash lookup for SELL automation
- automatic SELL automation
- automatic retry or correction orders

## Current Execution Layers

### Research And Data Layer

1. universe load
2. daily price collection
3. financial / relative-strength / label support
4. feature generation
5. ranking and reporting
6. backtest / forward test / paper validation

### Live Safety Layer

1. pre-trade check
2. kill switch
3. approval flow
4. Micro Live order review
5. status sync
6. reconciliation
7. operations report

### Limited BUY Automation Layer

1. Phase 8-2:
   - candidate loading
   - fail-safe risk guard
   - SHADOW / PAPER decision pipeline
2. Phase 8-3:
   - daily report
   - validation summary
   - PAPER performance tracking
3. Phase 8-4:
   - scheduler wrapper
   - daily pipeline hook
   - failure isolation
4. Phase 8-5:
   - cumulative PAPER performance evaluation
   - LIVE readiness scoring
   - promotion-policy review layer

### Limited SELL And Trade Orchestration Layer

1. Phase 8-7:
   - Paper-position reconstruction
   - SELL / PARTIAL_SELL / HOLD / REVIEW_REQUIRED decision pipeline
   - Paper SELL artifact logging only
2. Phase 8-8:
   - SELL-first orchestration
   - BUY / SELL conflict guard
   - cooldown and duplicate-BUY protection
   - integrated daily trade report
3. Phase 8-9:
   - orchestration scheduler guard
   - duplicate-run lock
   - post-run health check
   - operations checklist
   - daily pipeline integration

### Dashboard And Monitoring Layer

1. Phase 8-10:
   - daily dashboard design
   - Paper portfolio monitoring design
   - BUY / SELL / conflict monitor design
   - benchmark comparison design
   - scheduler / health visibility design
   - LIVE-readiness review surface design
2. Phase 8-11:
   - file-based JSON dashboard report
   - file-based Markdown dashboard report
   - `latest_dashboard.json` / `latest_dashboard.md` update
   - DB-first, file-fallback dashboard data loading
3. Phase 8-12:
   - optional scheduler hook after orchestration
   - dashboard health adapter
   - notification text/json payload generation
   - payload file persistence only, no actual delivery
4. Phase 8-13:
   - notification adapter design
   - file / console / email-dry-run / slack-dry-run channel policy
   - manual-approval delivery policy
   - severity and redaction policy
   - no real external delivery implementation

## Phase 8-4 Integration Point

Primary scheduler integration files:

- `python/us/buy_automation/scheduler_job.py`
- `python/us/run_us_buy_scheduler_job.py`
- `scripts/run_us_buy_scheduler_job.py`
- `python/us/run_us_daily_pipeline.py`

Current integration rule:

- BUY automation scheduler runs only after upstream US data/feature stages complete
- it is treated as a review stage, not an execution stage
- if ranking data is missing, the job records `SOURCE_DATA_MISSING` or `NO_CANDIDATE`
- by default, BUY automation scheduler failure does not fail the full pipeline

## Runtime Order

Recommended current order is:

```text
1. Load universe
2. Collect prices
3. Validate price quality
4. Build features
5. Build ranking / score snapshot
6. Run BUY automation scheduler job
7. Generate BUY automation report
8. Review daily outputs
```

Important note:

`python/us/run_us_daily_pipeline.py` is still a lighter US pipeline and does not fully own ranking generation yet. Phase 8-4 adds the BUY scheduler hook there as a post-feature stage, but the scheduler itself remains fail-safe when ranking data is not available.

## BUY Scheduler Contract

The scheduler wrapper is responsible for:

- reading scheduler ENV
- refusing `LIVE`
- running BUY automation in `SHADOW` or `PAPER` only
- generating JSON / Markdown report
- returning a structured summary for the parent pipeline
- isolating errors unless explicit fail-fast is enabled

Expected result structure:

```python
{
    "job": "us_buy_automation",
    "enabled": True,
    "mode": "SHADOW",
    "automation_executed": True,
    "report_executed": True,
    "success": True,
    "error": None,
    "pipeline_should_fail": False,
    "summary": {
        "loaded_candidates": 5,
        "allowed_candidates": 0,
        "blocked_candidates": 5,
        "paper_orders": 0,
    },
}
```

## LIVE Defense

Phase 8-4 keeps explicit scheduler-level `LIVE` blocking.

If `US_BUY_AUTOMATION_MODE=LIVE`:

- no BUY automation execution occurs
- no broker path exists
- no real order path exists
- report generation may still write a disabled-state artifact
- the scheduler returns `LIVE_DISABLED_IN_SCHEDULER`

Even if `US_BUY_SCHEDULER_ALLOW_LIVE=1`, this phase still does not permit real-order execution.

## Failure Isolation

Default rule:

- `US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR=0`

Meaning:

- BUY automation errors are logged
- BUY report errors are logged
- the parent US pipeline continues unless explicitly configured otherwise

Optional stricter rule:

- `US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR=1`

Meaning:

- scheduler exceptions are raised to the caller
- parent pipeline may fail intentionally

## Output Locations

Current outputs:

- raw BUY automation logs:
  - `output/us_stock_buy_automation/`
- BUY report JSON / Markdown:
  - `reports/lee_trader_us/buy_automation/` by default
  - may follow ENV override

## What Operators Must Review Daily

1. whether scheduler execution actually ran
2. whether mode stayed `SHADOW` or `PAPER`
3. whether `LIVE_DISABLED_IN_SCHEDULER` appeared
4. block-summary distribution
5. fail-safe-triggered status
6. invalid decision log warnings
7. PAPER order count and PAPER performance section
8. repeated `SOURCE_DATA_MISSING` or `NO_CANDIDATE` events

## Safety Statement

Phase 8-4 is not a real-trading scheduler integration.

It is a daily automated review integration for:

- SHADOW candidate evaluation
- PAPER virtual order tracking
- report generation
- operator verification

Real BUY execution remains prohibited.

## Phase 8-8 Orchestration Boundary

Trade orchestration is still a Paper-only operating layer.

- SELL runs before BUY
- SELL or REVIEW_REQUIRED state can block same-day BUY
- existing Paper position can block repeat BUY
- cooldown after full exit can block re-entry
- no real-order path is added
- scheduler conflict between BUY-only and orchestration mode must be treated as configuration error

## Phase 8-9 Scheduler Integration Point

Actual current integration point in this repository:

- `python/us/run_us_daily_pipeline.py`

Current order after upstream data/feature work:

1. optional trade orchestration scheduler job
2. optional BUY-only scheduler job

Protection rules:

- orchestration must run after ranking/score availability
- orchestration scheduler can disable BUY-only scheduler through ENV
- duplicate run for the same `trade_date` is blocked by file lock
- `LIVE` mode remains blocked in scheduler guard

## Phase 8-10 Dashboard Read Boundary

The dashboard layer is read-only and sits after orchestration artifacts exist.

Intended future order:

1. upstream data / ranking stages
2. orchestration scheduler job
3. integrated report
4. dashboard assembly from persisted artifacts
5. dashboard-aware health check
6. notification payload generation

Dashboard scope:

- consumes Paper-only logs, snapshots, and summaries
- does not place orders
- does not call broker APIs
- does not read real account balances or positions

Current implementation location:

- `python/us/dashboard/`

Current scheduler connection point:

- `python/us/trade_orchestration/scheduler_job.py`

## Phase 8-13 Notification Adapter Boundary

The notification-adapter layer sits after dashboard payload generation and after dashboard-aware health checks.

Recommended order:

1. orchestration scheduler job
2. integrated report
3. dashboard report
4. dashboard-aware health check
5. notification payload generation
6. notification adapter routing
7. scheduler final result persistence

Adapter scope:

- consumes already-generated dashboard payloads
- may render channel-specific dry-run messages
- may write local file artifacts
- must not alter orchestration or dashboard decisions
- must not submit any broker order
- must not call SMTP, Slack webhook, or other external APIs in this phase
