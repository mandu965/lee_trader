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

The US module still does not include:

- unrestricted real BUY execution
- broker order submission from BUY automation
- real account cash lookup for BUY automation
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
