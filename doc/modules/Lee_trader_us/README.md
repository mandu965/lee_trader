# Lee_trader_us

> 문서 역할: `현재 기준 문서`
>
> 이 문서는 `Lee_trader_us` 문서 세트의 시작점이다. 먼저 읽고, 여기서 연결된 기준 문서로 이동하면 된다.

## Purpose

`Lee_trader_us` is the Project C module for a US-stock recommendation, paper-trading, and tightly gated Micro Live review track.

## Current Status

The codebase is now effectively through Phase 7.

Implemented layers:

- Phase 1: US universe, price collection, quality checks, baseline features
- Phase 2: financial data, feature engineering, relative strength, labels
- Phase 3: Rule-based ranking and operator reports
- Phase 4: backtest, regime analysis, weight experiments, forward test
- Phase 5: paper-trading account, orders, fills, snapshot, rebalance, validation
- Phase 6: live safety policy, risk policy, pre-trade check, kill switch, approval flow
- Phase 7: Micro Live mock/sandbox/live-gated order review, status sync, fill sync, reconciliation, operations report, incident runbook

## Safety Boundary

This module is still not unrestricted live auto-trading.

Current boundary:

- no automatic real-order rollout
- no automatic re-order
- no automatic position correction
- no automatic broker-state overwrite
- Live account/order access remains disabled by default
- Korean real-trading logic must stay untouched

## What Phase 7 Means

Phase 7 does not mean production auto-trading.

It means the project now has:

- a manual approval gate
- pre-trade blocking rules
- kill switch controls
- Micro order request lifecycle
- broker status / fill synchronization
- broker-vs-internal reconciliation
- integrated daily operations reporting
- incident-response runbook

## Primary Documents

- [CONTEXT.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/CONTEXT.md)
- [ARCHITECTURE.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/ARCHITECTURE.md)
- [FILE_INDEX.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/FILE_INDEX.md)
- [ENV.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/ENV.md)
- [OPERATIONS.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/OPERATIONS.md)
- [DB_SCHEMA.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/DB_SCHEMA.md)
- [BUY_AUTOMATION_DESIGN.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/BUY_AUTOMATION_DESIGN.md)
- [US_STOCK_RANKING_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_RANKING_V1.md)
- [US_STOCK_BACKTEST_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md)
- [US_STOCK_PAPER_TRADING.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_PAPER_TRADING.md)
- [US_STOCK_LIVE_TRADING_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_TRADING_POLICY.md)
- [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md)
- [US_STOCK_LIVE_OPERATION_RUNBOOK.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_OPERATION_RUNBOOK.md)
- [PHASE7_SUMMARY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/PHASE7_SUMMARY.md)

## Document Priority

Use the documents in this order.

| Document | Role | When To Read |
| --- | --- | --- |
| `README.md` | current entry document | start here |
| `PHASE7_SUMMARY.md` | current phase handoff summary | when preparing the next step |
| `FLOW.md` | current end-to-end runtime flow | when you need the big picture |
| `CONTEXT.md` | architecture boundary and phase extension history | when you need why the structure exists |
| `ARCHITECTURE.md` | current runtime integration map | when you need exact hook locations |
| `ENV.md` | current environment-variable reference | when touching config or safety flags |
| `OPERATIONS.md` | current command reference | when running scripts manually |
| `DB_SCHEMA.md` | current schema summary and proposed design tables | when reviewing data contracts |
| `BUY_AUTOMATION_DESIGN.md` | Phase 8-1 limited BUY design baseline | when preparing Phase 8 implementation |
| `FILE_INDEX.md` | code and script map | when locating implementation files |
| `US_STOCK_*` detailed docs | detailed reference documents | when working inside a specific phase/domain |

## Next Step

The next step is Phase 8-6: PAPER lifecycle 정교화와 수동 승격 워크플로우 보강입니다.

That phase must remain constrained:

- BUY only
- 1 day 1 order or less
- very small order notional
- limit orders only
- no automatic SELL expansion until separately validated

Current implementation status:

- Phase 8-1 design document is complete
- Phase 8-2 BUY automation skeleton is now implemented in SHADOW / PAPER review form
- Phase 8-3 report / validation / PAPER performance layer is now implemented
- Phase 8-4 scheduler integration is now implemented in SHADOW / PAPER only form
- Phase 8-5 readiness evaluation and promotion-policy layer is now implemented
- `LIVE` remains blocked and non-executable
