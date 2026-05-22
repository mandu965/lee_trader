# Lee_trader_us

> 문서 역할: `현재 기준 문서`
>
> 이 문서는 `Lee_trader_us` 문서 트리의 시작점입니다. 먼저 여기서 현재 상태를 확인하고, 목적에 맞는 상세 문서로 이동하면 됩니다.

## Purpose

`Lee_trader_us` is the Project C module for a US-stock recommendation and paper-trading track.

As of 2026-05-22, the operating principle is:

- US remains `paper-only`
- real-capital rollout is `not` an active goal in the current operating plan
- live / Micro Live documents are retained as reference artifacts only

## Current Status

The codebase contains documents and code up through Phase 8-6 design / validation work.
However, the **current operating mode** is narrower than the full design history.

Implemented layers:

- Phase 1: US universe, price collection, quality checks, baseline features
- Phase 2: financial data, feature engineering, relative strength, labels
- Phase 3: Rule-based ranking and operator reports
- Phase 4: backtest, regime analysis, weight experiments, forward test
- Phase 5: paper-trading account, orders, fills, snapshot, rebalance, validation
- Phase 6: live safety policy, risk policy, pre-trade check, kill switch, approval flow
- Phase 7: Micro Live mock/sandbox/live-gated review artifacts, status sync, fill sync, reconciliation, operations report, incident runbook
- Phase 8-1 to 8-5: limited BUY automation design, SHADOW/PAPER evaluation, reporting, scheduler integration, readiness evaluation
- Phase 8-6: limited SELL / Exit design

Current practical focus:

- ranking quality
- backtest reproducibility
- paper-trading lifecycle
- buy/sell policy validation
- scheduler / report stability

## Safety Boundary

This module is still not unrestricted live auto-trading.

Current boundary:

- paper trading is the only active operating mode
- no automatic real-order rollout
- no automatic re-order
- no automatic position correction
- no automatic broker-state overwrite
- no LIVE SELL activation
- Live account/order access remains disabled by default
- Korean real-trading logic must stay untouched

## Primary Documents

- [CURRENT_DOC_PRIORITY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/CURRENT_DOC_PRIORITY.md)
- [CONTEXT.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/CONTEXT.md)
- [ARCHITECTURE.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/ARCHITECTURE.md)
- [FILE_INDEX.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/FILE_INDEX.md)
- [ENV.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/ENV.md)
- [OPERATIONS.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/OPERATIONS.md)
- [DB_SCHEMA.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/DB_SCHEMA.md)
- [BUY_AUTOMATION_DESIGN.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/BUY_AUTOMATION_DESIGN.md)
- [SELL_AUTOMATION_DESIGN.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/SELL_AUTOMATION_DESIGN.md)
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
| `CURRENT_DOC_PRIORITY.md` | active-vs-historical reading guide | before diving into detailed docs |
| `PHASE7_SUMMARY.md` | historical phase handoff summary | when reviewing Micro Live design history |
| `FLOW.md` | current end-to-end runtime flow | when you need the big picture |
| `CONTEXT.md` | architecture boundary and phase extension history | when you need why the structure exists |
| `ARCHITECTURE.md` | current runtime integration map | when you need exact hook locations |
| `ENV.md` | current environment-variable reference | when touching config or safety flags |
| `OPERATIONS.md` | current command reference | when running scripts manually |
| `DB_SCHEMA.md` | current schema summary and proposed design tables | when reviewing data contracts |
| `BUY_AUTOMATION_DESIGN.md` | Phase 8 BUY design baseline | when preparing entry-policy implementation |
| `SELL_AUTOMATION_DESIGN.md` | Phase 8-6 limited SELL / Exit design baseline | when preparing exit-policy implementation |
| `FILE_INDEX.md` | code and script map | when locating implementation files |
| `US_STOCK_*` detailed docs | detailed reference documents | when working inside a specific phase/domain |

## Current Next Step

The next step is not live rollout.

The current next-step focus is:

- paper 30-trading-day accumulation
- buy/sell lifecycle validation
- current-policy gross / net verification
- paper-first SELL / HOLD / REVIEW_REQUIRED refinement

Any next implementation must remain constrained:

- no real SELL order submission
- Paper position first
- reviewable exit logic
- no LIVE SELL activation
- no broker path addition

## Current Implementation Status

- Phase 8-1 BUY design document is complete
- Phase 8-2 BUY automation skeleton is implemented in SHADOW / PAPER review form
- Phase 8-3 BUY report / validation / PAPER performance layer is implemented
- Phase 8-4 scheduler integration is implemented in SHADOW / PAPER only form
- Phase 8-5 readiness evaluation and promotion-policy layer is implemented
- Phase 8-6 SELL / Exit policy design is documented
- `LIVE` remains blocked and non-executable
- current operating plan does not promote US into live trading
