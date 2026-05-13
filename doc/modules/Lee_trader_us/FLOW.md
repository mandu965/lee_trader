# Lee_trader_us Flow

> 문서 역할: `현재 기준 문서`
>
> 현재 시점의 전체 흐름을 설명한다. 상세 정책이나 명령어보다 먼저 큰 구조를 볼 때 사용한다.

## Current End-to-End Flow

Current Project C US-stock flow is:

`Universe -> Price -> Validation -> Features -> Ranking -> Backtest / Forward Test -> Paper Trading -> Live Safety -> Micro Live -> Status Sync -> Reconciliation -> Operations Report`

## Current Phase Boundary

The project is now effectively through Phase 7.

Implemented layers:

1. Universe management
2. OHLCV collection
3. data quality validation
4. baseline and financial/RS feature generation
5. Rule-based ranking
6. backtest and forward-test reporting
7. paper-trading lifecycle
8. live safety policy and approval flow
9. Micro order request / send review path
10. status and fill synchronization
11. reconciliation
12. daily operations reporting

## What Phase 7 Means In The Flow

Phase 7 is not unrestricted live trading.

Phase 7 adds:

1. approved Micro order review
2. mock / sandbox / gated live path separation
3. broker status sync
4. fill sync
5. internal-vs-broker reconciliation
6. operations report and incident-response workflow

## What Is Still Outside The Flow

The following are still outside the allowed runtime path:

1. unrestricted production live auto-trading
2. automatic re-order
3. automatic position correction
4. automatic broker-state overwrite
5. unrestricted sell automation

## Recommended Reading Path

1. [PHASE7_SUMMARY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/PHASE7_SUMMARY.md)
2. [CONTEXT.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/CONTEXT.md)
3. [OPERATIONS.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/OPERATIONS.md)
4. [US_STOCK_LIVE_TRADING_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_TRADING_POLICY.md)
5. [US_STOCK_LIVE_RISK_POLICY.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md)
6. [US_STOCK_LIVE_OPERATION_RUNBOOK.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_LIVE_OPERATION_RUNBOOK.md)
