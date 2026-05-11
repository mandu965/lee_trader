# Lee_trader_us

## Purpose

`Lee_trader_us` is the Project C module for a future US stock recommendation system.

## Phase 1 Scope

Phase 1 implements only the US data foundation:

- NASDAQ100 universe loading
- US daily OHLCV collection
- data quality validation
- baseline price feature generation
- standalone US daily pipeline orchestration

## Not Live Trading

This module is not real trading.

- no broker order path
- no Alpaca / IBKR order API
- no Korean KIS order integration

## Not AI Training

This phase is not an AI/ML training phase.

- no model training
- no model inference
- no final ranking logic

## Not Paper Trading

Paper trading is out of scope in this phase.

## Current Goal

The current goal is to prepare a stable NASDAQ100-based US data pipeline that remains fully separated from Korean auto-trading.
