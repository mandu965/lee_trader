# Lee_trader_ai DB Schema

## Project C Phase 2-2: US Financial Raw Data

This phase adds schema design and collector write targets for future US stock financial data ingestion.

### Purpose

- store raw US financial statement values separately from price/features
- preserve nullable source values from `yfinance`
- support annual and quarterly collection
- keep Korean AI / RULE operational tables untouched

## New Tables

### `raw.us_stock_financial_statement`

Purpose:

- stores statement-style raw values such as revenue, profit, balance sheet, and cashflow figures

Key columns:

- `ticker`
- `market`
- `period_type`
- `fiscal_date`
- `reported_date`
- `currency`
- `revenue`
- `gross_profit`
- `operating_income`
- `net_income`
- `ebitda`
- `total_assets`
- `total_liabilities`
- `total_equity`
- `operating_cash_flow`
- `investing_cash_flow`
- `financing_cash_flow`
- `free_cash_flow`
- `source`
- `source_updated_at`
- `collected_at`
- `created_at`
- `updated_at`

### `raw.us_stock_financial_metric`

Purpose:

- stores ratio / metric style values and market-linked financial indicators

Key columns:

- `ticker`
- `market`
- `period_type`
- `fiscal_date`
- `reported_date`
- `currency`
- `eps`
- `roe`
- `roa`
- `shares_outstanding`
- `market_cap`
- `per`
- `pbr`
- `psr`
- `ev_ebitda`
- `debt_to_equity`
- `current_ratio`
- `dividend_yield`
- `source`
- `source_updated_at`
- `collected_at`
- `created_at`
- `updated_at`

## Upsert Key

Both tables use:

- `(ticker, period_type, fiscal_date, source)`

This supports idempotent re-collection from the same source.

## Index Strategy

- `(ticker, fiscal_date DESC)`
- `(ticker, period_type, fiscal_date DESC)`
- `(source, collected_at DESC)`

## Nullable Policy

Nullable metrics are required because:

- `yfinance` field coverage differs by ticker
- some statement sections are missing for certain periods
- market-based metrics from `info` are not guaranteed to exist
- `reported_date` and `source_updated_at` are not reliably exposed by `yfinance`

## yfinance Mapping Notes

Statement table receives values from:

- `financials`
- `quarterly_financials`
- `balance_sheet`
- `quarterly_balance_sheet`
- `cashflow`
- `quarterly_cashflow`

Metric table receives values from:

- financial statement frames for fields like `eps`
- `info` for fields like `market_cap`, `per`, `pbr`, `psr`, `ev_ebitda`, `dividend_yield`

Note:

- market snapshot values from `info` may repeat across multiple fiscal periods because `yfinance` exposes them as latest ticker-level metadata rather than period-specific history.
