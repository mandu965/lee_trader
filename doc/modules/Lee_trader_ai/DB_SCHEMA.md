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

## Project C Phase 2-3: US Financial Feature Layer

### Design Decision

Selected approach:

- create a separate feature table

Rejected approach:

- extending `feature.us_stock_feature_daily`

Reason:

- `feature.us_stock_feature_daily` is keyed by daily `feature_date`
- financial raw data is keyed by `fiscal_date + period_type`
- combining them in one table would mix daily and fiscal-period time axes
- a separate table avoids changes to the existing daily price feature pipeline
- future model datasets can still join daily and fiscal features using as-of logic

### New Table

### `feature.us_stock_financial_feature`

Purpose:

- stores derived US financial features created from raw financial statement / metric tables
- preserves fiscal-period granularity
- keeps growth, margin, stability, valuation, and temporary score-skeleton fields separate from daily price features

Key columns:

- `ticker`
- `market`
- `period_type`
- `fiscal_date`
- `source`
- `revenue`
- `gross_profit`
- `operating_income`
- `net_income`
- `ebitda`
- `eps`
- `total_assets`
- `total_liabilities`
- `total_equity`
- `operating_cash_flow`
- `free_cash_flow`
- `shares_outstanding`
- `market_cap`
- `revenue_growth_yoy`
- `revenue_growth_qoq`
- `net_income_growth_yoy`
- `net_income_growth_qoq`
- `eps_growth_yoy`
- `eps_growth_qoq`
- `free_cash_flow_growth_yoy`
- `free_cash_flow_growth_qoq`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `ebitda_margin`
- `roe`
- `roa`
- `debt_to_equity`
- `debt_ratio`
- `equity_ratio`
- `current_ratio`
- `free_cash_flow_margin`
- `per`
- `pbr`
- `psr`
- `ev_ebitda`
- `dividend_yield`
- `financial_quality_score`
- `financial_growth_score`
- `financial_value_score`
- `raw_collected_at`
- `feature_created_at`
- `created_at`
- `updated_at`

### Upsert Key

- `(ticker, period_type, fiscal_date, source)`

### Index Strategy

- `(ticker, fiscal_date DESC)`
- `(ticker, period_type, fiscal_date DESC)`
- `(source, feature_created_at DESC)`

### Nullable Policy

Nullable features are required because:

- raw `yfinance` statement and metric coverage differs by ticker and fiscal period
- growth fields need prior comparable periods and are null when those references do not exist
- margin and leverage fields are null when denominators are missing or zero
- valuation fields from `info` are snapshot-style and may be absent
- temporary score-skeleton fields are nullable when too many underlying inputs are missing

### Feature Mapping Notes

Feature builder source path:

- `raw.us_stock_financial_statement`
- `raw.us_stock_financial_metric`

Derived feature families:

- growth:
  - `revenue_growth_yoy`
  - `revenue_growth_qoq`
  - `net_income_growth_yoy`
  - `net_income_growth_qoq`
  - `eps_growth_yoy`
  - `eps_growth_qoq`
  - `free_cash_flow_growth_yoy`
  - `free_cash_flow_growth_qoq`
- margin:
  - `gross_margin`
  - `operating_margin`
  - `net_margin`
  - `ebitda_margin`
  - `free_cash_flow_margin`
- stability:
  - `debt_ratio`
  - `equity_ratio`
  - `debt_to_equity`
  - `current_ratio`
- valuation:
  - `per`
  - `pbr`
  - `psr`
  - `ev_ebitda`
  - `dividend_yield`
  - `market_cap`

### Source Tracking

- `source` preserves the collector source such as `yfinance`
- `raw_collected_at` preserves the source-row collection timestamp
- `feature_created_at` tracks when the derived feature row was built

## Project C Phase 2-4: Relative Strength Feature Layer

### Design Decision

Selected approach:

- create a separate daily relative strength table

Rejected approach:

- extending `feature.us_stock_feature_daily`

Reason:

- the existing Phase 1 daily feature table and builder are already stable and in use
- `ret_5d`, `ret_20d`, and `ret_60d` already exist there, so expanding that schema would require changing current build contracts
- a separate table allows standalone validation of benchmark coverage, rank percentiles, and null handling
- later modeling can join the daily price feature table and the relative strength table by `ticker + trade_date`

### New Table

### `feature.us_stock_relative_strength_daily`

Purpose:

- stores daily stock returns, benchmark returns, and stock-minus-benchmark relative strength values
- keeps SPY / QQQ benchmark logic independent from the existing Phase 1 daily feature builder

Key columns:

- `ticker`
- `market`
- `trade_date`
- `price_column_used`
- `ret_5d`
- `ret_20d`
- `ret_60d`
- `ret_120d`
- `ret_252d`
- `spy_ret_5d`
- `spy_ret_20d`
- `spy_ret_60d`
- `spy_ret_120d`
- `spy_ret_252d`
- `qqq_ret_5d`
- `qqq_ret_20d`
- `qqq_ret_60d`
- `qqq_ret_120d`
- `qqq_ret_252d`
- `rs_spy_5d`
- `rs_spy_20d`
- `rs_spy_60d`
- `rs_spy_120d`
- `rs_spy_252d`
- `rs_qqq_5d`
- `rs_qqq_20d`
- `rs_qqq_60d`
- `rs_qqq_120d`
- `rs_qqq_252d`
- `rs_spy_20d_rank_pct`
- `rs_spy_60d_rank_pct`
- `rs_qqq_20d_rank_pct`
- `rs_qqq_60d_rank_pct`
- `source`
- `created_at`
- `updated_at`

### Upsert Key

- `(ticker, trade_date, source)`

### Index Strategy

- `(ticker, trade_date DESC)`
- `(trade_date DESC)`
- `(trade_date DESC, rs_spy_60d DESC)`
- `(trade_date DESC, rs_qqq_60d DESC)`

### Nullable Policy

Nullable relative strength fields are required because:

- early listing history may be shorter than the requested window
- benchmark rows may be missing for some dates
- `adj_close_price` or `close_price` may be absent for a given ticker/date
- percentile rank fields are null when the base relative strength field is null

### Price Mapping Notes

- source price table:
  - `market.us_stock_daily_price`
- price selection policy:
  - prefer `adj_close_price`
  - fallback to `close_price`
- benchmark scope:
  - `SPY`
  - `QQQ`

## Project C Phase 2-5: Label Layer

### `label.us_stock_label_daily`

Purpose:

- stores future-return labels for later US model experiments
- keeps training targets separate from feature tables and production ranking flows

Key columns:

- `ticker`
- `market`
- `trade_date`
- `price_column_used`
- `future_ret_5d`
- `future_ret_20d`
- `future_ret_60d`
- `future_ret_20d_rank_pct`
- `future_ret_60d_rank_pct`
- `label_positive_20d`
- `label_positive_60d`
- `label_top20_20d`
- `label_top20_60d`
- `source`
- `label_created_at`
- `created_at`
- `updated_at`

### Upsert Key

- `(ticker, trade_date, source)`

### Index Strategy

- `(ticker, trade_date DESC)`
- `(trade_date DESC)`
- `(trade_date DESC, label_top20_20d)`
- `(trade_date DESC, label_top20_60d)`

### Nullable Policy

Nullable labels are required because:

- recent rows do not have enough future trading days
- current price may be missing or zero
- future price may be missing
- same-date universe size may be smaller than the configured top20 minimum
- excluded benchmarks such as `SPY` and `QQQ` do not receive top20 labels

### Forward Return Rules

- `future_ret_Nd = future_close_Nd / close_today - 1`
- trading-day shift is used, not calendar-day subtraction
- `adj_close_price` is preferred, `close_price` is fallback
- `NaN` / `inf` values are not stored

### Top20 Label Rules

- same-date universe percentile ranks are computed from `future_ret_20d` and `future_ret_60d`
- default cutoff is top 20%
- default benchmark exclusions:
  - `SPY`
  - `QQQ`

### Leakage Notes

- features must use `trade_date` or earlier information only
- labels use future price information only
- financial features require reported-date-aware as-of join and are not auto-joined in this phase

## Project C Phase 3-1: Recommendation Universe Master

### `meta.us_stock_universe`

Purpose:

- stores the recommendation candidate master for the future US ranking engine
- keeps recommendation filters separate from `market.us_stock_universe`, which is used for collection membership

Key columns:

- `symbol`
- `company_name`
- `market`
- `sector`
- `industry`
- `universe_group`
- `is_active`
- `is_etf`
- `is_leveraged`
- `is_inverse`
- `source`
- `market_cap`
- `avg_volume`
- `currency`
- `country`
- `exchange`
- `first_included_date`
- `last_checked_date`
- `exclude_reason`
- `feature_quality_score`
- `created_at`
- `updated_at`

### Upsert Key

- `(symbol)`

### Filter Notes

The recommendation universe is intended to sit before ranking:

- `meta.us_stock_universe`
- `feature.us_stock_feature_daily`
- `feature.us_stock_financial_feature`
- `feature.us_stock_relative_strength_daily`
- future `recommend.us_stock_rank_daily`

Active-universe filtering uses:

- active flag
- ETF inclusion policy
- leveraged / inverse exclusion flags
- recent price availability
- average volume threshold
- market cap threshold
- derived or stored feature-quality score

### Universe Group Notes

`universe_group` stores merged membership tags such as:

- `SP500`
- `NASDAQ100`
- `ETF`
- future manual groups such as `LARGE_CAP` or `MANUAL`
