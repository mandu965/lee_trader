# Lee_trader_us ENV

## Purpose

This document describes the environment variables reserved for Project C US stock preparation.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_STOCK_ENABLED` | `false` | Master switch for the US stock pipeline | US-only pipeline | `run_us_daily_pipeline.py` skips unless `--force` is provided |
| `US_STOCK_UNIVERSE` | `NASDAQ100` | Universe tag to manage | universe | Phase 1 default universe |
| `US_STOCK_DATA_SOURCE` | `yfinance` | US daily price data source | price collection | Phase 1 supports only `yfinance` |
| `US_STOCK_PRICE_BACKFILL_YEARS` | `5` | Default initial history range | backfill | Overridden by explicit start date |
| `US_STOCK_PRICE_START_DATE` | blank | Explicit backfill start date | backfill | Optional |
| `US_STOCK_PRICE_END_DATE` | blank | Explicit collection end date | backfill / replay | Optional |
| `US_STOCK_STALE_DAYS_LIMIT` | `3` | Data freshness threshold | quality validation | Calendar-day based in Phase 1 |
| `US_STOCK_BATCH_SIZE` | `20` | Batch size for collection | price collection | Used by `download_us_prices.py` |
| `US_STOCK_REQUEST_SLEEP_SEC` | `1` | Sleep between batches | price collection | Used by `download_us_prices.py` |
| `US_PAPER_TRADING_ENABLED` | `false` | Reserved for future paper trading | future phases | Not used in this phase |
| `US_LIVE_TRADING_ENABLED` | `false` | Reserved for future live trading | future phases | Must not trigger any order code |
| `US_LIVE_BROKER` | `none` | Reserved broker selector | future phases | Must remain unused in this phase |

## Phase 1-6 Rules

- `run_us_daily_pipeline.py` is independent from Korean schedulers.
- `US_STOCK_ENABLED=false` means the pipeline is skipped by default.
- `--force` is only a manual override for standalone execution.
- `US_PAPER_TRADING_ENABLED` is ignored in this phase.
- `US_LIVE_TRADING_ENABLED` is ignored in this phase and must not trigger any order function.
