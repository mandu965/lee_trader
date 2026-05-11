# Lee_trader_us Operations

## Manual Commands

```powershell
python -m python.us.load_us_universe --universe NASDAQ100
python -m python.us.download_us_prices --universe NASDAQ100 --backfill
python -m python.us.download_us_prices --universe NASDAQ100 --incremental
python -m python.us.validate_us_price_data --universe NASDAQ100 --as-of-date 2026-05-11
python -m python.us.build_us_features --universe NASDAQ100
python -m python.us.build_us_features --universe NASDAQ100 --ticker AAPL --verbose
python -m python.us.build_us_features --universe NASDAQ100 --limit 5
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --backfill
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --incremental
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --skip-prices
python -m python.us.run_us_daily_pipeline --universe NASDAQ100 --force --skip-quality
```

## Backfill Method

- use `US_STOCK_PRICE_BACKFILL_YEARS` as the default historical range
- allow explicit override with `US_STOCK_PRICE_START_DATE` and `US_STOCK_PRICE_END_DATE`
- run backfill as a US-only standalone operation

## Incremental Collection Method

- run the US pipeline separately from Korean scheduling
- collect only the latest missing US trading dates
- write collection status to `market.us_stock_data_collect_log`

## Data Quality Checks

- verify missing rows by universe member
- verify stale latest trade date
- verify invalid OHLC relationships
- verify invalid volume values
- remember stale checks are calendar-day based in Phase 1

## Feature Build Notes

- `build_us_features.py` creates only baseline price features
- it is not AI/ML model training
- it does not create final ranking scores
- it does not connect to Korean auto-trading

## Pipeline Notes

- `run_us_daily_pipeline.py` is the standalone US-only Phase 1 runner
- if `US_STOCK_ENABLED=false`, the pipeline is skipped unless `--force` is provided
- `--skip-universe`, `--skip-prices`, `--skip-quality`, `--skip-features` are for controlled manual runs
- if quality status is `FAILED`, feature generation is skipped by default
- `--force-features` can override the default feature skip after a quality failure

## Feature Failure Check Order

1. confirm `DATABASE_URL`
2. confirm US migrations applied
3. confirm active universe rows exist
4. confirm latest price rows exist in `market.us_stock_daily_price`
5. confirm quality report status if needed
6. confirm feature rows exist in `feature.us_stock_feature_daily`
7. inspect pipeline summary logs and ticker-level logs from manual execution

## Safety Principle

The US stock pipeline must not affect Korean auto-trading.

- no connection to `run_daily_scheduler.py` in this phase
- no connection to KIS order files
- no paper trading code
- no live trading code
- US pipeline failure must not break Korean trading operations
