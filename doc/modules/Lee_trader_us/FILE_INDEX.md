# Lee_trader_us File Index

## Python Files

- `python/us/__init__.py`: package marker for the US module
- `python/us/us_config.py`: US environment configuration loader
- `python/us/us_db.py`: shared DB access and upsert/query helpers for US-only modules
- `python/us/load_us_universe.py`: NASDAQ100 universe seed loader
- `python/us/download_us_prices.py`: US OHLCV collector using `yfinance`
- `python/us/validate_us_price_data.py`: daily US price quality validator and quality report writer
- `python/us/build_us_features.py`: baseline price feature builder for `feature.us_stock_feature_daily`
- `python/us/build_us_ranking_placeholder.py`: placeholder entry point for future ranking stage
- `python/us/run_us_daily_pipeline.py`: standalone Phase 1 daily pipeline orchestrator for universe, prices, quality, and features

## Documents

- `doc/modules/Lee_trader_us/README.md`: module purpose and current phase scope
- `doc/modules/Lee_trader_us/CONTEXT.md`: architecture boundaries and long-term direction
- `doc/modules/Lee_trader_us/FLOW.md`: target pipeline flow and current phase boundary
- `doc/modules/Lee_trader_us/ENV.md`: US_STOCK environment variable reference
- `doc/modules/Lee_trader_us/OPERATIONS.md`: manual operations and failure checks

## DDL

- `migrations/us_stock_phase1.sql`: Project C Phase 1 US stock schema bootstrap
- `migrations/us_stock_phase1_2_universe.sql`: Phase 1-2 universe table adjustment migration
- `migrations/us_stock_phase1_3_price_collect.sql`: Phase 1-3 OHLCV price and collect-log alignment
- `migrations/us_stock_phase1_4_quality_report.sql`: Phase 1-4 quality report table alignment
- `migrations/us_stock_phase1_5_feature_daily.sql`: Phase 1-5 feature table alignment

## Data Files

- `data/us/nasdaq100_universe.csv`: static NASDAQ100 universe seed file
