# Lee_trader_ai Module Interface

## Korean AI Runtime

- `run_pipeline.py`
  - builds Korean AI features, labels, model outputs, and ranking
- `run_live_auto_trade_cycle.py`
  - handles Korean live auto-trading flow

These runtime paths are high risk and must remain isolated from Project C US expansion work.

## Project C Phase 2-2 -> Phase 2-3

### Raw Financial Collector

- module: `python.us.collect_us_financials_yfinance`
- input:
  - `market.us_stock_universe`
- output:
  - `raw.us_stock_financial_statement`
  - `raw.us_stock_financial_metric`

### Financial Feature Builder

- module: `python.us.build_us_financial_features`
- input:
  - `raw.us_stock_financial_statement`
  - `raw.us_stock_financial_metric`
- output:
  - `feature.us_stock_financial_feature`

### Relative Strength Builder

- module: `python.us.build_us_relative_strength_features`
- input:
  - `market.us_stock_daily_price`
- output:
  - `feature.us_stock_relative_strength_daily`

### Label Builder

- module: `python.us.build_us_stock_labels`
- input:
  - `market.us_stock_daily_price`
- output:
  - `label.us_stock_label_daily`

### Dataset Validator

- module: `python.us.validate_us_stock_ml_dataset`
- input:
  - `feature.us_stock_feature_daily`
  - `feature.us_stock_relative_strength_daily`
  - `feature.us_stock_financial_feature`
  - `label.us_stock_label_daily`
- output:
  - `reports/us_stock_dataset_validation.md`

### Current Boundary

- raw financial collection and financial feature generation are standalone only
- no connection to:
 - no connection to:
  - Korean AI ranking
  - Korean RULE ranking
  - label generation
  - model training
  - paper trading
  - live trading
  - `run_daily_scheduler.py`

## Future Join Pattern

- `feature.us_stock_feature_daily`
  - daily price/volume/technical feature layer
- `feature.us_stock_financial_feature`
  - fiscal-period financial feature layer
- `feature.us_stock_relative_strength_daily`
  - daily stock-vs-benchmark relative strength layer
- `label.us_stock_label_daily`
  - future-return label layer

Future modeling can join these layers with explicit as-of logic, but that integration is intentionally out of scope through Phase 2-5.
