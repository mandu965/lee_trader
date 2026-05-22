# Lee_trader_ai Context

## Overview

- `Lee_trader_ai` covers the Korean AI recommendation, ranking, preview, and live auto-trading path.
- The main runtime path is centered on `run_pipeline.py` and `run_live_auto_trade_cycle.py`.
- Ranking, preview payloads, order submission, and web sync are tightly coupled, so changes in this area require extra caution.

## Main Runtime Flow

- Data / feature / model path:
  - `fetch_market_data -> download_prices_kis -> clean_prices -> create_adjusted_prices -> quality_builder -> feature_builder -> label_builder -> model_train -> model_predict -> ranking_builder`
- Order preparation path:
  - `build_trade_intents.py -> build_live_order_preview.py -> submit_live_orders.py`
- Live operation path:
  - `run_live_auto_trade_cycle.py -> sync_live_order_fills.py -> sync_live_account_holdings.py -> sync_web_display_data.py`

## Operational Notes

- `ranking_builder.py` affects score output, live ranking, preview payloads, and UI payload ordering.
- `submit_live_orders.py` and `run_live_auto_trade_cycle.py` are part of the Korean live order path and must be treated as high-risk files.
- `DATABASE_URL` and Postgres are the default persistence path for current AI operations.

## Related Modules

- `Lee_trader_score`
  - shared score calculation logic
- `Lee_trader_backTest`
  - prediction history, ranking history, and outcome analysis
- `Lee_trader_rule`
  - Korean rule-based trading path — **운영 중단 (2026-05-22, `RULE_LIVE_ENABLED=0`)**. 소스는 유지되나 AI 경로와 독립적으로 비활성 상태.

## Project C Phase 2-2

- Project C is expanding into a separate US stock financial data track.
- Phase 2-2 implements a standalone `yfinance`-based financial raw collector.
- This phase covers raw financial statement / metric collection only.
- Feature generation, label generation, ranking integration, paper trading, and live trading are still out of scope.

## Project C Phase 2-3

- Phase 2-3 adds a standalone financial feature layer on top of the US raw financial tables.
- Raw financial data and financial features are intentionally separated:
  - raw tables preserve nullable source values
  - feature tables store growth, margin, stability, valuation, and score-skeleton fields
- Financial features remain separate from `feature.us_stock_feature_daily` because fiscal periods do not share the same time axis as daily price features.
- This phase still does not connect US financial features to ranking, label generation, model training, paper trading, or live trading.

## Project C Phase 2-4

- Phase 2-4 adds standalone relative strength features versus `SPY` and `QQQ`.
- This layer is daily and price-based, but it is still kept separate from the existing Phase 1 daily feature table to avoid changing the current daily feature builder contract.
- Relative strength features capture stock return minus benchmark return over fixed trading-day windows.
- This phase still does not connect relative strength features to ranking, label generation, model training, paper trading, or live trading.

## Project C Phase 2-5

- Phase 2-5 adds standalone US stock label generation and dataset validation.
- Labels are forward-return based and are built from future trading-day prices only.
- Dataset validation focuses on row coverage, null ratios, label distribution, duplicate keys, and leakage-risk notes.
- This phase still does not run model training, ranking, recommendation, paper trading, or live trading.

## Separation Principle

- Korean auto-trading and US financial data collection must remain operationally separate.
- Failure in a future US financial collector must not stop Korean AI or RULE pipelines.
- `US_FINANCIAL_*` settings are reserved for the future Project C collector only.
- `US_FINANCIAL_FEATURE_*` settings are reserved for the standalone Project C financial feature builder only.
- `US_RELATIVE_STRENGTH_*` settings are reserved for the standalone Project C relative strength builder only.
- `US_LABEL_*` settings are reserved for the standalone Project C label builder only.
- `US_DATASET_*` settings are reserved for the standalone Project C dataset validator only.
- Phase 2-3 must not modify Korean KIS order execution, Korean scoring, or Korean scheduler wiring.
