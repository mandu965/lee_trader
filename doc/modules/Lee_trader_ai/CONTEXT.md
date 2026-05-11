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
  - Korean rule-based trading path kept separate from the AI order path

## Project C Phase 2-2

- Project C is expanding into a separate US stock financial data track.
- Phase 2-2 implements a standalone `yfinance`-based financial raw collector.
- This phase covers raw financial statement / metric collection only.
- Feature generation, label generation, ranking integration, paper trading, and live trading are still out of scope.

## Separation Principle

- Korean auto-trading and US financial data collection must remain operationally separate.
- Failure in a future US financial collector must not stop Korean AI or RULE pipelines.
- `US_FINANCIAL_*` settings are reserved for the future Project C collector only.
- Phase 2-2 must not modify Korean KIS order execution, Korean scoring, or Korean scheduler wiring.
