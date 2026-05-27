# Lee_trader_ai Context

> **기준일: 2026-05-27**

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
  - **서비스 종료 (2026-05-21, `RULE_LIVE_ENABLED=0`)**. 코드는 유지되나 스케줄러 비활성.

## US Stock (미운영)

- US 주식 관련 코드(`python/us/`)와 환경변수(`US_*`)는 코드베이스에 존재하지만 현재 미운영.
- 스케줄러(`scheduler-us-*`)는 비활성화 상태.
- 한국 AI 파이프라인과 완전 독립적으로 설계되어 있어, 미운영 상태에서도 KR 경로에 영향 없음.
- US 서비스 재개 시 별도 설계 문서 작성 필요.
