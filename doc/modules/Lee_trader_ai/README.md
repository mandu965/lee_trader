# Lee_trader_ai

## Purpose

이 모듈은 AI 기반 선별, 랭킹, 자동매매 후보 해석과 실자동매매 운영 흐름을 정리합니다.

범위:
- 데이터 수집과 feature 생성
- 모델 학습/예측
- 최종 점수와 랭킹 생성
- AI 주문 preview / execution
- live 계좌/체결 동기화
- 운영 화면과 web payload 반영

## Main Files

- `python/run_pipeline.py`
- `python/model_train.py`
- `python/model_predict.py`
- `python/ranking_builder.py`
- `python/run_operational_refresh.py`
- `python/run_live_auto_trade_cycle.py`
- `python/submit_live_orders.py`
- `python/sync_live_account_holdings.py`
- `python/sync_live_order_fills.py`
- `python/sync_web_display_data.py`

## Main Outputs

- `data/predictions.csv`
- `data/ranking_final.csv`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`
- `outputs/live_trade_review_report.json`
- `serving/daily_recommendations.json`

## Read First

- [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/CONTEXT.md>)
- [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FLOW.md>)
- [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FILE_INDEX.md>)
- [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/ENV.md>)
- [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/OPERATIONS.md>)
