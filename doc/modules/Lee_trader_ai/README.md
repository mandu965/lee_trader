# Lee_trader_ai

## 모듈 목적
- 실제 저장소에는 `Lee_trader_ai`라는 별도 소스 폴더가 없으며, 이 문서는 `python/run_pipeline.py`를 중심으로 한 AI/랭킹/실행 파이프라인을 기준으로 정리한다.
- 목적은 일별 데이터 수집, 특성 생성, 모델 학습/예측, 최종 랭킹 생성, 주문 의사결정 산출과 웹 노출까지의 AI 기반 운영 흐름을 관리하는 것이다.

## 핵심 기능
- `python/run_pipeline.py`: 일일 배치 오케스트레이션
- `python/model_train.py`: LightGBM 회귀/분류 모델 학습 및 `model.pkl` 패키징
- `python/model_predict.py`: 최신 feature snapshot 기반 예측 생성
- `python/ranking_builder.py`: `final_score`, `final_score_v2`, `final_score_v3`, `rank_final`, `live_rank` 생성
- `python/run_live_auto_trade_cycle.py`: 운영 리프레시, 주문 제출, 체결/계좌/리뷰 동기화
- `python/submit_live_orders.py`: `trade_intents.json`을 실제 주문 요청으로 변환 및 제출
- `python/sync_web_display_data.py`: core table / payload를 웹 DB로 동기화

## 입력 데이터
- CSV
  - `data/features.csv`
  - `data/labels.csv`
  - `data/universe.csv`
  - `data/market_status.csv`
  - `data/predictions.csv`
  - `data/scores_final.csv`
  - `data/live_account_holdings.csv`
- 모델 파일
  - `data/model.pkl`
  - `artifacts/models/run_<run_id>_<model_version>.pkl`
- DB
  - `public.features`
  - `public.predictions`
  - `public.daily_ranking`
  - `research.app_payload_store`
  - `research.live_trade_decision`, `research.live_order_request`, `research.live_order_execution`, `research.live_order_fill`, `research.live_position_snapshot`, `research.live_trade_review`
- 환경 변수
  - `DATABASE_URL`
  - `WEB_DATABASE_URL`
  - `MODEL_VERSION`
  - `HORIZON_DAYS`
  - `TOP_N`
  - `AUTO_TRADE_EXECUTE`
  - `AUTO_TRADE_ALLOW_BUY`

## 출력 데이터
- CSV/JSON
  - `data/predictions.csv`
  - `data/ranking_final.csv`
  - `outputs/trade_intents.json`
  - `outputs/order_requests_preview.json`
  - `outputs/order_requests_execution.json`
  - `outputs/live_account_balance_summary.json`
  - `outputs/live_trade_review_report.json`
  - `serving/daily_recommendations.json`
  - `serving/model_portfolio.json`
- DB
  - `public.predictions`
  - `public.daily_ranking`
  - `research.app_payload_store`
  - live trade ledger 관련 `research.*` 테이블

## 주요 실행 파일
- `python/run_pipeline.py`
- `python/run_operational_refresh.py`
- `python/run_live_auto_trade_cycle.py`
- `python/model_train.py`
- `python/model_predict.py`
- `python/ranking_builder.py`
- `python/submit_live_orders.py`
- `python/sync_web_display_data.py`
