# Lee_trader_ai Context

## 개요

- 이 모듈은 데이터 수집, feature 생성, 모델 예측, 랭킹 산출, AI 주문 preview, 실자동매매, 웹 반영까지 포함합니다.
- AI 운영 경로는 `run_pipeline.py`와 `run_live_auto_trade_cycle.py`를 중심으로 이어집니다.
- 배치, DB, 웹, 실계좌 동기화가 강하게 연결되어 있어 단일 파일 변경도 연쇄 영향이 큽니다.

## 핵심 흐름

- 데이터/예측 파이프라인
  - `fetch_market_data -> fetch_top_universe -> download_prices_kis -> clean_prices -> create_adjusted_prices -> fetch_fundamentals_dart -> quality_builder -> feature_builder -> label_builder -> model_train -> model_predict -> ranking_builder`
- 점수/랭킹
  - `ranking_builder.py`가 예측, 품질, 기술, 리스크, overlay 정보를 합쳐 `final_score`, `live_rank`, `rank_final` 등을 계산합니다.
- 주문 준비
  - `build_trade_intents.py`가 매매 의도를 만듭니다.
  - `build_live_order_preview.py`가 preview와 차단 사유를 생성합니다.
- 실주문
  - `run_live_auto_trade_cycle.py`가 자동매매 사이클을 실행합니다.
  - `submit_live_orders.py`가 KIS 주문 호출 직전 최종 검증과 제출을 담당합니다.
- 웹 반영
  - `sync_web_display_data.py`가 JSON 산출물과 DB payload를 동기화합니다.
  - `node/index.js`와 프론트 JS가 이를 화면에 노출합니다.

## 운영상 주의점

- `ranking_builder.py`는 점수, 순위, 주문 후보, UI 정렬에 동시에 영향을 줍니다.
- `submit_live_orders.py`와 `run_live_auto_trade_cycle.py`는 실주문 경로이므로 보수적으로 수정해야 합니다.
- `DATABASE_URL`과 Postgres 경로가 기본 운영 기준입니다.
- preview와 execution JSON 스키마가 바뀌면 프론트와 API를 같이 수정해야 합니다.
- AI 일반 경로와 RULE 경로는 계좌, 앱키, 실주문 상태를 섞지 않습니다.

## 연관 모듈

- `Lee_trader_score`
  - 점수 계산의 핵심 로직을 공유합니다.
- `Lee_trader_backTest`
  - prediction history, ranking history, outcome 계산과 연결됩니다.
- `Lee_trader_rule`
  - 일부 데이터와 웹 동기화 경로는 공유하지만 주문 경로는 분리됩니다.

## 확인 포인트

- 최신 ranking 산출물이 preview와 화면까지 일관되게 반영되는지
- 실주문 guard가 preview 해석과 서로 어긋나지 않는지
- live account/fill sync 결과가 UI payload에 반영되는지
