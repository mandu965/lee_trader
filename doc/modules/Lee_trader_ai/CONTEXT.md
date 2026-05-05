# Lee_trader_ai Context

## 상세 설명
- 이 모듈은 데이터 적재부터 모델 예측, 점수 계산, 운영 주문 전처리, 웹/API 노출까지를 연결한다.
- 실제 핵심 흐름은 `python/run_pipeline.py`의 `STEPS` 배열에 정의되어 있으며 `fetch_market_data -> fetch_top_universe -> merge_universe -> download_prices_kis -> clean_prices -> create_adjusted_prices -> fetch_fundamentals_dart -> quality_builder -> feature_builder -> label_builder -> model_train -> model_predict -> ranking_builder` 순서로 실행된다.
- `python/ranking_builder.py`는 프로젝트의 단일 랭킹 계산 기준점으로 선언되어 있고, `predictions.csv`, `scores_final.csv`, `features.csv`, `universe.csv`, `market_status.csv`를 합쳐 최종 점수를 계산한다.
- 실거래 쪽은 `python/run_live_auto_trade_cycle.py`와 `python/submit_live_orders.py`가 담당한다. 주문은 `trade_intents.json`에서 시작해 `order_requests_preview.json`, `order_requests_execution.json`으로 이어진다.

## 전략/로직 개요
- 모델 학습
  - `python/model_train.py`는 `target_log_<h>d`, `target_mdd_<h>d`, `target_<h>d_top20` 타깃을 기준으로 LightGBM 회귀/분류 모델을 학습한다.
- 예측 생성
  - `python/model_predict.py`는 종목별 최신 feature row만 사용해 `pred_return_60d`, `pred_return_90d`, `pred_mdd_60d`, `pred_mdd_90d`, `prob_top20_60d`, `prob_top20_90d`를 만든다.
- 점수 계산
  - `python/ranking_builder.py`는 `ret_score`, `prob_score`, `qual_score`, `tech_score`, `risk_penalty`와 시장 regime, theme overlay, confidence 계열 컬럼을 반영해 `final_score`, `final_score_v2`, `final_score_v3`, `live_score`, `rank_final`, `live_rank`를 만든다.
- 주문 의사결정
  - `python/strategy_core.py`는 `apply_execution_policy.py`의 함수들을 조합해 후보군, 보유 종목, 쿨다운, 실행 액션을 평가한다.
  - `python/submit_live_orders.py`는 `common_live_risk_guard`, KIS 계좌 조회, 현재 랭킹 컨텍스트를 결합해 제출 가능 주문만 실행한다.

## 운영상 주의사항
- `DATABASE_URL` 없이는 Postgres 적재 경로가 동작하지 않는다. 일부 파일은 SQLite fallback 분기를 갖지만 기본 운영 경로는 Postgres다.
- `ranking_builder.py`는 컬럼 수가 매우 많고 `daily_ranking` 스키마를 적극적으로 확장한다. 점수 컬럼명을 바꾸면 웹/API와 저장 스키마가 함께 깨질 수 있다.
- `run_live_auto_trade_cycle.py`는 `AUTO_TRADE_CONFIRM_TEXT=LIVE_ORDER`가 없으면 실제 주문 실행을 허용하지 않는다.
- theme overlay 동작은 `ENABLE_THEME_OVERLAY`, `THEME_OVERLAY_MODE`, production config에 의해 달라진다.
- `model_predict.py`는 최신 snapshot 날짜보다 오래된 stale feature row를 예측 전에 제거한다.

## 다른 모듈과의 관계
- `Lee_trader_rule`
  - 독립적인 룰 엔진이지만 웹 동기화는 동일한 `sync_web_display_data.py`와 `research.app_payload_store`를 공유한다.
- `Lee_trader_backTest`
  - 동일한 모델/점수 개념을 재사용하지만 저장 대상은 `research.prediction_history`, `research.ranking_history`, `research.backtest_outcome`이다.
- `node/index.js`
  - `/api/ranking`, `/api/top20`, `/api/trade-intents`, `/api/order-requests-preview`, `/api/order-requests-execution`, `/api/live-account/*` 등으로 AI 운영 결과를 노출한다.

## 확인 필요
- `Lee_trader_ai`의 정확한 원래 범위가 사용자 정의 개념인지, 아니면 현재 `python/` 전체 AI 경로를 의미하는지 저장소 내 명시가 없다.
