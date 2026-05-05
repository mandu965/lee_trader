# Lee_trader_ai Flow

## 실행 흐름
1. `python/run_pipeline.py`
2. `fetch_market_data.py`, `fetch_top_universe.py`, `merge_universe.py`
3. `download_prices_kis.py`, `clean_prices.py`, `create_adjusted_prices.py`
4. `fetch_fundamentals_dart.py`, `quality_builder.py`, `feature_builder.py`, `label_builder.py`
5. `model_train.py`
6. `model_predict.py`
7. `ranking_builder.py`
8. 후속 리포트/보조 단계
   - `build_confidence_calibration_map.py`
   - `analyze_top20_meaningfulness.py`
   - `build_confidence_score_v2.py`
   - `build_top20_buyability_report.py`
   - `build_walkforward_acceptance.py`
   - `sync_csv_db_parity.py`

## 주요 함수 호출 순서
- 배치
  - `run_pipeline.py`의 `STEPS`
  - `create_model_run_id()`
  - `maybe_run_theme_overlay_steps()`
- 예측
  - `model_predict.load_model()`
  - `model_predict.load_features_latest()`
  - `model_predict.predict_all()`
  - `model_predict.save_predictions()`
- 랭킹
  - `ranking_builder.main()`
  - shared scoring 함수
    - `compute_component_scores`
    - `apply_baseline_final_score`
    - `attach_market_columns`
    - `compute_risk_penalty`
- 실거래
  - `run_live_auto_trade_cycle.main()`
  - `submit_live_orders.main()`
  - KIS 호출
    - `inquire_balance`
    - `inquire_psbl_order`
    - `order_cash`

## 데이터 흐름
- `features.csv` + `labels.csv` -> `model.pkl`
- `features.csv` + `model.pkl` -> `predictions.csv` / `public.predictions`
- `predictions.csv` + `scores_final.csv` + `features.csv` + `universe.csv` + `market_status.csv` -> `ranking_final.csv` / `public.daily_ranking`
- `ranking_final.csv` + holdings + 정책 -> `trade_intents.json`
- `trade_intents.json` + KIS 계좌/시세 -> `order_requests_preview.json` -> `order_requests_execution.json`
- JSON/CSV 산출물 -> `sync_web_display_data.py` -> `research.app_payload_store` -> `node/index.js` API

## 외부 의존성
- Python
  - `pandas`
  - `numpy`
  - `sqlalchemy`
  - `lightgbm`
  - `scikit-learn`
- 서비스
  - Postgres (`DATABASE_URL`)
  - 한국투자증권 KIS API
- 웹
  - Express / Node.js

## 확인 필요
- `trade_intents.json` 생성의 정확한 엔트리포인트는 이 문서 작성에 사용한 범위에서는 직접 확인하지 못했다. 관련 후보 파일은 `python/build_trade_intents.py`, `python/run_operational_refresh.py`다.
