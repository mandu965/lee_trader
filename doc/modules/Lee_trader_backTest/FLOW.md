# Lee_trader_backTest Flow

## 실행 흐름
1. split 정의
   - `walkforward_splits.py` 또는 `walkforward_backtest.py` 내부 window 생성
2. run 메타 생성
   - `db.create_research_model_run()`
3. 모델 학습
   - `model_train.py`
4. 예측/점수 적재
   - `build_backtest_predictions.py`
   - 대상: `research.prediction_history`
5. 랭킹 적재
   - `build_backtest_ranking.py`
   - 대상: `research.ranking_history`
6. 실제 성과 적재
   - `build_backtest_outcome.py`
   - 대상: `research.backtest_outcome`
7. 요약/검증
   - `check_walkforward_runs.py`
   - `build_walk_forward_score_validation_from_runs.py`

## 주요 함수 호출 순서
- `walkforward_backtest.py`
  - `load_feature_dates()`
  - `build_quarterly_windows()`
  - `run_walkforward_window()`
- `run_walkforward_backtest.py`
  - `load_splits()`
  - `create_run_for_split()`
  - `run_command()` for
    - `build_backtest_predictions.py`
    - `build_backtest_ranking.py`
    - `build_backtest_outcome.py`
- `build_backtest_predictions.py`
  - `load_model()`
  - `load_features()`
  - `parse_dates()`
  - `predict_for_date()`
  - `compute_scores()`
- `build_backtest_outcome.py`
  - `load_predictions()`
  - `build_outcome_rows()`
  - `build_run_summary()`

## 데이터 흐름
- `features.csv` + `labels.csv` -> `model_train.py` -> model artifact
- model artifact + feature date slice -> `research.prediction_history`
- `research.prediction_history` -> `research.ranking_history`
- 실제 가격 이력 + `research.prediction_history` key -> `research.backtest_outcome`
- run summary / validation -> `outputs/` 및 `data/history/walkforward_runs/`

## 외부 의존성
- Python
  - `pandas`
  - `numpy`
  - `sqlalchemy`
  - `lightgbm`
- DB
  - Postgres `research.dim_model_run`
  - `research.prediction_history`
  - `research.ranking_history`
  - `research.backtest_outcome`

## 확인 필요
- 운영 wrapper인 `run_operational_walkforward.py`가 생성하는 README 일부 문자열은 인코딩이 깨져 보이며, 원문 의도 해석은 추가 확인이 필요하다.
