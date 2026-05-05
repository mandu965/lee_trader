# Lee_trader_backTest

## 모듈 목적
- 실제 저장소에는 `Lee_trader_backTest` 폴더가 없으며, 이 문서는 `walkforward_*`, `build_backtest_*`, `run_operational_walkforward.py` 계열 파일을 기준으로 정리한다.
- 목적은 모델 기반 백테스트와 walk-forward 검증을 수행하고, 결과를 `research.prediction_history`, `research.ranking_history`, `research.backtest_outcome`에 적재하는 것이다.

## 핵심 기능
- `python/walkforward_backtest.py`: 분기별 expanding-window walk-forward 실행
- `python/run_walkforward_backtest.py`: split schedule 기반 다중 run 생성
- `python/run_operational_walkforward.py`: 운영용 wrapper 및 run artifact 정리
- `python/build_backtest_predictions.py`: 시점별 예측/점수 생성 후 `research.prediction_history` 적재
- `python/build_backtest_ranking.py`: 예측 결과를 날짜별 rank로 변환 후 `research.ranking_history` 적재
- `python/build_backtest_outcome.py`: 실제 가격 기반 outcome / maturity 계산 후 `research.backtest_outcome` 적재
- `python/walkforward_splits.py`: split schedule 생성

## 입력 데이터
- `data/features.csv`
- `data/labels.csv`
- `data/market_status.csv`
- `data/model.pkl` 또는 `artifacts/models/run_<run_id>_<model_version>.pkl`
- split schedule CSV
- DB
  - `research.dim_model_run`
  - `research.prediction_history`
  - `research.ranking_history`
  - `research.backtest_outcome`
  - 가격 참조용 실제 price history

## 출력 데이터
- DB
  - `research.dim_model_run`
  - `research.prediction_history`
  - `research.ranking_history`
  - `research.backtest_outcome`
- 파일
  - `outputs/walkforward_run_summary.csv`, `.md`
  - `outputs/walk_forward_score_validation.csv`, `.md`
  - `data/history/walkforward_runs/<timestamp>*/manifest.json`
  - `data/history/walkforward_runs/<timestamp>*/README.md`

## 주요 실행 파일
- `python/walkforward_backtest.py`
- `python/run_walkforward_backtest.py`
- `python/run_operational_walkforward.py`
- `python/build_backtest_predictions.py`
- `python/build_backtest_ranking.py`
- `python/build_backtest_outcome.py`
- `python/walkforward_splits.py`
