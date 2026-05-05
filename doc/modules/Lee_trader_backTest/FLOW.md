# Lee_trader_backTest Flow

## Execution Flow

### 1. Walkforward Split

주요 파일:

- `python/walkforward_splits.py`
- `python/walkforward_backtest.py`

역할:

- 기간 분할
- expanding-window / quarterly split 생성

### 2. Run Creation

주요 파일:

- `python/run_walkforward_backtest.py`
- `python/run_operational_walkforward.py`

역할:

- split별 run 생성
- `research.dim_model_run` 메타데이터 적재
- run artifact 정리

### 3. Prediction / Ranking / Outcome

주요 파일:

- `python/build_backtest_predictions.py`
- `python/build_backtest_ranking.py`
- `python/build_backtest_outcome.py`

역할:

1. point-in-time prediction history 생성
2. ranking history 생성
3. 실제 가격 기반 outcome / maturity 계산

### 4. Validation / Summary

주요 파일:

- `python/check_walkforward_runs.py`
- `python/build_walk_forward_score_validation_from_runs.py`
- `python/rule_portfolio_backtest.py`

역할:

- run sufficiency 확인
- score validation 생성
- RULE 포트폴리오 백테스트 요약 생성

## Data Flow

### Input

- `data/features.csv`
- `data/labels.csv`
- 모델 artifact
- 실제 가격 이력
- Postgres history tables

### Output

- `research.prediction_history`
- `research.ranking_history`
- `research.backtest_outcome`
- `outputs/walkforward_run_summary.csv`
- `outputs/walk_forward_score_validation.csv`
- `outputs/rule_portfolio_backtest_report.json`
- `outputs/rule_portfolio_backtest_trades.csv`
- `outputs/rule_portfolio_backtest_equity.csv`

## Main Checks

- 기간 분할이 point-in-time 조건을 지키는지
- prediction / ranking / outcome key가 일치하는지
- run metadata가 추적 가능한지
- CAGR, MDD, Sharpe, 거래 수가 기대 범위인지

## Notes

- RULE 포트폴리오 백테스트는 AI walk-forward와 별도 산출물이지만, 운영 해석상 같이 읽히는 경우가 많습니다.
- 비교 문서는 [docs/rule_backtest_comparison.md](</d:/ai/lee_trader/docs/rule_backtest_comparison.md>)를 참고합니다.
