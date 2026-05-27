# Lee_trader_backTest

## Purpose

이 모듈은 백테스트와 walk-forward 검증 흐름을 정리합니다.

범위:
- walk-forward split 설계
- 모델 기반 prediction history 생성
- ranking history 생성
- outcome / maturity 계산
- 성과 비교와 검증 리포트 생성

> RULE 포트폴리오 백테스트 (`rule_portfolio_backtest.py`)는 RULE 서비스 종료(2026-05-21) 이후 이력 보관용으로만 유지됩니다.

## Main Files

- `python/walkforward_backtest.py`
- `python/run_walkforward_backtest.py`
- `python/run_operational_walkforward.py`
- `python/build_backtest_predictions.py`
- `python/build_backtest_ranking.py`
- `python/build_backtest_outcome.py`
- `python/walkforward_splits.py`
- `python/rule_portfolio_backtest.py`

## Main Outputs

- `outputs/walkforward_run_summary.csv`
- `outputs/walk_forward_score_validation.csv`
- `outputs/rule_portfolio_backtest_report.json`
- `outputs/rule_portfolio_backtest_report.md`
- `outputs/rule_portfolio_backtest_trades.csv`
- `outputs/rule_portfolio_backtest_equity.csv`

## Read First

- [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/CONTEXT.md>)
- [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/FLOW.md>)
- [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/FILE_INDEX.md>)
- [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/ENV.md>)
- [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/OPERATIONS.md>)
