# Lee_trader_backTest File Index

## 목적

walk-forward 검증, prediction history, outcome 계산, RULE 포트폴리오 백테스트 관련 핵심 파일 인덱스입니다.
이 모듈은 연구 재현성과 결과 비교가 중요하므로 입력 기간, split 기준, run metadata를 항상 같이 확인합니다.

## 핵심 파일

| 파일 | 역할 | 수정 위험도 | 함께 확인할 파일 |
| --- | --- | --- | --- |
| `python/walkforward_backtest.py` | walk-forward 전체 실행 엔진 | 매우 높음 | `walkforward_splits.py`, `build_backtest_predictions.py`, `build_backtest_outcome.py` |
| `python/run_walkforward_backtest.py` | split 스케줄 기반 다회 실행 진입점 | 높음 | `walkforward_backtest.py`, `db.py` |
| `python/run_operational_walkforward.py` | 운영형 wrapper와 결과 정리 | 중간 | `run_walkforward_backtest.py`, `build_walk_forward_score_validation_from_runs.py` |
| `python/walkforward_splits.py` | expanding-window split 생성 | 높음 | `run_walkforward_backtest.py`, `walkforward_backtest.py` |
| `python/build_backtest_predictions.py` | point-in-time prediction history 생성 | 매우 높음 | `model_predict.py`, `db.py`, `build_backtest_ranking.py` |
| `python/build_backtest_ranking.py` | prediction history를 ranking history로 변환 | 높음 | `ranking_builder.py`, `build_backtest_predictions.py` |
| `python/build_backtest_outcome.py` | 실제 가격 기준 outcome와 maturity 계산 | 매우 높음 | `outcome_maturity.py`, `db.py` |
| `python/outcome_maturity.py` | 미래 성과 부착과 maturity 판정 유틸리티 | 높음 | `build_backtest_outcome.py` |
| `python/build_walk_forward_score_validation_from_runs.py` | run 결과 기준 점수 검증 리포트 생성 | 중간 | `check_walkforward_runs.py`, `walkforward_compare.py` |
| `python/check_walkforward_runs.py` | run 누락/충분성 점검 | 중간 | `db.py`, `run_operational_walkforward.py` |
| `python/rule_portfolio_backtest.py` | RULE 포트폴리오 기준 거래/자산곡선 백테스트 | 높음 | `rule_portfolio_manager.py`, `rule_signal_builder.py` |
| `python/db.py` | run metadata와 이력 테이블 적재 | 높음 | 대부분의 backtest 배치 |

## 보조 파일

| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `python/walkforward_compare.py` | run 간 결과 비교 | 실험 차이 분석용 |
| `python/build_backtest_predictions.py` | 백테스트 예측 생성 | AI 모델 버전 변경 시 반드시 재검토 |
| `python/build_benchmark_comparison.py` | 벤치마크 비교 리포트 | 성과 해석 보강용 |
| `outputs/rule_portfolio_backtest_trades.csv` | 개별 거래 결과 | RULE 성능 원인 분석용 |
| `outputs/rule_portfolio_backtest_equity.csv` | 자산곡선 | MDD, CAGR 검토용 |

## 수정 원칙

- split 기준 변경은 과거 결과와 직접 비교가 깨지므로 문서와 결과 폴더를 같이 갱신합니다.
- `build_backtest_predictions.py`와 `build_backtest_outcome.py`는 재현성 핵심 파일로 취급합니다.
- RULE 백테스트 수치가 바뀌면 `docs/rule_backtest_comparison.md` 같은 해석 문서도 같이 갱신합니다.
