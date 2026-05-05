# Lee_trader_backTest File Index

## 소스 파일 목록
| 파일 | 역할 | 수정 가능 여부 | 수정 시 주의사항 |
| --- | --- | --- | --- |
| `python/walkforward_backtest.py` | expanding-window quarterly walk-forward 실행 | 신중 수정 | 모델 학습/예측/랭킹/성과 단계가 순차 연결됨 |
| `python/run_walkforward_backtest.py` | split schedule 기반 다중 run 실행 | 신중 수정 | `research.dim_model_run` 메타데이터 계약 유지 필요 |
| `python/run_operational_walkforward.py` | 운영용 wrapper, manifest/summary 정리 | 수정 가능 | 결과 복사 경로와 README 생성이 포함됨 |
| `python/walkforward_splits.py` | split schedule 생성 | 수정 가능 | downstream CSV 컬럼명 유지 필요 |
| `python/build_backtest_predictions.py` | point-in-time 예측 및 `research.prediction_history` 적재 | 핵심 파일, 신중 수정 | 모델 출력, 점수 컬럼, run_id 적재 계약 유지 |
| `python/build_backtest_ranking.py` | `research.ranking_history` 생성 | 수정 가능 | `final_score`, `rank`, `in_top_n` 컬럼 계약 유지 |
| `python/build_backtest_outcome.py` | 실제 가격 기반 outcome / maturity 계산 | 핵심 파일, 신중 수정 | `research.backtest_outcome` PK 및 maturity 로직 영향 큼 |
| `python/outcome_maturity.py` | 미래 가격 가용성 / 성과 attach 보조 | 신중 수정 | outcome 품질에 직접 영향 |
| `python/db.py` | `research.dim_model_run` 생성 및 공통 DB 헬퍼 | 신중 수정 | 전 백테스트 run 메타데이터 진입점 |
| `python/build_walk_forward_score_validation_from_runs.py` | run 결과 기반 점수 검증 리포트 | 수정 가능 | 운영 문서/검증 아티팩트와 연결 |
| `python/check_walkforward_runs.py` | run sufficiency 점검 | 수정 가능 | 운영 판단용 summary에 영향 |

## 수정 기준
- `build_backtest_predictions.py`와 `build_backtest_outcome.py`는 연구 결과 해석에 직접 영향을 준다.
- `run_walkforward_backtest.py`의 `config_json` 필드 변경은 `research.dim_model_run` 비교분석 도구와 맞물린다.

## 확인 필요
- `build_backtest_predictions.py` 내부의 legacy 점수 계산 경로가 현재도 실사용 비교 대상으로 남아 있는지 여부는 코드만으로는 최종 운영 의도를 단정하기 어렵다.
