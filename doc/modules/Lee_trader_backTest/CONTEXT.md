# Lee_trader_backTest Context

## 상세 설명
- 백테스트 모듈은 운영용 AI 점수 체계를 과거 시점 기준으로 재현하고, run 단위로 예측/랭킹/성과를 저장한다.
- run 메타데이터는 `db.create_research_model_run()`을 통해 `research.dim_model_run`에 기록된다.
- 단순 일회성 백테스트보다 walk-forward를 중요하게 다루며, split 단위 run을 여러 개 생성한다.

## 전략/로직 개요
- run 생성
  - `walkforward_backtest.py`는 expanding-window quarterly window를 만든 뒤, 각 window마다 train/predict 구간을 분리한다.
  - `run_walkforward_backtest.py`는 사전 생성된 split schedule CSV를 기반으로 split별, horizon별 run을 생성한다.
- 예측/점수
  - `build_backtest_predictions.py`는 model pack으로 특정 `as_of_date` 범위의 예측을 만들고, feature/market_status를 결합해 `final_score`까지 계산한 뒤 `research.prediction_history`에 append한다.
- 랭킹
  - `build_backtest_ranking.py`는 `final_score` 내림차순으로 rank를 만들고 `in_top_n` 플래그를 계산한다.
- 성과
  - `build_backtest_outcome.py`는 실제 가격 이력을 이용해 `realized_return`, `realized_mdd`, `is_matured`, `maturity_status`를 계산한다.

## 운영상 주의사항
- `build_backtest_outcome.py`는 현재 `horizon_days in {60, 90}`에 대해 검증됐다고 경고를 남긴다.
- `run_walkforward_backtest.py`는 split당 horizon별로 별도 run을 만든다. 90일 horizon에서 빈 결과가 나오면 warning을 남기도록 구현돼 있다.
- `build_backtest_predictions.py`는 legacy 점수 계산 함수와 shared score 함수 경로를 둘 다 포함한다. 어떤 경로가 실제 운영 비교 기준인지 해석할 때 주의가 필요하다.
- 백테스트 결과는 운영 `public.daily_ranking`이 아니라 `research.*` 히스토리 테이블에 적재된다.

## 다른 모듈과의 관계
- `Lee_trader_ai`
  - 동일한 model pack, feature CSV, market status, scoring 로직 일부를 공유한다.
- `Lee_trader_rule`
  - `rule_backtest.py`는 규칙 전략 전용 백테스트로 별도 경로이며, 본 모듈의 `research.prediction_history` 흐름과는 다르다.
- 분석/리포트
  - `build_walk_forward_score_validation_from_runs.py`
  - `check_walkforward_runs.py`
  - 관련 결과는 `outputs/`와 `data/history/walkforward_runs/`에 저장된다.

## 확인 필요
- 실제 price history를 `outcome_maturity.load_price_history()`가 어느 우선순위 파일/테이블에서 읽는지는 이번 문서 작성 범위에서 직접 끝까지 추적하지 못했다.
