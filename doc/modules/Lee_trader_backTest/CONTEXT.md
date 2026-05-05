# Lee_trader_backTest Context

## 개요

- 이 모듈은 walk-forward 검증, prediction history 적재, ranking history 생성, outcome 계산, RULE 포트폴리오 백테스트를 다룹니다.
- 목적은 현재 운영 로직을 과거 시점 기준으로 재현하고, 전략 성과와 안정성을 비교 가능한 형태로 남기는 것입니다.

## 핵심 흐름

- split 생성
  - `walkforward_splits.py`가 expanding-window 기준 split을 만듭니다.
- run 실행
  - `walkforward_backtest.py`, `run_walkforward_backtest.py`가 split별 run을 수행합니다.
  - run metadata는 `research.dim_model_run`에 기록됩니다.
- prediction/ranking history
  - `build_backtest_predictions.py`가 point-in-time prediction을 적재합니다.
  - `build_backtest_ranking.py`가 ranking history를 생성합니다.
- outcome 계산
  - `build_backtest_outcome.py`가 실제 가격을 바탕으로 `realized_return`, `realized_mdd`, `maturity_status` 등을 계산합니다.
- RULE 포트폴리오 검증
  - `rule_portfolio_backtest.py`가 RULE 전략의 거래와 자산곡선을 생성합니다.

## 운영상 주의점

- split 기준이나 horizon 기준을 바꾸면 과거 결과와 직접 비교가 깨집니다.
- `build_backtest_predictions.py`와 `build_backtest_outcome.py`는 재현성 핵심 파일입니다.
- 백테스트 결과는 운영 `public.daily_ranking`이 아니라 `research.*` 이력 테이블과 `outputs/` 산출물에 남습니다.
- RULE 백테스트 규칙이 바뀌면 성과 해석 문서도 같이 갱신해야 합니다.

## 연관 모듈

- `Lee_trader_ai`
  - 모델, feature, scoring 개념을 공유합니다.
- `Lee_trader_rule`
  - RULE 전략 백테스트와 운영 전략 비교에 연결됩니다.

## 확인 포인트

- run metadata, split 기준, horizon 기준이 문서와 일치하는지
- prediction history와 outcome 테이블이 같은 기준일을 보고 있는지
- 백테스트 수치 변경 시 비교 문서까지 같이 갱신됐는지
