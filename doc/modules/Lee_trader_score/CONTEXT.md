# Lee_trader_score Context

## 개요

- 이 모듈은 종목별 최종 점수와 운영 정렬 기준을 정의합니다.
- 핵심은 `ranking_builder.py`와 `scoring/final_score.py`이며, 설명 컬럼과 confidence 계열 보조 계산이 이어집니다.
- 점수 수식 변경은 ranking, top 후보, preview, 화면 정렬에 모두 영향을 줍니다.

## 핵심 흐름

- 입력 결합
  - 예측값, quality/technical feature, universe 정보, market status를 결합합니다.
- component score 계산
  - `ret_score`, `prob_score`, `qual_score`, `tech_score`와 보조 진단 점수를 계산합니다.
- risk/regime 반영
  - market regime과 `pred_mdd_*` 기반 penalty를 반영합니다.
- final score 계산
  - `final_score`, `final_score_v2`, `final_score_v3`를 계산합니다.
- 운영 정렬
  - runtime flag와 production config에 따라 `live_rank`, `rank_final` 기준이 결정됩니다.
- 설명/검증
  - `score_explainer.py`와 confidence 관련 스크립트가 해석 정보를 보강합니다.

## 운영상 주의점

- 현재 운영 정렬 기준은 수식 자체보다 runtime 설정에도 영향을 받습니다.
- `final_score`, `live_rank`, `rank_final` 기준이 바뀌면 `RUNTIME_SORTING.md`를 같이 갱신해야 합니다.
- 점수 컬럼명 변경은 API, DB, UI, 검증 리포트까지 연쇄 영향이 있습니다.
- 실험용 점수 변경과 운영 기준 변경은 한 번에 섞지 않는 편이 안전합니다.
- 점수 수식 버전 전환은 `SCORE_FORMULA_VERSION` 환경변수로 제어됩니다. 운영 모드에서는 `ranking_builder.resolve_score_formula_version()`의 가드가 적용되므로 환경변수 단독으로는 전환되지 않습니다. 승격 기준 전체는 `doc/shadow_promotion_criteria.md` 참조.

## 연관 모듈

- `Lee_trader_ai`
  - 점수 산출과 운영 랭킹을 직접 사용합니다.
- `Lee_trader_backTest`
  - 백테스트에서 동일하거나 유사한 점수 개념을 사용합니다.
- `Lee_trader_rule`
  - 직접 같은 점수를 쓰는 것은 아니지만, UI/DB payload 해석과 운영 문서 관점에서 연결됩니다.

## 확인 포인트

- 현재 runtime flag 기준 정렬이 문서 설명과 일치하는지
- 점수 수식 변경 후 ranking 결과와 상위 후보 구성이 의도대로 달라졌는지
- score explain, confidence, overlay 해석이 화면과 일치하는지
