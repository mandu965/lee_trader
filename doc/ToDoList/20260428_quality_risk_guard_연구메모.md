# 2026-04-28 quality_risk_guard 연구 메모

## 결론

`quality_risk_guard`는 바로 production 랭킹을 바꾸는 기능이 아니라, 기존 점수는 유지한 채 품질이 너무 낮거나 리스크가 높은 종목만 소폭 감점하는 shadow 가드입니다.

현재 산출물 기준으로는 `READY_FOR_PROMOTION_REVIEW`까지는 타당합니다. 다만 `walkforward_acceptance`가 아직 `REJECTED`이므로, 즉시 production 승격보다는 shadow 유지 + 산출물 안정화 + 추가 관찰이 맞습니다.

## 현재 공식

위치: `python/ranking_builder.py`

- `qual_score < 20`이면 6점 감점
- `risk_penalty >= 12`이면 4점 감점
- 최종 shadow 점수: `final_score - shadow_quality_risk_guard_penalty`
- 산출 컬럼:
  - `shadow_quality_risk_guard_penalty`
  - `shadow_quality_risk_guard_applied`
  - `shadow_final_score_quality_risk_guard`
  - `shadow_rank_quality_risk_guard`

이 방식은 가중치 전체를 바꾸는 것이 아니라 기존 production 점수 위에 작은 safety overlay를 얹는 구조라서, rebalanced weight보다 운영 리스크가 낮습니다.

## 증거 요약

기준 산출물: `outputs/walkforward_weight_variant_analysis.md`

| 항목 | baseline | quality_risk_guard |
| --- | ---: | ---: |
| top20 평균수익률 | 34.13% | 40.47% |
| top50 평균수익률 | 36.72% | 36.38% |
| universe 평균수익률 | 27.75% | 27.75% |
| top20 평균 MDD | -27.11% | -26.57% |
| score/return corr | 0.0390 | 0.0448 |
| top20 저품질 수 | 9 | 4 |
| top20 고위험 수 | 7 | 5 |
| ordering | NO | YES |

baseline은 top20이 top50보다 약해서 ordering이 깨졌고, `quality_risk_guard`는 top20 > top50 > universe ordering을 회복했습니다. 같은 비교에서 `rebalanced_weights`는 저품질/고위험 수를 더 줄였지만 top20 성과가 baseline보다 낮아졌습니다. 따라서 현재 방향은 전면 재가중보다 guard 방식이 더 낫습니다.

## 승격 보류 사유

`outputs/walkforward_acceptance.md` 기준 전체 walk-forward acceptance는 아직 `REJECTED`입니다.

주요 사유:

- `ordering_not_stable`
- `drawdown_too_deep`
- `confidence_monotonicity_missing`
- 실거래 체결 기반 evidence는 아직 제한적이거나 unavailable

또한 variant 분석의 최신 matured date가 2025-12-09라서, 현재 실운영 국면까지 충분히 검증됐다고 보기에는 증거가 얇습니다.

## 이번 점검 중 발견한 산출물 리스크

`outputs/shadow_quality_risk_guard_daily_report.json`에 JSON 표준이 아닌 `NaN` 값이 포함되어 있었습니다. 브라우저/Node의 `JSON.parse` 같은 엄격 파서는 이 파일을 읽지 못합니다.

조치:

- 현재 산출물의 `NaN`을 `null`로 정리했습니다.
- 다음 재생성부터 같은 문제가 반복되지 않도록 아래 스크립트에 strict JSON sanitizer를 추가했습니다.
  - `python/build_shadow_quality_risk_guard_daily_report.py`
  - `python/build_shadow_quality_risk_guard_repeatability_report.py`
  - `python/build_quality_risk_guard_promotion_report.py`

검증:

- `python -m py_compile` 통과
- Node `JSON.parse`로 관련 JSON 3개 파싱 통과

## 다음 판단

1. 지금은 production 승격보다 shadow 유지가 맞습니다.
2. 실서버에서 운영 refresh 후 위 3개 JSON이 계속 strict JSON으로 생성되는지 확인해야 합니다.
3. 최소 며칠 더 `shadow_quality_risk_guard_repeatability_report`를 누적 관찰합니다.
4. 이후 `walkforward_acceptance`가 `ACCEPTED`로 바뀌거나, 별도 guard-only acceptance가 안정적으로 통과하면 production 반영을 검토합니다.

## 운영 판단

현재 단계의 적절한 표현은 `승격 준비 후보`입니다. `매수 모델에 즉시 반영` 단계는 아직 아닙니다.
