# Shadow 승격 기준 (Shadow Promotion Criteria)

> 작성일: 2026-05-07  
> 관련 과제: 2-A Shadow 승격 기준 확정  
> 관련 파일: `python/build_quality_risk_guard_live_review.py`, `python/build_walkforward_acceptance.py`, `python/production_config.py`

---

## 개요

"Shadow 승격"은 `quality_risk_guard` 로직을 shadow 관측 모드에서 **운영 모드**로 전환하는 것을 의미합니다.  
승격이 이루어지면 guard가 실제 BUY 후보 필터링에 직접 반영됩니다.

**원칙**: 두 가지 기준(Quality Guard Live Review + Walk-forward Acceptance)이 모두 충족될 때만 승격을 검토합니다.  
현재(`2026-05-07`) 기준 **두 조건 모두 미충족** — 승격 보류 상태입니다.

---

## 기준 1 — Quality Guard Live Review

소스: `python/build_quality_risk_guard_live_review.py` → `outputs/quality_risk_guard_live_review.json`

### Sample Status 분류 (`_sample_status()`)

| observed_count | status | 의미 |
|---|---|---|
| < 30 | `INSUFFICIENT_SAMPLE` | 관측 부족, 통계적 신뢰 불가 |
| 30 ~ 99 | `MONITOR_ONLY` | 참고용, 승격 아직 불가 |
| ≥ 100 | `ACTIONABLE` | 통계적으로 유효, 승격 검토 가능 |

### 승격 상태 분류 (`_promotion_status()`)

| promotion_status | 조건 | 의미 |
|---|---|---|
| `KEEP_SHADOW` | 하나 이상의 blocker 존재 | 승격 불가 |
| `REVIEW_READY` | blocker 없음 AND D+5 observed < 100 | 검토 가능, 승격은 아직 |
| `PROMOTE_CANDIDATE` | blocker 없음 AND D+5 observed ≥ 100 | 승격 후보 |

### Promotion Blockers 상세 (모두 해소돼야 KEEP_SHADOW 탈출)

| Blocker | 해소 조건 |
|---|---|
| D+5 observed_count < 30 | D+5 horizon 관측 건수 ≥ 30 |
| Guard-applied observed_count < 30 | guard 적용 종목 관측 ≥ 30 |
| Guard-not-applied observed_count < 30 | guard 미적용 종목 관측 ≥ 30 |
| Closed-trade observed_count < 30 | 청산 거래 관측 ≥ 30 |
| Closed-trade PnL snapshot fallback | 모든 청산 거래의 매수 단가가 실체결가로 매칭 완료 |

---

## 기준 2 — Walk-forward Acceptance

소스: `python/build_walkforward_acceptance.py` → `outputs/walkforward_acceptance.json`

### 승격을 위해 필요한 최소 조건 (ACCEPTED 또는 CONDITIONAL)

| 항목 | 기준 | 변수명 |
|---|---|---|
| 초과 수익 | top20 excess_return > 0 | `performance_ok` |
| 승률 | top20 hit_rate ≥ 0.55 | `performance_ok` |
| 순서 정렬 | avg_return: top20 > top50 > universe | `ordering_ok` |
| 최대낙폭 | top20 avg_mdd ≥ −0.25 | `risk_ok` |
| Confidence 단조성 | 5d hit_rate 단조 증가 AND stable bucket ≥ 2 | `confidence_ok` |
| 실체결 증거 | fill_ratio ≥ 0.60 AND median_abs_slippage ≤ 0.03 | `execution_ok` |

### 승격 단계와 요구 조건

| walkforward status | 조건 | 승격 가능 여부 |
|---|---|---|
| `ACCEPTED` | 6개 항목 모두 충족 | 승격 검토 가능 |
| `CONDITIONAL` | performance_ok AND risk_ok | 추가 검증 필요 (승격 보류 권장) |
| `REJECTED` | performance_ok 또는 risk_ok 미충족 | 승격 불가 |

---

## 현재 상태 (as of 2026-05-06)

### Quality Guard Live Review

```
sample_status:       INSUFFICIENT_SAMPLE
promotion_status:    KEEP_SHADOW
```

| Horizon | count | observed_count | sample_status |
|---|---|---|---|
| D+0 | 25 | 24 | INSUFFICIENT_SAMPLE |
| D+1 | 25 | 23 | INSUFFICIENT_SAMPLE |
| D+3 | 25 | 18 | INSUFFICIENT_SAMPLE |
| D+5 | 25 | 7 | INSUFFICIENT_SAMPLE |

**Promotion Blockers 현황 (5개 모두 존재)**:
- D+5 observed_count = 7 (기준: ≥ 30)
- Guard-applied observed_count = 6 (기준: ≥ 30)
- Guard-not-applied observed_count = 1 (기준: ≥ 30)
- Closed-trade observed_count = 15 (기준: ≥ 30)
- Closed-trade snapshot fallback count = 7 (기준: 0)

### Walk-forward Acceptance

```
status:  REJECTED
```

| 항목 | 현재 값 | 기준 | 결과 |
|---|---|---|---|
| top20 excess_return | +6.38% | > 0% | ✓ |
| top20 hit_rate | 0.90 | ≥ 0.55 | ✓ |
| ordering (top20 > top50 > universe) | top50(36.7%) > top20(34.1%) | top20 최상위 | ✗ |
| top20 avg_mdd | −27.11% | ≥ −25% | ✗ |
| confidence monotonicity | False | True | ✗ |
| execution evidence | unavailable | — | ✓ (N/A) |

---

## 승격 절차 (Promotion Checklist)

승격을 실행할 준비가 됐을 때 따라야 할 절차입니다. **현재 실행하지 않습니다.**

1. `build_quality_risk_guard_live_review.py` 실행 → `promotion_status == "PROMOTE_CANDIDATE"` 확인
2. `build_walkforward_acceptance.py` 실행 → `status == "ACCEPTED"` 확인
3. Shadow 비교 실행 (ranking 결과가 예상대로 바뀌는지 검증)
4. `SCORE_FORMULA_VERSION` 환경변수 확인 (아래 feature flag 섹션 참고)
5. Paper trading 3일 이상 검증
6. `config/production_v1.yaml` 갱신 또는 운영자 승인 후 적용

---

## SCORE_FORMULA_VERSION Feature Flag

점수 수식 버전 전환을 위한 인프라가 준비되어 있습니다. **현재 비활성화 상태입니다.**

### 관련 파일

| 파일 | 역할 |
|---|---|
| `python/production_config.py` | `get_score_formula_version()` — 버전 문자열 단일 소스 |
| `python/ranking_builder.py` | `resolve_score_formula_version()` — 운영 모드 가드 포함 |
| `config/production_v1.yaml` | `metadata.score_formula_version` — 운영 기준값 |
| `.env` | `SCORE_FORMULA_VERSION` — 환경변수 오버라이드 (비활성화) |

### 현재 운영 기준 버전

```
ranking_builder_v8_return_prob_tech_regime
```

### 전환 가드 (안전 장치)

`ranking_builder.resolve_score_formula_version()`은 **운영 모드(`LEE_TRADER_RUNTIME_MODE=operational`)에서 환경변수를 무시**합니다.  
환경변수로 수식 버전을 전환하려면:
1. `.env`에 `SCORE_FORMULA_VERSION=<new_version>` 설정
2. `ranking_builder.py`의 `resolve_score_formula_version()` 함수에서 운영 모드 가드를 명시적으로 해제
3. Shadow 비교 실행 후 결과 검증

이 두 단계를 모두 거쳐야만 운영 환경에서 공식이 바뀌므로 단일 설정 변경만으로 실수로 전환되는 것을 방지합니다.

---

## 연관 문서

- [Quality Guard Live Review 예시](../outputs/quality_risk_guard_live_review.json)
- [Walk-forward Acceptance 결과](../outputs/walkforward_acceptance.json)
- [Score Formula Version 명명 규칙](python/score_formula_version.md)
- [운영 정렬 기준](modules/Lee_trader_score/RUNTIME_SORTING.md)
