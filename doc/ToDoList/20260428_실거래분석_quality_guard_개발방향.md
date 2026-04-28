# 2026-04-28 실거래 분석 + quality_risk_guard 개발 방향

## 1. 목적

이 문서는 아래 두 문서를 실제 시스템 개발 방향으로 연결하기 위한 실행 문서다.

- `doc/ToDoList/20260428_quality_risk_guard_연구메모.md`
- `doc/ToDoList/20260428_분석인프라준비.md`

핵심 목표는 `quality_risk_guard`를 즉시 production 랭킹에 반영하는 것이 아니다.

목표는 다음과 같다.

```text
실거래 데이터 기반 분석 인프라를 만들고,
그 분석 인프라로 quality_risk_guard의 production 승격 여부를 판단한다.
```

## 2. 현재 판단

### 2.1 quality_risk_guard 상태

`quality_risk_guard`는 현재 production 점수 체계를 전면 변경하는 기능이 아니라, 기존 점수 위에 소폭 감점을 얹는 shadow safety overlay다.

현재 공식:

- `qual_score < 20`이면 6점 감점
- `risk_penalty >= 12`이면 4점 감점
- `shadow_final_score_quality_risk_guard = final_score - shadow_quality_risk_guard_penalty`

현재 판단:

- 연구 산출물 기준으로는 승격 검토 후보가 될 수 있다.
- 그러나 `walkforward_acceptance`가 아직 `REJECTED`다.
- 실거래 체결 기반 evidence가 아직 충분하지 않다.
- 따라서 production 승격이 아니라 shadow 유지가 맞다.

### 2.2 분석 인프라 상태

운영 서버 기준으로는 다음 체인이 형성되어 있다.

```text
매매 판단
→ 주문 요청
→ 브로커 제출
→ 실제 체결
→ 계좌 스냅샷
→ 사후성과 리뷰
```

따라서 다음 개발의 중심은 체결 데이터를 신뢰 가능한 성과 데이터로 변환하는 것이다.

## 3. 개발 원칙

1. production 랭킹을 즉시 바꾸지 않는다.
2. `quality_risk_guard`는 shadow 상태로 유지한다.
3. 운영 원천 테이블은 변경하지 않는다.
4. 분석은 `analytics` schema/view 또는 별도 리포트 계층에서 수행한다.
5. 표본이 부족하면 매매 파라미터를 바꾸지 않는다.
6. D0 성과만으로 판단하지 않는다.
7. 판단 품질 KPI와 실제 실현손익 KPI를 분리한다.
8. JSON 산출물은 strict JSON으로 생성한다. `NaN`, `Infinity`는 허용하지 않는다.

## 4. 개발 범위

### 4.1 1차 개발 범위

우선순위는 아래 순서다.

```text
analytics.live_trade_fact
analytics.live_review_kpi
analytics.live_score_bucket_kpi
analytics.live_quality_guard_kpi
outputs/live_kpi_daily_report.json
outputs/live_kpi_daily_report.md
outputs/quality_risk_guard_live_review.json
outputs/quality_risk_guard_live_review.md
```

### 4.2 2차 개발 범위

1차가 안정된 뒤 진행한다.

```text
analytics.live_daily_account_nav
analytics.live_closed_trade
outputs/live_closed_trade_report.json
outputs/live_closed_trade_report.md
자동매매 화면 KPI 섹션 확장
별도 KPI 앱 분리 검토
```

## 5. analytics view 설계

### 5.1 analytics.live_trade_fact

목적:

체결 row를 주문 판단, 점수, 랭킹, guard shadow 정보와 결합한 기본 fact view다.

원천:

- `research.live_order_fill`
- `research.live_order_request`
- `research.live_order_execution`
- `research.live_trade_decision`

필수 컬럼:

```text
request_id
intent_id
as_of_date
filled_at
code
name
side
intent_type
filled_qty
filled_price
filled_amount
fee
tax
ranking_run_id
ranking_rank
final_score
confidence_score
risk_penalty
ret_score
prob_score
qual_score
tech_score
liquidity_score
safety_score
dominant_theme
score_driver_1
score_driver_2
score_driver_3
risk_factor_1
risk_factor_2
gate_status
source_action
submission_status
broker_order_id
```

quality guard 검증을 위해 추가해야 할 컬럼:

```text
shadow_quality_risk_guard_applied
shadow_quality_risk_guard_penalty
shadow_final_score_quality_risk_guard
shadow_rank_quality_risk_guard
production_rank
shadow_rank_delta
guard_applied_bucket
guard_penalty_bucket
```

주의:

- 위 shadow 컬럼이 live order/request에 없으면 ranking snapshot 또는 ranking_history와 조인해 보강한다.
- 운영 테이블에 직접 컬럼을 늘리는 대신 analytics view에서 계산/조인하는 방식을 우선한다.

### 5.2 analytics.live_review_kpi

목적:

D0/D+1/D+3/D+5 사후성과를 intent, rank, score, confidence 기준으로 요약한다.

필수 group:

```text
horizon
intent_type
rank_bucket
final_score_bucket
confidence_bucket
risk_penalty_bucket
dominant_theme
market_gate
sample_status
```

필수 metric:

```text
count
observed_count
win_rate
avg_return
weighted_avg_return
avg_win
avg_loss
payoff_ratio
expectancy
```

### 5.3 analytics.live_score_bucket_kpi

목적:

점수 체계가 실제 성과와 단조 관계를 가지는지 확인한다.

필수 bucket:

```text
final_score_bucket
confidence_bucket
risk_penalty_bucket
rank_bucket
```

질문:

- 점수가 높을수록 평균 성과가 좋은가
- confidence가 높을수록 손실률이 낮은가
- risk_penalty가 높은 그룹은 실제로 위험했는가
- rank_1_3이 rank_9_20보다 좋은가

### 5.4 analytics.live_quality_guard_kpi

목적:

`quality_risk_guard`를 production에 승격해도 되는지 실거래 기반으로 판단한다.

필수 group:

```text
horizon
shadow_quality_risk_guard_applied
guard_penalty_bucket
shadow_rank_delta_bucket
production_rank_bucket
shadow_rank_bucket
sample_status
```

필수 metric:

```text
count
observed_count
win_rate
avg_return
weighted_avg_return
expectancy
avg_downside_return
max_loss
```

핵심 비교:

```text
guard_applied vs guard_not_applied
production_top20 vs shadow_top20
shadow_rank_up vs shadow_rank_down
penalty_0 vs penalty_4 vs penalty_6 vs penalty_10
```

## 6. 리포트 설계

### 6.1 live_kpi_daily_report

목적:

매일 운영 상태와 성과 품질을 한 번에 확인한다.

산출물:

```text
outputs/live_kpi_daily_report.json
outputs/live_kpi_daily_report.md
```

포함 내용:

- 기준일
- 총자산, 현금, 현금비중
- 오늘 판단/요청/제출/체결 수
- 체결 누락, 계좌 반영 누락, 주문 실패 사유
- D0/D+1/D+3/D+5 리뷰 요약
- intent별 성과
- rank bucket별 성과
- score bucket별 성과
- confidence bucket별 성과
- risk penalty bucket별 성과
- sample_status
- 운영 주의사항

### 6.2 quality_risk_guard_live_review

목적:

`quality_risk_guard`의 shadow 성과를 실거래 기준으로 검토한다.

산출물:

```text
outputs/quality_risk_guard_live_review.json
outputs/quality_risk_guard_live_review.md
```

포함 내용:

- guard 적용/미적용 그룹별 성과
- guard penalty bucket별 성과
- shadow rank 상승/하락 그룹별 성과
- production top20 vs shadow top20 비교
- D0/D+1/D+3/D+5 horizon별 성과
- sample_status
- promotion_status
- promotion 차단 사유

promotion_status 후보:

| 상태 | 의미 |
| --- | --- |
| `KEEP_SHADOW` | shadow 유지 |
| `REVIEW_READY` | 검토 가능하지만 production 반영 전 |
| `PROMOTE_CANDIDATE` | production 반영 후보 |
| `REJECT` | guard 방향 재검토 |

## 7. 표본 신뢰도 기준

기본 기준:

| 상태 | 기준 | 의미 |
| --- | --- | --- |
| `INSUFFICIENT_SAMPLE` | `observed_count < 30` | 참고만 가능 |
| `MONITOR_ONLY` | `30 <= observed_count < 100` | 감시 가능, 파라미터 변경 금지 |
| `ACTIONABLE` | `observed_count >= 100` | 운영 판단 후보 |
| `DEGRADE_CONFIRMED` | 충분한 표본에서 기대값 악화 반복 | 룰 변경 검토 가능 |

quality guard 승격 검토 최소 조건:

- D+5 관찰 30건 이상
- guard 적용 그룹 관찰 30건 이상
- guard 미적용 그룹 관찰 30건 이상
- production top20과 shadow top20 비교 가능
- 최근 산출물 JSON strict parse 통과
- walkforward acceptance 또는 guard-only acceptance가 개선 방향

## 8. promotion 판단 기준

`quality_risk_guard` promotion은 단순 평균수익률 하나로 결정하지 않는다.

승격 후보 조건:

1. D+5 expectancy가 production baseline보다 악화되지 않는다.
2. guard 적용군의 downside가 guard 미적용군보다 크다는 증거가 있다.
3. shadow top20이 production top20보다 평균 성과 또는 downside 측면에서 개선된다.
4. top20 > top50 > universe ordering이 유지된다.
5. 저품질/고위험 종목 노출이 감소한다.
6. 표본 상태가 최소 `MONITOR_ONLY` 이상이다.
7. JSON/report 생성이 안정적이다.

승격 금지 조건:

- D0만 좋고 D+3/D+5가 나쁘다.
- observed_count가 기준 미달이다.
- guard 적용군과 미적용군 표본이 불균형하다.
- 실거래 성과와 walkforward evidence가 충돌한다.
- 산출물 생성이 불안정하다.

## 9. 개발 순서

### 1단계: fact view

- `analytics` schema 생성
- `analytics.live_trade_fact` view 작성
- request/fill/order/ranking 연결 확인
- guard shadow 컬럼 포함 여부 확인

### 2단계: review KPI view

- `analytics.live_review_kpi`
- `analytics.live_score_bucket_kpi`
- sample_status 계산

### 3단계: quality guard KPI view

- `analytics.live_quality_guard_kpi`
- production vs shadow 비교
- penalty bucket 비교

### 4단계: 리포트 생성

- `python/build_live_kpi_daily_report.py`
- `python/build_quality_risk_guard_live_review.py`
- strict JSON sanitizer 공통화

### 5단계: 화면 반영

자동매매 화면에는 요약만 추가한다.

- Live KPI 요약
- Score bucket KPI
- Quality guard shadow review
- sample_status
- promotion_status

### 6단계: Closed Trade

리뷰 KPI가 안정된 뒤 실현손익 계산을 추가한다.

- 평균단가 방식
- 부분청산 처리
- TRIM/EXIT 분리
- realized/unrealized 구분

## 10. 하지 말아야 할 것

- `quality_risk_guard`를 즉시 production 점수에 반영하지 않는다.
- 표본 부족 상태에서 score cutoff를 바꾸지 않는다.
- D0 성과만 보고 매수 룰을 바꾸지 않는다.
- 운영 원천 테이블에 분석 편의 컬럼을 무리하게 추가하지 않는다.
- strict JSON이 아닌 산출물을 운영 화면에서 읽지 않는다.
- raw JSON을 중복 저장하는 물리 테이블을 먼저 만들지 않는다.

## 11. 체크리스트

- [ ] `analytics.live_trade_fact` 설계
- [ ] guard shadow 컬럼 조인 가능 여부 확인
- [ ] `analytics.live_review_kpi` 설계
- [ ] `analytics.live_score_bucket_kpi` 설계
- [ ] `analytics.live_quality_guard_kpi` 설계
- [ ] sample_status 계산 규칙 구현
- [ ] `outputs/live_kpi_daily_report.json` 생성
- [ ] `outputs/live_kpi_daily_report.md` 생성
- [ ] `outputs/quality_risk_guard_live_review.json` 생성
- [ ] `outputs/quality_risk_guard_live_review.md` 생성
- [ ] strict JSON sanitizer 공통화
- [ ] 자동매매 화면 요약 섹션 설계
- [ ] promotion_status 기준 구현
- [ ] Closed Trade 설계는 2차로 분리

## 12. 최종 방향

다음 개발 목표는 다음 한 문장으로 정리한다.

```text
실거래 데이터 기반 analytics/KPI 계층을 만들어 quality_risk_guard를 production에 넣을지 판단할 수 있게 한다.
```

이 개발은 매매 로직 변경이 아니라 측정 체계 구축이다. 측정 체계가 안정되고 표본 기준을 충족하기 전까지는 `quality_risk_guard`를 shadow 후보로 유지한다.
