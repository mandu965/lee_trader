# AI 기반 자동매매 전략 재정의: Confidence와 진입가격 통제

작성일: 2026-04-30  
대상: AI 기반 자동매매 엔진

---

## 1. 현재 AI 전략의 핵심 문제

현재 AI 기반 자동매매는 점수 산정, gate, execution policy, order intent, submit 구조를 가지고 있다. 구조 자체는 좋다.

하지만 실전 수익 관점에서는 다음 두 문제가 가장 크다.

```text
1. confidence가 실제 수익 신뢰도로 충분히 검증되지 않았다.
2. 전일 종가 기준 점수가 실제 매수 시점 가격을 통제하지 못한다.
```

따라서 AI 전략의 다음 단계는 모델 복잡도 증가가 아니라 다음이다.

```text
raw score → 실전 신뢰도 보정 → 진입가격 검증 → 소액/표준/차단 결정
```

---

## 2. AI 자동매수 운영 원칙

### 원칙 1. final_score는 매수 후보 선정용이다

`final_score`가 높다는 것은 “분석 기준상 좋은 후보”라는 뜻이지, “지금 가격에 사도 된다”는 뜻이 아니다.

운영 해석:

```text
final_score 상위권: 관심 후보
buy_gate 통과: 매수 검토 가능
entry_price_gate 통과: 실제 진입 가능
risk_guard 통과: 주문 제출 가능
```

### 원칙 2. confidence는 등급화 후 사용한다

기존 confidence는 그대로 실전 비중에 사용하지 않는다.

분리 구조:

```text
raw_confidence_score
calibrated_confidence_score
live_confidence_grade
execution_weight_scale
```

### 원칙 3. 진입가격 괴리율은 독립 게이트로 둔다

AI 점수가 높아도 다음 조건이면 매수하지 않는다.

```text
전일 종가 대비 현재가 과도한 상승
전일 종가 대비 현재가 과도한 하락
현재가 조회 실패
호가 스프레드 과도
장 초반 변동성 과도
```

---

## 3. live_confidence_grade 정의

## 3.1 등급 체계

| 등급 | 의미 | 주문 정책 |
|---|---|---|
| A | 실전/검증 성과가 충분히 우수 | 표준 비중 허용 |
| B | 양호하지만 표본 또는 안정성 일부 부족 | 50% 비중 |
| C | 표본 부족 또는 불확실 | preview 또는 20% 이하 소액 |
| D | 성과 부진/위험 | BUY 차단 |

---

## 3.2 등급 산정 입력값

```text
raw_confidence_score
calibrated_confidence_score
confidence_bucket
bucket_sample_count
bucket_hit_rate
bucket_avg_return
bucket_excess_return
recent_10_trade_return
recent_10_trade_hit_rate
max_drawdown_by_bucket
market_regime
```

---

## 3.3 등급 산정 규칙 초안

### A등급

```text
bucket_sample_count >= 30
bucket_hit_rate >= 55%
bucket_avg_return > 0
bucket_excess_return > 0
recent_10_trade_return >= 0
max_drawdown_by_bucket within limit
```

### B등급

```text
bucket_sample_count >= 20
bucket_avg_return >= 0
bucket_excess_return >= -0.5%
recent_10_trade_return >= -1%
```

### C등급

```text
bucket_sample_count < 20
또는 성과가 애매한 구간
또는 최근 10건 성과가 불안정
```

### D등급

```text
bucket_excess_return < -1%
또는 recent_10_trade_return < -2%
또는 max_drawdown limit 초과
```

---

## 4. 주문 비중 정책

| live_confidence_grade | 신규매수 정책 | weight scale | position cap scale |
|---|---|---:|---:|
| A | 표준 BUY 허용 | 1.00 | 1.00 |
| B | 축소 BUY 허용 | 0.50 | 0.50 |
| C | 파일럿/관찰 | 0.20 | 0.30 |
| D | BUY 차단 | 0.00 | 0.00 |

초기 실전 표본이 부족한 현재 상태에서는 대부분 C 이하로 시작하는 것이 맞다.

---

## 5. 진입가격 게이트

## 5.1 신규 필드

AI 주문 프리뷰에 다음 필드를 추가한다.

```text
score_close_price
previous_close
live_price
live_price_source
entry_price_gap_pct
entry_price_gate_status
entry_price_gate_reason
```

---

## 5.2 게이트 기준

| 조건 | 판단 | 사유 코드 |
|---|---|---|
| 현재가 조회 실패 | BUY 차단 | live_price_unavailable |
| 전일 종가 대비 +3% 초과 | BUY 차단 | entry_gap_up_blocked |
| 전일 종가 대비 +5% 초과 | hard block | entry_gap_up_hard_blocked |
| 전일 종가 대비 -4% 이하 | BUY 차단 | entry_gap_down_blocked |
| 괴리율 -4%~+3% | 통과 | entry_gap_ok |

---

## 5.3 왜 갭 하락도 차단하는가

하락은 싸게 살 기회일 수도 있지만, 자동매매에서는 다음 위험이 있다.

```text
악재 발생
전일 데이터가 이미 무효화
유동성 악화
장 초반 투매
모델 입력과 현재 시장 상황 불일치
```

따라서 급락은 “싸다”가 아니라 “모델 신호 무효화 가능성”으로 봐야 한다.

---

## 6. AI 자동매수 상태 정의

기존 BUY/HOLD/BLOCK 외에 다음 상태를 명확히 한다.

```text
WATCH_ONLY: 점수는 좋지만 실전 조건 부족
PILOT_BUY: 소액 파일럿만 허용
STANDARD_BUY: 표준 비중 허용
BLOCKED_BY_CONFIDENCE: 신뢰도 부족
BLOCKED_BY_ENTRY_PRICE: 진입가격 부적합
BLOCKED_BY_RISK_GUARD: 공통 리스크 차단
```

---

## 7. 적용 대상 파일

### 1차 수정

```text
python/submit_live_orders.py
```

수정 내용:

```text
BUY preview 생성 시 entry_price_gate 계산
common_live_risk_guard 결과 반영
blocked_reason 확장
order_requests_preview.md에 차단 사유 표시
```

### 2차 수정

```text
python/apply_execution_policy.py
```

수정 내용:

```text
live_confidence_grade별 target_weight scale 적용
D등급 BUY 차단
C등급 watch 또는 pilot 제한
```

### 3차 수정

```text
python/build_confidence_calibration_report.py
python/calibrate_operational_confidence.py
```

수정 내용:

```text
bucket별 실제 실전 성과 반영
표본 부족 구간 grade 강등
recent N trade 성과 반영
```

---

## 8. AI 리포트 추가 항목

일일 리포트에 다음을 추가한다.

```text
AI 후보 수
AI BUY 허용 수
AI BUY 차단 수
confidence grade 분포
entry price gate 차단 종목
상위 점수였지만 미매수한 종목과 사유
실제 체결가 기준 수익률
전일 종가 대비 체결가 괴리율
```

---

## 9. 운영 기준

현재 표본 부족 단계의 AI 운영 기준:

```text
A등급: 아직 거의 나오지 않는 것이 정상
B등급: 소액 허용 가능
C등급: 기본은 관찰, 필요 시 극소액
D등급: 매수 금지
```

권장 설정:

```text
AUTO_TRADE_BUY_APPROVAL_REQUIRED=1
AI 신규 BUY 일일 최대 1~2건
AI 일일 신규 BUY 금액 총합 제한
entry gap 통과 필수
common risk guard 통과 필수
```

---

## 10. 완료 기준

```text
1. AI 주문 프리뷰에 confidence grade와 entry price gate가 표시된다.
2. confidence 표본 부족 구간은 자동으로 C 이하가 된다.
3. 전일 종가 대비 +3% 초과 종목은 자동매수되지 않는다.
4. 차단된 종목의 사유가 리포트에 남는다.
5. 실전 체결 결과가 confidence grade별로 집계된다.
```
