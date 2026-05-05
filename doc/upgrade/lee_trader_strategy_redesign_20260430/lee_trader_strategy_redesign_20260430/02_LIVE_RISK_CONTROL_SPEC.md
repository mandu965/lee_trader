# 실전 자동매매 리스크 통제 명세서

작성일: 2026-04-30  
목적: 실제 서버에서 자동매매가 동작 중인 상태에서 손실 확대와 운영 사고를 막기 위한 공통 통제 체계 정의

---

## 1. 설계 원칙

실전 자동매매에서 가장 중요한 원칙은 다음이다.

```text
BUY는 적극적으로 차단해도 된다.
SELL/EXIT은 신중하게 허용해야 한다.
데이터가 불완전하면 신규 BUY는 하지 않는다.
체결 확인이 불완전하면 신규 BUY는 하지 않는다.
계좌 상태가 불명확하면 신규 BUY는 하지 않는다.
```

자동매매 사고는 대부분 “좋은 종목을 못 사서”가 아니라 다음에서 발생한다.

```text
중복 주문
잘못된 수량
잘못된 계좌
동기화 실패
급등 가격 추격
시장 급락 중 신규매수
API 장애 중 재시도
손실 제한 부재
```

---

## 2. 공통 Risk Guard 모듈 제안

### 신규 파일

```text
python/common_live_risk_guard.py
```

### 역할

AI와 RULE에서 공통으로 사용할 신규매수 차단 판단 모듈이다.

```text
입력: 주문 후보, 계좌 상태, 체결 상태, 시장 상태, 환경변수
출력: allowed 여부, block_reasons, risk_snapshot
```

### 함수 초안

```python
def evaluate_common_buy_guard(order_context: dict) -> tuple[bool, list[str], dict]:
    """
    BUY 주문 직전 공통 리스크 검증.
    SELL/EXIT 판단에는 별도 함수 사용.
    """
```

---

## 3. BUY 차단 조건

## 3.1 전역 Kill Switch

### 환경변수

```bash
GLOBAL_KILL_SWITCH=0
```

### 조건

```text
GLOBAL_KILL_SWITCH=1이면 모든 신규 BUY 차단
```

### 사유 코드

```text
global_kill_switch_on
```

---

## 3.2 일일 총 매수금액 제한

### 환경변수

```bash
GLOBAL_MAX_DAILY_BUY_AMOUNT=500000
```

### 조건

```text
오늘 이미 체결된 BUY 금액 + 신규 BUY 후보 금액 > 한도이면 차단
```

### 사유 코드

```text
daily_buy_amount_limit_exceeded
```

### 권장 초기값

| 총 운용예산 | 권장 일일 신규 BUY 한도 |
|---:|---:|
| 50만원 | 10만~20만원 |
| 100만원 | 20만~30만원 |
| 500만원 | 50만~100만원 |

초기 파일럿에서는 총 예산이 500만원이어도 하루 신규매수는 50만원 이하가 적절하다.

---

## 3.3 주간 총 매수금액 제한

### 환경변수

```bash
GLOBAL_MAX_WEEKLY_BUY_AMOUNT=1500000
```

### 조건

```text
이번 주 누적 BUY 체결 금액 + 신규 BUY 후보 금액 > 한도이면 차단
```

### 사유 코드

```text
weekly_buy_amount_limit_exceeded
```

---

## 3.4 일일 손실 제한

### 환경변수

```bash
GLOBAL_MAX_DAILY_LOSS_PCT=0.01
```

### 조건

```text
당일 실현손익 + 평가손익이 총자산 대비 -1% 이하이면 신규 BUY 차단
```

### 사유 코드

```text
daily_loss_limit_reached
```

### 주의

손실 제한은 신규 BUY 차단용이다. 무조건 SELL을 막으면 안 된다.

---

## 3.5 주간 손실 제한

### 환경변수

```bash
GLOBAL_MAX_WEEKLY_LOSS_PCT=0.03
```

### 조건

```text
주간 손익이 총자산 대비 -3% 이하이면 신규 BUY 차단
```

### 사유 코드

```text
weekly_loss_limit_reached
```

---

## 3.6 체결/잔고 동기화 최신성

### 환경변수

```bash
GLOBAL_BLOCK_BUY_ON_SYNC_STALE=1
GLOBAL_SYNC_MAX_AGE_MINUTES=30
```

### 조건

```text
마지막 보유잔고 동기화가 30분 이상 오래되면 BUY 차단
마지막 체결 동기화가 30분 이상 오래되면 BUY 차단
최근 주문 후 동기화가 한 번도 없으면 BUY 차단
```

### 사유 코드

```text
holdings_sync_stale
fills_sync_stale
post_order_sync_missing
```

---

## 3.7 동일 종목 당일 재매수 차단

### 환경변수

```bash
GLOBAL_BLOCK_SAME_SYMBOL_BUY_SAME_DAY=1
```

### 조건

```text
같은 code에 대해 당일 BUY 성공 이력이 있으면 신규 BUY 차단
```

### 사유 코드

```text
same_symbol_buy_already_filled_today
```

---

## 3.8 실시간 진입가격 괴리율 제한

### 환경변수

```bash
ENTRY_GAP_BLOCK_UP_PCT=0.03
ENTRY_GAP_HARD_BLOCK_UP_PCT=0.05
ENTRY_GAP_BLOCK_DOWN_PCT=-0.04
```

### 조건

```text
전일 종가 대비 현재가 +3% 초과: BUY 차단
전일 종가 대비 현재가 +5% 초과: hard block
전일 종가 대비 현재가 -4% 이하: 급락 위험으로 BUY 차단
현재가 조회 실패: BUY 차단
```

### 사유 코드

```text
entry_gap_up_blocked
entry_gap_up_hard_blocked
entry_gap_down_blocked
live_price_unavailable
```

---

## 3.9 시장상태 기반 차단

### 환경변수

```bash
GLOBAL_BLOCK_BUY_ON_MARKET_DEFENSIVE=1
GLOBAL_BLOCK_BUY_ON_MARKET_STATUS_MISSING=1
```

### 조건

```text
market_status가 defensive이면 신규 BUY 차단
market_status 파일 또는 payload가 없으면 신규 BUY 차단
```

### 사유 코드

```text
market_defensive_mode
market_status_missing
```

---

## 4. SELL/EXIT 통제 원칙

SELL/EXIT은 BUY와 다르게 처리해야 한다.

### 차단하면 안 되는 경우

```text
손절
강제청산
시장 방어모드 축소
리스크 축소
```

### SELL도 preview only로 돌려야 하는 경우

```text
보유수량 조회 실패
계좌 불일치
API 인증 실패
주문가능수량 조회 실패
```

### 사유 코드

```text
holding_qty_missing
account_context_unavailable
sellable_qty_unavailable
```

---

## 5. AI 적용 위치

### 1차 적용

```text
submit_live_orders.py
```

BUY 주문 후보가 만들어진 뒤 실제 제출 전 다음 필드를 계산한다.

```text
common_risk_allowed
common_risk_block_reasons
entry_price_gap_pct
entry_price_gate_status
```

### 2차 적용

```text
build_trade_intents.py
apply_execution_policy.py
```

사전 차단을 통해 프리뷰 단계에서부터 위험 후보를 줄인다.

---

## 6. RULE 적용 위치

### 1차 적용

```text
rule_account_guard.py
```

기존 `assert_order_allowed()`에 공통 guard 결과를 통합한다.

### 2차 적용

```text
rule_order_preview_builder.py
```

프리뷰 JSON에 공통 차단 사유를 명확히 표시한다.

### 3차 적용

```text
rule_order_submitter.py
```

실제 제출 직전 한 번 더 검증한다.

---

## 7. 산출물

### JSON

```text
outputs/common_live_risk_guard.json
```

예시:

```json
{
  "as_of": "2026-04-30T09:30:00+09:00",
  "global_kill_switch": false,
  "daily_buy_amount_used": 200000,
  "daily_buy_amount_limit": 500000,
  "daily_loss_pct": -0.002,
  "weekly_loss_pct": -0.006,
  "holdings_sync_age_minutes": 5,
  "fills_sync_age_minutes": 5,
  "buy_allowed": true,
  "block_reasons": []
}
```

### Markdown

```text
outputs/common_live_risk_guard_report.md
```

포함 항목:

```text
오늘 신규매수 가능 여부
차단 사유
일일/주간 매수 한도 사용률
일일/주간 손실률
동기화 최신성
시장상태
```

---

## 8. 테스트 시나리오

### 테스트 1. Kill Switch

```bash
GLOBAL_KILL_SWITCH=1
```

기대 결과:

```text
모든 BUY blocked_reason에 global_kill_switch_on 포함
SELL/EXIT은 계좌 상태가 정상일 때 유지
```

### 테스트 2. 일일 매수 한도 초과

```bash
GLOBAL_MAX_DAILY_BUY_AMOUNT=100000
```

기대 결과:

```text
BUY 후보 금액이 10만원을 넘으면 차단
```

### 테스트 3. 동기화 오래됨

마지막 sync timestamp를 1시간 전으로 설정.

기대 결과:

```text
holdings_sync_stale 또는 fills_sync_stale로 BUY 차단
```

### 테스트 4. 갭 상승

전일 종가 10,000원, 현재가 10,500원.

기대 결과:

```text
entry_gap_pct = 5.0%
entry_gap_up_hard_blocked
BUY 차단
```

---

## 9. 완료 기준

이 명세의 완료 기준은 다음이다.

```text
1. AI와 RULE 모두 공통 risk guard 결과를 주문 프리뷰에 표시한다.
2. 공통 risk guard가 BUY 차단이면 실주문 제출이 불가능하다.
3. 차단 사유가 웹/리포트에서 확인 가능하다.
4. SELL/EXIT은 신규 BUY 차단과 별도로 처리된다.
5. 테스트 시나리오 4개가 모두 통과한다.
```
