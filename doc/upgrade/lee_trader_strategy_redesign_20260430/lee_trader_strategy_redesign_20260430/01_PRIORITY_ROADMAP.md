# Lee Trader 우선순위별 개선 로드맵

작성일: 2026-04-30  
목적: 실제 자동매매 운영 중인 시스템을 안전하게 개선하기 위한 실행 순서 정의

---

## 전체 로드맵 요약

| 우선순위 | 기간 | 핵심 목표 | 주요 산출물 |
|---|---:|---|---|
| P0 | 즉시~3일 | 손실/운영 사고 차단 | 공통 Kill Guard, 진입가격 게이트, 체결동기화 차단 |
| P1 | 1주 | 실전 리뷰 데이터 강화 | live trade ledger 확장, 주문 사유/성과 기록 |
| P2 | 2주 | confidence 재정의 | live confidence grade, calibration report |
| P3 | 2~3주 | RULE 실전 파일럿 안정화 | 청산/보유/손절 룰 명문화 및 구현 |
| P4 | 3~4주 | AI/RULE 통합 노출관리 | Master Risk Manager |
| P5 | 1개월+ | 수익성 검증 고도화 | 체결가 기준 walk-forward/live evaluation |

---

# P0. 실전 안전장치 강화

## P0-1. 공통 신규매수 차단 조건 추가

### 문제

현재 AI와 RULE에는 각각 안전장치가 있지만, 전체 계좌 기준의 공통 차단 조건이 부족하다.

### 적용 대상

- AI: `submit_live_orders.py` 또는 그 직전 preview 생성 단계
- RULE: `rule_account_guard.py`, `rule_order_preview_builder.py`, `rule_order_submitter.py`
- 공통 신규 모듈 후보: `python/common_live_risk_guard.py`

### 추가할 차단 조건

```text
1. 전일 또는 당일 체결 동기화 실패 시 BUY 차단
2. 보유잔고 조회 실패 시 BUY 차단
3. 현금/주문가능금액 조회 실패 시 BUY 차단
4. 시장 데이터 최신성 기준 초과 시 BUY 차단
5. 같은 종목 당일 BUY 성공 이력이 있으면 재BUY 차단
6. 일일 총 BUY 금액 한도 초과 시 BUY 차단
7. 주간 손실 한도 초과 시 BUY 차단
8. 수동 kill switch 활성화 시 BUY 차단
```

### 권장 환경변수

```bash
GLOBAL_KILL_SWITCH=0
GLOBAL_MAX_DAILY_BUY_AMOUNT=500000
GLOBAL_MAX_WEEKLY_BUY_AMOUNT=1500000
GLOBAL_MAX_DAILY_LOSS_PCT=0.01
GLOBAL_MAX_WEEKLY_LOSS_PCT=0.03
GLOBAL_BLOCK_BUY_ON_SYNC_STALE=1
GLOBAL_SYNC_MAX_AGE_MINUTES=30
GLOBAL_BLOCK_SAME_SYMBOL_BUY_SAME_DAY=1
```

### 완료 기준

```text
공통 guard가 False를 반환하면 AI/RULE 어느 쪽에서도 BUY 주문이 제출되지 않는다.
SELL/EXIT은 원칙적으로 차단하지 않는다. 단, 계좌 조회 실패 시에는 SELL도 preview only 처리한다.
```

---

## P0-2. AI 실시간 진입가격 게이트 추가

### 문제

AI 점수는 전일 종가 기준 추천에 가깝다. 실제 매수 시점의 가격이 급등하면 기대값이 훼손된다.

### 적용 위치

우선순위 1안:

```text
submit_live_orders.py의 BUY preview 생성 시점
```

우선순위 2안:

```text
build_trade_intents.py 또는 apply_execution_policy.py에서 사전 차단
```

실전에서는 주문 직전 가격이 가장 중요하므로 1안이 우선이다.

### 차단 기준 초안

```text
전일 종가 대비 현재가 +3% 초과: 신규 BUY 차단
전일 종가 대비 현재가 +5% 초과: 당일 관찰만
전일 종가 대비 현재가 -4% 이하: 급락 위험으로 신규 BUY 차단
호가/현재가 조회 실패: BUY 차단
현재가가 없고 종가만 있으면 preview only
```

### 신규 필드

주문 프리뷰에 다음 필드를 추가한다.

```text
previous_close
live_price
entry_price_gap_pct
entry_price_gate_status
entry_price_gate_reason
```

### 완료 기준

```text
전일 종가 대비 현재가 괴리율 때문에 차단된 종목이 order_requests_preview.json 또는 report md에 명확히 표시된다.
```

---

## P0-3. 스케줄러/실행 상태 감시 강화

### 문제

컨테이너가 떠 있어도 특정 스케줄이 실패하거나 실행 누락될 수 있다.

### 대상 스케줄러

```text
scheduler
scheduler-recovery
scheduler-auto-buy
scheduler-live-account-sync
scheduler-rule-after-close
scheduler-rule-before-open
scheduler-rule-after-open
```

### 추가할 상태 항목

```text
last_started_at
last_finished_at
last_success_at
last_failed_at
last_error
last_step_name
last_step_status
next_run_at
today_run_completed
buy_submission_attempted
buy_submission_succeeded
sync_completed_after_order
```

### 완료 기준

```text
웹 UI 또는 outputs JSON만 봐도 오늘 어떤 스케줄이 성공/실패/미실행인지 판단 가능하다.
```

---

# P1. 실전 리뷰 데이터 강화

## P1-1. live trade ledger 필드 확장

### 반드시 저장할 필드

```text
trade_id
order_id
request_id
engine_type
strategy_id
run_mode
code
name
side
qty
order_price
filled_price
filled_amount
order_time
filled_time
source_score_date
final_score
prob_score
ret_score
tech_score
quality_score
confidence_score
calibrated_confidence
live_confidence_grade
liquidity_score
market_regime
entry_price_gap_pct
entry_gate_status
entry_gate_reason
buy_reason
sell_reason
portfolio_action_reason
benchmark_name
benchmark_return_until_exit
strategy_return
excess_return
holding_days
exit_reason
review_status
```

### 완료 기준

```text
한 건의 거래만 봐도 “왜 샀는지, 어떤 조건에서 샀는지, 얼마에 체결됐는지, 결과가 어땠는지” 설명 가능해야 한다.
```

---

## P1-2. 일일 리포트에 추가할 섹션

`build_live_kpi_daily_report.py`, `build_live_trade_review.py`, `build_live_trade_review_summary.py`에 다음 섹션을 추가한다.

```text
1. 오늘 BUY 후보 수
2. BUY 차단 수와 사유별 건수
3. 실제 제출 주문 수
4. 실제 체결 주문 수
5. 체결 실패/미체결 수
6. 평균 entry_price_gap_pct
7. AI/RULE별 실현/미실현 손익
8. benchmark 대비 초과수익
9. 신규매수 차단 조건 발동 여부
10. 데이터 최신성/동기화 상태
```

---

# P2. confidence calibration 재구성

## 목표

confidence를 “모델이 자신 있어 하는 점수”가 아니라 “실제 운영에서 믿을 수 있는 등급”으로 바꾼다.

## 등급 정의 초안

| 등급 | 조건 | 주문 비중 |
|---|---|---:|
| A | 표본 충분, 승률/초과수익 모두 양호 | 표준 비중 100% |
| B | 표본 일부 충분, 성과 보통 이상 | 표준 비중 50% |
| C | 표본 부족 또는 성과 불명확 | 관찰/소액 20% 이하 |
| D | 성과 음수 또는 불안정 | BUY 금지 |

## 표본 부족 처리

```text
confidence bucket별 실전 체결 표본 < 20건: 최대 C등급
최근 10건 손익 음수: 최대 C등급
benchmark 대비 초과수익 음수: 최대 C등급
MDD 기준 초과: D등급
```

---

# P3. RULE 전략 고도화

## 목표

RULE은 먼저 실전 파일럿 엔진으로 안정화한다.

## 추가할 청산/축소 규칙

```text
1. rule_score_v2 < 35: EXIT 유지
2. defensive mode + rule_score_v2 < 45: REDUCE 유지
3. 최대 보유일 초과: REDUCE 또는 EXIT
4. 매수가 대비 -5%: 손절 검토/EXIT
5. 고점 대비 -4%: trailing stop 후보
6. 20일선 이탈 + 거래량 감소: EXIT 후보
7. 시장 방어모드 전환: 신규 BUY 금지, 보유 종목 점검
```

## 추가할 리포트

```text
RULE 매수 사유별 성과
RULE 청산 사유별 성과
RULE 보유일별 수익률
RULE gap_risk 차단 후 실제 성과
RULE strong_entry 실제 승률
```

---

# P4. Master Risk Manager

## 목표

AI와 RULE이 별도로 주문을 만들어도 최종 주문은 하나의 공통 리스크 관리자를 통과하게 한다.

## 입력

```text
AI order preview
RULE order preview
live account holdings
live cash/buying power
today filled orders
market status
daily/weekly PnL
```

## 출력

```text
approved_orders.json
blocked_orders.json
risk_summary.json
```

## 승인 기준

```text
1. engine별 예산 한도 통과
2. 전체 일일 매수 한도 통과
3. 동일 종목 중복 없음
4. 섹터/테마 총 노출 한도 통과
5. 현금 비중 하한 유지
6. 일/주간 손실 제한 미도달
7. 체결/잔고 동기화 최신
8. 신규 매수 가격 괴리율 통과
```

---

# P5. 수익성 검증 강화

## 목표

전략을 믿을지 말지를 실제 숫자로 판단한다.

## 핵심 지표

```text
총 거래 수
승률
평균 수익률
중앙값 수익률
평균 손익비
최대낙폭
profit factor
benchmark 대비 초과수익
시장 국면별 성과
AI/RULE별 성과
confidence grade별 성과
entry_price_gap 구간별 성과
```

## 운영 확대 기준

```text
최소 실전 체결 20건 이상
동기화/리포트 누락 0건
일일 손실 제한 작동 확인
AI/RULE별 손익 분리 가능
benchmark 대비 초과수익 양수
MDD 허용범위 내
```
