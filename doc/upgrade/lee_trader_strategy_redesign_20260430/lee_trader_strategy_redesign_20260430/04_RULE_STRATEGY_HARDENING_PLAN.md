# RULE 기반 자동매매 전략 고도화 계획

작성일: 2026-04-30  
대상: `RULE_TREND_LIQUIDITY_V1`

---

## 1. 현재 RULE 전략 평가

현재 RULE 기반 자동매매는 AI보다 해석 가능성이 높고, 실전 파일럿에 더 적합하다.

확인된 구조:

```text
rule_signal_builder.py
→ rule_portfolio_manager.py
→ rule_order_preview_builder.py
→ rule_order_submitter.py
→ rule_order_fill_sync.py
```

현재 장점:

```text
strong_entry 기반 BUY 제한
market_defensive_mode BUY 차단
gap_risk 차단
trading_value 기준
sector limit
cooldown
cash limit
paper/pilot/live 모드 분리
RULE 계좌 분리
RULE_KILL_SWITCH
RULE_MAX_ORDER_AMOUNT
```

하지만 보완해야 할 점은 다음이다.

```text
청산/축소 규칙의 실전 성과 추적 부족
최대 보유일 미흡
트레일링 스탑 미흡
실제 체결가 기준 리뷰 부족
RULE 신호별 성과 통계 부족
```

---

## 2. RULE 전략의 역할 재정의

AI는 장기적으로 확장성이 있지만, 현재 표본 부족 단계에서는 RULE이 더 안정적인 실전 파일럿 엔진이다.

RULE의 역할:

```text
1. 소액 실전 파일럿의 기준 엔진
2. AI 성과 비교용 baseline
3. 명확한 손절/청산 규칙 테스트베드
4. 사업화 시 설명 가능한 전략 샘플
```

---

## 3. 진입 규칙 유지/강화

현재 진입 규칙은 대체로 유지한다.

유지할 핵심 조건:

```text
strong_entry_signal=True
market_defensive_mode=False
trading_value_pass=True
gap_risk_blocked=False
sector_limit_pass=True
cooldown_pass=True
cash_limit_pass=True
```

추가 권장 조건:

```text
1. 전일 종가 대비 현재가 +3% 초과 시 BUY 차단
2. 당일 고가 대비 현재가 급락 중이면 BUY 차단
3. 장 시작 후 5~10분 이내 급변동이면 BUY 보류
4. 당일 같은 종목 재BUY 차단
5. 최근 3일 내 손절 종목 재진입 차단
```

---

## 4. 청산/축소 규칙 명문화

현재 `rule_portfolio_manager.py`에는 다음 구조가 있다.

```text
보유 중 defensive + rule_score_v2 < 45 → reduce
보유 중 rule_score_v2 < 35 또는 gap_risk_blocked → exit
그 외 hold
```

이 구조를 다음처럼 확장한다.

---

## 4.1 강제 EXIT 조건

```text
1. rule_score_v2 < 35
2. 매수가 대비 -5% 이하
3. 20일선 이탈 + rule_score_v2 하락
4. 거래대금 기준 미달로 유동성 악화
5. 보유 종목에 gap_risk_blocked 발생
6. 최대 보유일 초과 + 수익률 부진
```

권장 사유 코드:

```text
rule_score_exit
stop_loss_exit
ma20_break_exit
liquidity_deterioration_exit
gap_risk_exit
max_holding_days_exit
```

---

## 4.2 REDUCE 조건

```text
1. market_defensive_mode=True and rule_score_v2 < 45
2. 고점 대비 -4% 하락
3. 수익 중이나 모멘텀 둔화
4. 섹터 노출 초과
5. 현금 비중 하한 위협
```

권장 사유 코드:

```text
defensive_reduce
trailing_stop_reduce
momentum_fade_reduce
sector_exposure_reduce
cash_buffer_reduce
```

---

## 4.3 HOLD 조건

```text
1. rule_score_v2 >= 45
2. 손절/트레일링 조건 미발동
3. 시장 방어모드 아님 또는 점수 유지
4. 유동성 기준 유지
5. 최대 보유일 미초과
```

---

## 5. 최대 보유일 도입

### 환경변수

```bash
RULE_MAX_HOLDING_DAYS=10
RULE_MAX_HOLDING_DAYS_PROFIT_BUFFER=0.02
```

### 규칙

```text
보유일 > 10일이고 수익률 < +2%이면 REDUCE 또는 EXIT
보유일 > 15일이면 성과와 무관하게 재평가
```

### 이유

단기 추세/거래량 기반 RULE은 장기 보유로 갈수록 신호의 효력이 약해진다.

---

## 6. 트레일링 스탑 도입

### 환경변수

```bash
RULE_TRAILING_STOP_PCT=0.04
RULE_TRAILING_STOP_MIN_PROFIT_PCT=0.03
```

### 규칙

```text
진입 후 최고 평가수익률이 +3% 이상이었다가
고점 대비 -4% 하락하면 REDUCE 또는 EXIT
```

### 필수 저장값

```text
entry_price
highest_price_since_entry
highest_return_since_entry
drawdown_from_high
```

---

## 7. 손절 규칙

### 환경변수

```bash
RULE_STOP_LOSS_PCT=0.05
```

### 규칙

```text
매수가 대비 -5% 이하이면 EXIT 후보
시장 급락일에는 즉시 시장가가 아니라 지정가/분할청산 검토
```

### 주의

손절은 수익률만으로 판단하면 안 된다. 유동성, 호가, 시장 급락 여부를 함께 봐야 한다.

---

## 8. RULE 성과 리포트 강화

### 신규/확장 리포트

```text
outputs/rule_strategy_performance_report.md
outputs/rule_signal_quality_report.md
outputs/rule_exit_reason_report.md
```

### 포함 항목

```text
strong_entry 발생 수
실제 BUY 수
BUY 차단 수와 사유
진입 후 1일/3일/5일/10일 수익률
청산 사유별 평균 수익률
손절 후 재상승 여부
gap_risk 차단 종목의 이후 수익률
trading_value 기준 통과/미통과 성과 비교
sector별 성과
```

---

## 9. RULE 운영 정책

현재 단계 권장:

```text
RULE_TRADING_RUN_MODE=pilot
RULE_LIVE_ENABLED=1
RULE_ORDER_SUBMIT_ENABLED=1
RULE_KILL_SWITCH=0
RULE_MAX_POSITIONS=3~5
RULE_NEW_ENTRY_WEIGHT=0.03~0.05
RULE_MAX_POSITION_WEIGHT=0.10~0.15
RULE_MIN_CASH_WEIGHT=0.30 이상
RULE_MAX_ORDER_AMOUNT=100000~300000
```

단, 실제 금액은 총 운용자금에 맞춰 더 작게 시작한다.

---

## 10. 완료 기준

```text
1. RULE BUY/REDUCE/EXIT/HOLD 사유가 모두 명확히 기록된다.
2. 최대 보유일, 손절, 트레일링 스탑이 구현된다.
3. RULE 성과가 AI와 분리되어 리포트된다.
4. RULE 차단 사유별 사후 성과를 확인할 수 있다.
5. RULE pilot에서 최소 20건 이상 체결 후 성과 평가가 가능하다.
```
