# 3차 과제: AI 자동매매 종목별 청산 로직 추가

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 02_buy_gate.md 완료 후 진행 권장
> 다음 과제: 없음 (독립 완결)

---

## 목적

현재 AI 자동매매는 매수 추천 로직은 있지만,
개별 종목 단위의 청산 기준이 없다.
계좌 전체 손실 guard(common_live_risk_guard.py)에만 의존하고 있어
한 종목이 크게 하락해도 개별 대응이 느리다.

Rule 자동매매에는 stop_loss, trailing_stop, max_holding_days,
profit_target이 이미 구현되어 있다.
AI 자동매매에도 동일한 수준의 종목별 청산 기준이 필요하다.

---

## 배경

Rule 자동매매의 evaluate_position_risk() 구조:
- stop_loss_pct (기본 5%)
- trailing_stop_pct (기본 4%, 고점 대비 하락폭)
- trailing_stop_min_profit_pct (기본 3%, 최소 수익 후 발동)
- max_holding_days (기본 10일)
- max_holding_days_defensive (기본 7일)
- profit_target_pct (기본 0, 비활성)

AI 자동매매에는 이에 상응하는 로직이 없다.

---

## 추가할 환경변수 (예정)

| 변수명 | 기본값 | 설명 |
|---|---|---|
| AI_POSITION_STOP_LOSS_PCT | 0.05 | 개별 종목 손절 기준 |
| AI_POSITION_TAKE_PROFIT_PCT | 0.0 | 익절 기준 (0이면 비활성) |
| AI_MAX_HOLDING_DAYS | 20 | 최대 보유일 |
| AI_TRAILING_STOP_PCT | 0.05 | 고점 대비 trailing stop |
| AI_TRAILING_STOP_MIN_PROFIT_PCT | 0.03 | trailing stop 발동 최소 수익 |
| AI_SCORE_DROP_EXIT_THRESHOLD | 0.0 | final_score 하락 시 청산 기준 |

---

## 수정 대상 파일 (작성 예정)

- python/submit_live_orders.py
- python/build_trade_intents.py
- python/run_live_auto_trade_cycle.py

### 신설 파일 (예정)
- python/ai_position_risk.py (Rule의 evaluate_position_risk에 대응)

---

## 완료 기준 (작성 예정)

- [ ] 개별 종목 stop loss 발동 확인
- [ ] trailing stop 발동 확인
- [ ] max_holding_days 초과 청산 확인
- [ ] AI_SCORE_DROP_EXIT 발동 확인
- [ ] 청산 사유 로그 출력 확인

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
