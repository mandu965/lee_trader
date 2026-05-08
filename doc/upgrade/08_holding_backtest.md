# 8차 과제: max_holding_days 청산 백테스트 실험

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 없음 (독립 진행 가능, 실험성 과제)
> 다음 과제: 없음 (결과에 따라 rule_portfolio_manager.py 변경 여부 결정)

---

## 목적

현재 Rule 자동매매의 보유기간 청산 로직:

```python
# max_holding_days(기본 10일) 초과 시
if current_return_pct < max_holding_profit_buffer:  # 기본 2%
    action = "reduce" if current_return_pct > 0 else "exit"
```

수익률 +1.9%인 종목도 청산 대상이 된다.
추세가 강하게 유지 중인 경우 조기 청산이 손실이 될 수 있다.

이 과제는 소스를 바로 바꾸지 않고,
세 가지 안을 백테스트로 비교한 뒤 결정한다.

---

## 실험안

### A안: 현재 방식 (기준선)

```
max_holding_days 초과 + 수익률 2% 미만이면 청산
```

### B안: 추세 유지 시 연장

```
max_holding_days 초과 시
  rule_score_v2 >= 70
  AND close > ma20
  AND market_entry_allowed = true
이면 5일 추가 보유 허용
그 외는 현재 방식과 동일
```

### C안: trailing stop 중심

```
보유기간 기준 대신 고점 대비 하락폭으로 청산
trailing_stop_pct = 0.04
trailing_stop_min_profit_pct = 0.03
보유기간 상한 완화 (15일)
```

---

## 백테스트 비교 지표

- 평균 수익률 (per trade)
- 승률
- MDD
- 평균 보유기간
- 조기 청산 후 추가 상승 비율 (기회비용)

---

## 수정 대상 파일 (작성 예정)

- python/rule_portfolio_backtest.py
- python/rule_backtest.py
- python/analyze_backtest_results.py

---

## 완료 기준

- [ ] A/B/C안 백테스트 결과 표 작성
- [ ] 통계적으로 유의미한 차이 확인
- [ ] 최종 채택안 결정 및 기록
- [ ] 채택안 적용 시 수정 파일 목록 작성

---

## 완료 후 기록

완료일:
백테스트 결과 요약:
채택안:
변경 파일 (채택 시):
주요 결정 사항:
