# 7차 과제: AI/Rule 시장 레짐 공통화

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 06_rule_entry.md 완료 후 진행 권장
> 다음 과제: 없음 (독립 완결)

---

## 목적

현재 AI와 Rule이 서로 다른 기준으로 시장 레짐을 판단한다.

AI 랭킹 (final_score.py detect_market_regime):
- 5개 조건: close>ma20, ma20>ma60, 20일 수익률,
            개별 종목 breadth, 변동성 flag
- 출력: bull / neutral / defensive

Rule (rule_signal_builder.py attach_market):
- 3개 조건: kospi<ma20, momentum<0, 변동성 rising/high
- 출력: regime_risk_flag (true/false)

두 시스템이 서로 다른 시장 판단 하에 동작할 수 있다.
예: AI는 neutral(매수 허용)인데 Rule은 defensive(매수 차단)

---

## 변경 방향

공통 레짐 판단 모듈을 만들고
AI와 Rule이 같은 결과를 읽도록 한다.

### 신설 파일 (예정)
- python/market_regime_service.py

### 출력 구조 (예정)

```python
{
  "market_regime": "bull" / "neutral" / "defensive",
  "market_entry_allowed": True / False,
  "volatility_state": "normal" / "elevated" / "high",
  "breadth_state": "strong" / "neutral" / "weak",
  "index_trend_state": "up" / "flat" / "down",
  "regime_reason_codes": ["close_gt_ma20", "ma20_gt_ma60", ...],
  "regime_score": 3,
  "as_of_date": "2026-05-07"
}
```

---

## 수정 대상 파일 (작성 예정)

- python/market_regime_service.py (신설)
- python/scoring/final_score.py (공통 모듈 참조)
- python/rule_signal_builder.py (공통 모듈 참조)

---

## 주의사항

- 기존 레짐 판단 로직을 바로 삭제하지 말 것
- 공통 모듈 도입 후 기존 로직과 결과를 비교 검증한 뒤 전환
- AI와 Rule의 레짐 판단 결과가 달라지는 날을 로그로 기록

---

## 완료 기준 (작성 예정)

- [ ] market_regime_service.py 단독 실행 확인
- [ ] AI ranking 레짐 판단 결과 일치 확인
- [ ] Rule signal 레짐 판단 결과 일치 확인
- [ ] 불일치 발생 시 경고 로그 확인

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
