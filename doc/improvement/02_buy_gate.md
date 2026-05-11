# 2차 과제: 자동매매 BUY gate 신뢰도 연결

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 01_trust_gate.md 완료 후 진행
> 다음 과제: 03_ai_exit.md

---

## 목적

1차 과제에서 ranking에 buy_eligible, score_trust_level이 추가되었다.
이 과제에서는 AI 자동매매 BUY gate에 다음 기준을 추가한다.

- `prob_score_raw` 절대 확률 최소 기준 적용
- `fallback_count` 기반 신뢰도 조건 적용
- `score_missing_flags` 기반 후보 제외
- 랭킹 1위라도 절대 확률이 낮으면 자동매매 차단

---

## 배경

현재 `prob_score`는 당일 종목군 내 percentile rank 성격이다.
시장 전체가 안 좋은 날에도 상대적으로 1등인 종목은 높은 `prob_score`를 받는다.
따라서 자동매매 BUY 조건은 다음 구조가 되어야 한다.

```
prob_score >= 상대 순위 기준 (기존)
AND prob_score_raw >= 절대 확률 기준 (신규)
AND confidence_score >= 최소 신뢰도 (신규)
AND fallback_count <= 허용치 (신규)
AND buy_eligible = true (1차 과제 결과 활용)
```

---

## 수정 대상 파일 (작성 예정)

- python/submit_live_orders.py
- python/common_live_risk_guard.py (또는 별도 gate 모듈)
- python/build_live_order_preview.py

---

## 환경변수 (예정)

| 변수명 | 기본값 | 설명 |
|---|---|---|
| AI_BUY_GATE_MIN_PROB_SCORE_RAW | 55 | 절대 확률 최소값 |
| AI_BUY_GATE_MAX_FALLBACK_COUNT | 1 | 허용 fallback 최대값 |
| AI_BUY_GATE_REQUIRE_ELIGIBLE | true | buy_eligible=true 필수 여부 |
| AI_BUY_GATE_MIN_CONFIDENCE | 60 | confidence_score 최소값 |

---

## 완료 기준 (작성 예정)

- [ ] prob_score_raw 기준 미달 종목 BUY 차단 확인
- [ ] buy_eligible=false 종목 BUY 차단 확인
- [ ] block 사유 로그 출력 확인

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
