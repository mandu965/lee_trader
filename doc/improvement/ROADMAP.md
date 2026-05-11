# Lee Trader 개선 로드맵

> 최초 작성: 2026-05-07
> 기준 분석: claude.md + gpt.md 통합 회의 결과

---

## 배경

2026-05-07 기준 소스 분석 결과, 현재 프로젝트는 기능 자체가 없는 상태가 아니라
**점수 신뢰도 · 자동매매 복구력 · 전략 기준 일관성**을 더 강하게 묶어야 하는 단계.

### 핵심 원칙

```
1. 점수가 높다  ≠  믿고 사도 된다
2. 주문 실패    ≠  그냥 실패 처리하고 끝
3. Rule 조건 통과  ≠  live에서 바로 매수
4. AI 매수      ≠  AI 청산 로직 없이 보유
```

### 구조적 방향

- `final_score`는 "좋은 종목인가?" 를 나타낸다
- `buy_eligible`은 "지금 실제로 사도 되는가?" 를 나타낸다
- 이 두 가지는 반드시 분리되어야 한다

---

## 과제 목록

| 순위 | 과제 | 상태 | 파일 |
|---:|---|:---:|---|
| 1 | Ranking Trust Gate 구축 | ⏸ 불필요 | 01_trust_gate.md |
| 2 | 자동매매 BUY gate 신뢰도 연결 | ⏸ 불필요 | 02_buy_gate.md |
| 3 | AI 자동매매 종목별 청산 로직 추가 | ✅ 완료 | 03_ai_exit.md |
| 4 | submit_unknown + broker lookup 복구 | ⏸ 보류 | 04_order_recovery.md |
| 5 | scheduler health ledger 구축 | ✅ 완료 | 05_health_ledger.md |
| 6 | Rule live strong_entry 기준 고정 | ⏸ 불필요 | 06_rule_entry.md |
| 7 | AI/Rule 시장 레짐 공통화 | ⏸ 보류 | 07_regime_unify.md |
| 8 | max_holding_days 청산 백테스트 실험 | ⏸ 보류 | 08_holding_backtest.md |
| 9 | 섹터 집중 hard cap 명시화 | ⏸ 불필요 | 09_sector_cap.md |
| 10 | theme overlay 활성화 시 알림/로그 추가 | ⏸ 불필요 | 10_theme_alert.md |

**상태 표시:** ✅ 완료 / 🔄 진행중 / ⬜ 대기 / ⏸ 보류

---

## 과제 간 의존성

```
1 (Trust Gate)
  └── 2 (BUY gate 연결)   ← 1 완료 후 진행 가능
        └── 3 (AI 청산)   ← 2 완료 후 진행 권장

4 (주문 복구)              ← 1과 독립적으로 진행 가능
5 (health ledger)         ← 1과 독립적으로 진행 가능

6 (Rule entry 기준)        ← 독립 진행 가능
7 (레짐 공통화)             ← 6 완료 후 진행 권장

8 (holding 백테스트)        ← 독립 진행 가능 (실험성)
9 (섹터 cap)               ← 독립 진행 가능
10 (theme alert)           ← 독립 진행 가능
```

---

## Claude Code 사용 가이드

### 새 대화 시작 시

```
doc/improvement/ROADMAP.md를 먼저 읽고
현재 진행 중인 과제 파일을 확인한 뒤
작업을 이어서 진행하라.
```

### 과제 완료 후 마무리

```
완료된 내용을 doc/improvement/[과제파일].md의
"완료 후 기록" 섹션에 기록하고
ROADMAP.md의 상태를 ✅ 완료로 변경하라.
```

### 새 과제 시작 시

```
doc/improvement/ROADMAP.md와
doc/improvement/[과제파일].md를 읽고
해당 과제의 필수 확인 파일을 파악한 뒤 작업을 시작하라.
파악이 완료되기 전에는 코드를 작성하지 말 것.
```

---

## 진행 원칙

1. 한 번에 하나의 과제만 진행한다.
2. 과제 시작 전 ROADMAP.md와 해당 과제 파일을 먼저 읽는다.
3. 해당 과제의 수정 허용 파일 외에는 수정하지 않는다.
4. 기존 실계좌 주문/KIS API 호출은 테스트 목적으로 실행하지 않는다.
5. 변경 후 반드시 검증 결과를 과제 파일의 "완료 후 기록"에 남긴다.
6. 완료 후 ROADMAP.md 상태를 갱신한다.


## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-05-07 | 최초 작성. 10개 과제 정의. |
| 2026-05-12 | 3차 과제(AI 청산 로직) 완료. ai_position_risk.py 신설, apply_execution_policy.py·run_daily_scheduler.py·run_live_auto_trade_cycle.py 수정. |
| 2026-05-12 | 5차 과제(scheduler health ledger) 완료. scheduler_health.py 신설, run_daily_scheduler.py 수정. |
| 2026-05-12 | 전체 과제 재검토. 1·2·6·9·10번 불필요(이미 구현되어 있거나 현실과 불일치), 4·7·8번 보류. 실질 완료. |
