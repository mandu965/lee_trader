# RULE 전략 공식 수정 — 버그 픽스 v1

**작성일**: 2026-05-14  
**대상 파일**: `python/rule_signal_builder.py`, `python/rule_portfolio_manager.py`  
**수정 성격**: 버그 픽스 (로직 오류 교정) — 전략 방향 변경 아님

---

## 수정 내용 요약

### Fix 1 — RSI 진입 구간 dead zone 제거

**파일**: `python/rule_signal_builder.py` (line 322)

**문제**
```python
# 변경 전
& (rsi.between(45, 75))   # base_conditions 진입 허용 구간

overheated = (rsi > 80)   # 과열 차단 구간
```

RSI 75 초과 ~ 80 이하 구간(예: RSI=77)이 **어떤 조건에도 해당하지 않는 논리 공백**이었습니다.
- `between(45, 75)` 기준으로 탈락 (진입 불가)
- `rsi > 80` 기준으로 과열 아님
- 결과: 모멘텀이 강한 RSI 75~80 종목이 이유 없이 걸러짐

**수정**
```python
# 변경 후
& (rsi.between(45, 80))   # 과열 임계값(80)과 정렬
```

**효과**
- RSI 75~80 구간 종목이 정상적으로 평가됨
- `overheated = (rsi > 80)` 차단 로직과 완벽히 정렬
- 강한 모멘텀 종목 중 일부가 새로 진입 후보에 포함될 수 있음

---

### Fix 2 — 트레일링 스탑 파라미터 교정

**파일**: `python/rule_portfolio_manager.py` (line 184~185)

**문제**
```python
# 변경 전
trailing_stop_pct = cfg_float("RULE_TRAILING_STOP_PCT", 0.04)           # 최고점 대비 -4% 하락 시 발동
trailing_stop_min_profit_pct = cfg_float("RULE_TRAILING_STOP_MIN_PROFIT_PCT", 0.03)  # 최소 수익 +3% 이상일 때만 발동
```

트레일링 스탑의 목적은 **이익 보호**이지만, 이 설정으로는 오히려 손실로 청산됩니다:

```
시나리오: 최고 수익 +3% 도달 후 -4% 하락
  → 현재 수익률 = +3% - 4% = -1%
  → 손실 상태에서 청산 (이익 보호 실패)
```

**수정**
```python
# 변경 후
trailing_stop_pct = cfg_float("RULE_TRAILING_STOP_PCT", 0.025)          # 최고점 대비 -2.5% 하락 시 발동
trailing_stop_min_profit_pct = cfg_float("RULE_TRAILING_STOP_MIN_PROFIT_PCT", 0.05)  # 최소 수익 +5% 이상일 때만 발동
```

**효과 검증**
```
수정 후 시나리오: 최고 수익 +5% 도달 후 -2.5% 하락
  → 현재 수익률 = +5% - 2.5% = +2.5%
  → 수익 확정 청산 (이익 보호 달성)
```

**환경변수 오버라이드 가능** (코드 변경 없이 조정 가능):
```bash
RULE_TRAILING_STOP_PCT=0.025
RULE_TRAILING_STOP_MIN_PROFIT_PCT=0.05
```

---

## 서버 작업 절차

### Step 1 — Git Pull (최신 코드 받기)

서버에서:
```bash
cd /path/to/lee_trader   # 서버의 프로젝트 경로
git pull --ff-only origin main
```

변경된 파일 확인:
```bash
git log --oneline -3
git diff HEAD~1 HEAD -- python/rule_signal_builder.py python/rule_portfolio_manager.py
```

---

### Step 2 — 백테스트 비교 실행

기존 전략과 수정된 전략을 비교합니다.

```bash
cd /path/to/lee_trader/python

# 백테스트 전체 재실행 (rule_signals.csv 기반)
python rule_backtest.py

# 변형 비교 리뷰 (4가지 variant 비교표 생성)
python rule_formula_review.py
```

출력 파일:
- `outputs/rule_strategy_backtest_report.json` — 전체 백테스트 결과
- `outputs/rule_strategy_backtest_report.md` — 마크다운 리포트
- `outputs/rule_formula_review.json` — 변형 비교 결과
- `outputs/rule_formula_review.md` — 변형 비교 마크다운

---

### Step 3 — 결과 확인 포인트

백테스트 결과에서 이 숫자들을 이전과 비교합니다:

| 지표 | 이전값 (2026-04-29) | 확인 포인트 |
|---|---|---|
| strong_entry 거래건수 | 1,267건 | 증가 여부 (RSI 구간 확대로 후보 증가 기대) |
| strong_entry 20일 승률 | 50.6% | 50% 이상 유지 확인 |
| strong_entry 20일 평균수익 | +1.87% | 변화 방향 확인 |
| 포트폴리오 최종 누적수익 | +140.37% | 유지 또는 개선 확인 |
| 포트폴리오 MDD | -33.01% | 악화되지 않는지 확인 |

---

### Step 4 — 시그널 재생성 (선택)

오늘 날짜 기준 시그널을 다시 생성하려면:
```bash
python rule_signal_builder.py
python rule_portfolio_manager.py
```

---

### Step 5 — 문제 발생 시 롤백

파라미터가 ENV 변수로 오버라이드 가능하므로, 코드 롤백 없이 이전 동작으로 복구할 수 있습니다:

```bash
# .env 또는 환경변수에 추가하여 이전 값으로 되돌리기
RULE_TRAILING_STOP_PCT=0.04
RULE_TRAILING_STOP_MIN_PROFIT_PCT=0.03
```

코드 자체를 롤백하려면:
```bash
git revert HEAD
git push origin main
```

---

## 향후 검토 예정 항목 (이번 수정에서 제외)

다음 항목들은 백테스트 결과 확인 후 2단계로 검토합니다:

| 항목 | 현재값 | 제안값 | 이유 |
|---|---|---|---|
| flow 가중치 | 15% | 25~30% | 외국인·기관 수급이 한국시장 핵심 알파 |
| max_holding_days | 10일 | 20일 | 모멘텀 수익 실현 지평선과 정렬 |
| 거래대금 기준 (paper) | 5억 | 10억 | 소형·테마주 노이즈 감소 |
| 포지션 크기 | 고정 5% | 신호강도 연동 4~7% | 자본배분 효율화 |

---

## 변경 로그

| 날짜 | 파일 | 변경 내용 |
|---|---|---|
| 2026-05-14 | rule_signal_builder.py | RSI 진입 구간 45~75 → 45~80 |
| 2026-05-14 | rule_portfolio_manager.py | 트레일링 스탑 min_profit 3%→5%, 하락폭 4%→2.5% |
