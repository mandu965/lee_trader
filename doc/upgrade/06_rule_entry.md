# 6차 과제: Rule live strong_entry 기준 고정

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 없음 (독립 진행 가능)
> 다음 과제: 07_regime_unify.md

---

## 목적

현재 Rule 자동매매의 BUY 진입 신호 구조:

```python
# rule_signal_builder.py
entry_signal = base_conditions & (~overheated) & (
    (rule_score >= 70) | (rule_score_v2 >= 65)   # OR 조건
)
strong_entry_signal = base_conditions & (~overheated) & (
    (rule_score >= 70) & (rule_score_v2 >= 60)   # AND 조건
)
```

`entry_signal`은 절대 점수(rule_score)와 상대 백분위(rule_score_v2)를
OR로 묶는다. `rule_score_v2 >= 65`는 시장 상위 35% 종목이므로
허들이 낮고, OR 구조에서는 더 관대한 조건이 지배한다.

현재 `rule_account_guard.py`에는 이미 다음 guard가 있다:

```python
# rule_account_guard.py L220-221
if order_context.get("signal_strength") != "strong_entry":
    reasons.append("buy_requires_strong_entry")
```

그러나 이 guard가 **항상** 발동하는 것이 아니라
조건부로 발동해야 하며, 현재는 모든 모드에서 동일하게 적용된다.

이 과제의 목표:
- `signal_strength != strong_entry`면 강제 차단하는 현재 guard를
  **환경변수로 모드별 제어 가능**하게 만든다.
- pilot/live: strong_entry 필수 (기본값 true)
- paper: entry_signal도 허용 가능 (기본값 true)

---

## 현재 구조 상세 파악

### rule_signal_builder.py — 신호 생성

```python
# 현재 threshold (환경변수로 제어 가능)
entry_rule_score_min = get_float_env("RULE_ENTRY_RULE_SCORE_MIN", 70.0)
entry_rule_score_v2_min = get_float_env("RULE_ENTRY_RULE_SCORE_V2_MIN", 65.0)
strong_rule_score_min = get_float_env("RULE_STRONG_RULE_SCORE_MIN", 70.0)
strong_rule_score_v2_min = get_float_env("RULE_STRONG_RULE_SCORE_V2_MIN", 60.0)

entry_signal = base_conditions & (~overheated) & (
    (rule_score >= entry_rule_score_min) |
    (rule_score_v2 >= entry_rule_score_v2_min)
)
strong_entry_signal = base_conditions & (~overheated) & (
    (rule_score >= strong_rule_score_min) &
    (rule_score_v2 >= strong_rule_score_v2_min)
)
signal_strength = select(
    [strong_entry_signal, entry_signal],
    ["strong_entry", "entry"],
    default="none"
)
```

### rule_account_guard.py — BUY guard

```python
# L220: 현재 항상 strong_entry만 허용
if order_context.get("signal_strength") != "strong_entry":
    reasons.append("buy_requires_strong_entry")
```

이 줄이 환경변수로 제어되지 않아
paper 모드에서도 entry_signal 종목이 차단된다.

### rule_order_preview_builder.py — 신호 전달

`signal_strength` 값이 order_context에 담겨
`rule_account_guard.py`로 전달된다.

---

## 변경 방향

### rule_account_guard.py 수정

`buy_requires_strong_entry` 체크를 환경변수로 제어한다.

```python
# 변경 전
if order_context.get("signal_strength") != "strong_entry":
    reasons.append("buy_requires_strong_entry")

# 변경 후
run_mode = str(order_context.get("run_mode") or "paper").lower()

if run_mode in {"pilot", "live"}:
    require_strong = _flag("RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY", "1")
else:
    # paper 모드
    require_strong = not _flag("RULE_PAPER_ALLOW_ENTRY_SIGNAL", "1")

if require_strong and order_context.get("signal_strength") != "strong_entry":
    reasons.append("buy_requires_strong_entry")
```

### .env.example 추가

```
# Rule 자동매매 진입 기준
RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true   # pilot/live에서 strong_entry만 허용
RULE_PAPER_ALLOW_ENTRY_SIGNAL=true          # paper에서 entry_signal 허용
```

---

## 동작 매트릭스

| run_mode | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY | RULE_PAPER_ALLOW_ENTRY_SIGNAL | entry_signal | strong_entry_signal |
|---|---|---|---|---|
| paper | - | true (기본) | ✅ 허용 | ✅ 허용 |
| paper | - | false | ❌ 차단 | ✅ 허용 |
| pilot | true (기본) | - | ❌ 차단 | ✅ 허용 |
| pilot | false | - | ✅ 허용 | ✅ 허용 |
| live | true (기본) | - | ❌ 차단 | ✅ 허용 |
| live | false | - | ✅ 허용 | ✅ 허용 |

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 수정 | python/rule_account_guard.py | buy_requires_strong_entry 환경변수 제어 |
| 수정 | .env.example | 신규 환경변수 추가 |

### 수정 금지 파일

- python/rule_signal_builder.py
  (entry_signal / strong_entry_signal 생성 로직 변경 금지)
- python/rule_order_preview_builder.py
- python/rule_order_submitter.py
- python/rule_portfolio_manager.py

---

## 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY | true | pilot/live에서 strong_entry 필수 여부 |
| RULE_PAPER_ALLOW_ENTRY_SIGNAL | true | paper 모드에서 entry_signal 허용 여부 |

기존 환경변수(변경 없음):

| 변수명 | 기본값 | 설명 |
|---|---|---|
| RULE_ENTRY_RULE_SCORE_MIN | 70.0 | entry_signal rule_score 기준 |
| RULE_ENTRY_RULE_SCORE_V2_MIN | 65.0 | entry_signal rule_score_v2 기준 |
| RULE_STRONG_RULE_SCORE_MIN | 70.0 | strong_entry rule_score 기준 |
| RULE_STRONG_RULE_SCORE_V2_MIN | 60.0 | strong_entry rule_score_v2 기준 |

---

## 검증 케이스

| # | run_mode | 설정 | signal_strength | 기대 결과 |
|---|---|---|---|---|
| 1 | live | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true | entry | ❌ buy_requires_strong_entry |
| 2 | live | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true | strong_entry | ✅ 허용 |
| 3 | live | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=false | entry | ✅ 허용 |
| 4 | pilot | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true | entry | ❌ buy_requires_strong_entry |
| 5 | pilot | RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true | strong_entry | ✅ 허용 |
| 6 | paper | RULE_PAPER_ALLOW_ENTRY_SIGNAL=true | entry | ✅ 허용 |
| 7 | paper | RULE_PAPER_ALLOW_ENTRY_SIGNAL=true | strong_entry | ✅ 허용 |
| 8 | paper | RULE_PAPER_ALLOW_ENTRY_SIGNAL=false | entry | ❌ buy_requires_strong_entry |
| 9 | paper | RULE_PAPER_ALLOW_ENTRY_SIGNAL=false | strong_entry | ✅ 허용 |

---

## 주의사항

- rule_signal_builder.py의 entry_signal / strong_entry_signal
  생성 로직은 변경하지 말 것
- 기존에 live/pilot에서 strong_entry만 허용되던 실운영 동작이
  기본값(RULE_LIVE_BUY_REQUIRES_STRONG_ENTRY=true)에 의해
  변경 후에도 동일하게 유지되는지 반드시 확인할 것
- paper 모드의 기존 동작은 현재 strong_entry만 허용이므로
  RULE_PAPER_ALLOW_ENTRY_SIGNAL=true(기본값) 적용 시
  paper에서 허용 범위가 넓어지는 것을 인지하고 배포할 것

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
