# 1차 과제: Ranking Trust Gate 구축

> 상태: 🔄 진행중
> 작성일: 2026-05-07
> 의존성: 없음 (첫 번째 과제)
> 다음 과제: 02_buy_gate.md (이 과제 완료 후 진행)

---

## 목적

현재 `ranking final_score`는 산출되지만, 점수 누락/보정/fallback 상태가
자동매매 BUY 후보 선정에 충분히 강하게 반영되지 않는다.

- 랭킹은 기존대로 보여준다
- 자동매매는 신뢰도 기준을 통과한 종목만 BUY 후보로 허용한다
- `final_score` 기존 산식과 rank 순위 산정은 변경하지 않는다

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 신설 | python/scoring/trust_gate.py | Trust Gate 판단 로직 전체 |
| 수정 | python/scoring/final_score.py | trust_gate import 연결부만 추가 |
| 수정 | python/ranking_builder.py | trust gate 적용, 3개 컬럼 추가, summary 출력 |
| 수정 | python/submit_live_orders.py | buy_eligible=false 종목 필터링 |

### 수정 금지 파일

- python/rule_order_preview_builder.py
- python/rule_account_guard.py
- python/rule_signal_builder.py
- python/common_live_risk_guard.py

---

## 추가되는 컬럼

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| buy_eligible | bool | true = BUY 후보 허용 |
| buy_block_reason | str | 제외 사유 (쉼표 구분), 허용 시 빈 문자열 |
| score_trust_level | str | HIGH / MEDIUM / LOW / BLOCKED |

### score_trust_level 기준

```
BLOCKED : buy_eligible = false
LOW     : buy_eligible = true, qual_score_missing = true
MEDIUM  : buy_eligible = true, qual_score_missing = false, fallback_count = 1
HIGH    : buy_eligible = true, qual_score_missing = false, fallback_count = 0
```

---

## 환경변수 목록

| 변수명 | 기본값 | 설명 |
|---|---|---|
| RANKING_TRUST_GATE_ENABLED | true | false이면 gate 전체 건너뜀 |
| RANKING_TRUST_BLOCK_RET_MISSING | true | ret_score 결측 시 제외 |
| RANKING_TRUST_BLOCK_PROB_MISSING | true | prob_score 결측 시 제외 |
| RANKING_TRUST_BLOCK_TECH_MISSING | true | tech_score 결측 시 제외 |
| RANKING_TRUST_MAX_FALLBACK_COUNT | 1 | 초과 시 제외 |
| RANKING_TRUST_MIN_PROB_SCORE_RAW | 55 | 미만 시 제외 |
| RANKING_TRUST_LOW_ON_QUAL_MISSING | true | qual 결측 시 LOW 처리 |
| RANKING_TRUST_ALLOW_LOW_IN_PAPER | true | paper 모드 LOW 허용 여부 |
| RANKING_TRUST_ALLOW_LOW_IN_LIVE | false | pilot/live 모드 LOW 허용 여부 |

---

## buy_block_reason 값 목록

```
ret_score_missing
prob_score_missing
tech_score_missing
fallback_count_exceeded
prob_score_raw_below_threshold
qual_score_missing_low_confidence
trust_level_low_not_allowed
```

---

## 구조화 로그 형식

buy_eligible=false 판정 시 WARNING 레벨로 출력.

```
TRUST_GATE_BLOCK | date=2026-05-07 | code=005930 | name=삼성전자 |
final_score=72.3 | fallback_count=2 | prob_score_raw=48.0 |
block_reasons=fallback_count_exceeded,prob_score_raw_below_threshold |
score_trust_level=BLOCKED
```

---

## ranking_trust_summary.json

출력 경로: `outputs/ranking_trust_summary.json`

```json
{
  "generated_at": "2026-05-07T09:00:00",
  "as_of_date": "2026-05-07",
  "total_ranked": 150,
  "buy_eligible_count": 98,
  "buy_blocked_count": 52,
  "score_trust_level": {
    "HIGH": 80,
    "MEDIUM": 18,
    "LOW": 10,
    "BLOCKED": 52
  },
  "block_reason_counts": {
    "fallback_count_exceeded": 20,
    "prob_score_raw_below_threshold": 18,
    "ret_score_missing": 5,
    "prob_score_missing": 3,
    "tech_score_missing": 2,
    "trust_level_low_not_allowed": 4
  },
  "gate_enabled": true,
  "run_mode": "live",
  "settings": {
    "max_fallback_count": 1,
    "min_prob_score_raw": 55,
    "block_ret_missing": true,
    "block_prob_missing": true,
    "block_tech_missing": true,
    "low_on_qual_missing": true,
    "allow_low_in_paper": true,
    "allow_low_in_live": false
  }
}
```

---

## 검증 케이스 7개

| # | 조건 | 기대 결과 |
|---|---|---|
| 1 | fallback_count=2 | buy_eligible=false, reason=fallback_count_exceeded |
| 2 | prob_score_raw=40 | buy_eligible=false, reason=prob_score_raw_below_threshold |
| 3 | ret_score_missing=true | buy_eligible=false, reason=ret_score_missing |
| 4 | prob_score_missing=true | buy_eligible=false, reason=prob_score_missing |
| 5 | qual_score_missing=true, 그 외 정상 | buy_eligible=true, score_trust_level=LOW |
| 6 | score_trust_level=LOW, ALLOW_LOW_IN_LIVE=false | buy_eligible=false, reason=trust_level_low_not_allowed |
| 7 | fallback_count=0, 모든 점수 정상, prob_score_raw=70 | buy_eligible=true, score_trust_level=HIGH |

---

## 주의사항

- 실계좌 주문 로직을 직접 실행하지 말 것
- KIS API 호출을 테스트 목적으로 실행하지 말 것
- 기존 운영 파일을 삭제하지 말 것
- final_score 기존 산식과 rank 순위 산정은 변경하지 말 것
- buy_eligible=false인 종목도 ranking 행에서 절대 제거하지 말 것
  (행 제거는 자동매매 BUY 후보 선정 단계에서만 수행)
- Rule 자동매매 파일은 이번 과제에서 수정하지 말 것

---

## 실행 프롬프트

```
지금부터 Lee Trader 프로젝트의 1차 개선 과제인
"Ranking Trust Gate 구축"을 진행합니다.

이 과제에서 수정 가능한 파일은 아래 명시된 파일에만 한정한다.
그 외 파일은 읽기만 허용한다.

---

## 목표

현재 ranking final_score는 산출되지만, 점수 누락/보정/fallback 상태가
자동매매 BUY 후보 선정에 충분히 강하게 반영되지 않는다.

이번 과제의 목표는 다음 구조를 만드는 것이다.
- 랭킹은 기존대로 보여준다
- 자동매매는 신뢰도 기준을 통과한 종목만 BUY 후보로 허용한다
- final_score 기존 산식과 rank 순위 산정은 변경하지 않는다

---

## Step 1. 필수 파일 읽기

아래 파일들을 반드시 먼저 읽고 현재 구조를 파악하라.
파악이 완료되기 전에는 코드를 작성하지 말 것.

읽기 대상:
- python/scoring/final_score.py
- python/ranking_builder.py
- python/submit_live_orders.py
- python/common_live_risk_guard.py
- python/rule_signal_builder.py
- python/rule_order_preview_builder.py
- python/rule_account_guard.py
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md

파악할 내용:
1. final_score 산정 흐름
2. ret_score, prob_score, tech_score, qual_score, risk_penalty 생성 위치
3. fallback_count 생성 및 사용 위치
4. ret_score_missing, prob_score_missing, tech_score_missing,
   qual_score_missing 컬럼 존재 여부
5. prob_score_raw 존재 여부 및 생성 방식,
   ranking 출력 파일까지 전달되는지 여부
6. AI 자동매매 BUY 후보 선정 조건 (submit_live_orders.py 기준)
7. API 또는 output ranking 파일에 현재 어떤 컬럼이 노출되는지

---

## Step 2. 수정 및 신설 파일

### 신설 파일
- python/scoring/trust_gate.py
  - Trust Gate 판단 로직 전체를 이 파일에 구현한다
  - final_score.py와 ranking_builder.py에서 import하여 사용한다

### 수정 파일
- python/scoring/final_score.py
  - trust_gate.py의 함수를 import하는 연결부만 추가
  - 기존 점수 산식은 절대 변경하지 말 것

- python/ranking_builder.py
  - trust gate 적용 후 buy_eligible, buy_block_reason,
    score_trust_level 컬럼을 출력에 포함
  - prob_score_raw가 출력에 없을 경우
    final_score.py에서 생성된 prob_score_raw를
    ranking 출력 컬럼에 포함시키는 방식으로 해결할 것
    (별도 재계산하지 말 것)
  - 기존 rank 순위 산정 로직은 변경하지 말 것
  - buy_eligible=false인 종목도 ranking 행에서 제거하지 말 것.
    모든 종목은 ranking 결과에 그대로 유지되어야 하며,
    buy_eligible, buy_block_reason, score_trust_level 컬럼만 추가된다.
    행 제거는 자동매매 BUY 후보 선정 단계에서만 수행한다.

- python/submit_live_orders.py
  - BUY 후보 선정 시 buy_eligible=false 종목 제외
  - 제외된 종목은 buy_block_reason을 로그에 남길 것

### 수정 금지 파일
- python/rule_order_preview_builder.py
- python/rule_account_guard.py
- python/rule_signal_builder.py
- python/common_live_risk_guard.py
- (Rule 자동매매 연결은 별도 과제로 분리한다)

---

## Step 3. 구현 요구사항

### 3-1. 추가할 컬럼

ranking 결과에 아래 3개 컬럼을 추가한다.

- buy_eligible: bool
- buy_block_reason: str (복수 사유는 쉼표 구분, 허용 시 빈 문자열)
- score_trust_level: str (HIGH / MEDIUM / LOW / BLOCKED)

### 3-2. score_trust_level 기준

우선순위 순서대로 판단한다.

BLOCKED : buy_eligible = false인 경우
LOW     : buy_eligible = true, qual_score_missing = true (그 외 결측 없음)
MEDIUM  : buy_eligible = true, qual_score_missing = false, fallback_count = 1
HIGH    : buy_eligible = true, qual_score_missing = false, fallback_count = 0

### 3-3. BUY 제외 조건

아래 조건 중 하나라도 해당하면 buy_eligible=false로 설정한다.
환경변수로 각 조건을 켜고 끌 수 있도록 구현한다.

RANKING_TRUST_GATE_ENABLED=true
  - false이면 trust gate 전체를 건너뛰고
    buy_eligible=true, score_trust_level=HIGH로 설정

RANKING_TRUST_BLOCK_RET_MISSING=true
  - ret_score_missing=true이면 BUY 제외
  - buy_block_reason: ret_score_missing

RANKING_TRUST_BLOCK_PROB_MISSING=true
  - prob_score_missing=true이면 BUY 제외
  - buy_block_reason: prob_score_missing

RANKING_TRUST_BLOCK_TECH_MISSING=true
  - tech_score_missing=true이면 BUY 제외
  - buy_block_reason: tech_score_missing

RANKING_TRUST_MAX_FALLBACK_COUNT=1
  - fallback_count > 설정값이면 BUY 제외
  - buy_block_reason: fallback_count_exceeded

RANKING_TRUST_MIN_PROB_SCORE_RAW=55
  - prob_score_raw < 설정값이면 BUY 제외
  - buy_block_reason: prob_score_raw_below_threshold

RANKING_TRUST_LOW_ON_QUAL_MISSING=true
  - qual_score_missing=true이면 score_trust_level=LOW
  - BUY 제외 여부는 아래 설정값에 따름

RANKING_TRUST_ALLOW_LOW_IN_PAPER=true
  - paper 모드에서 LOW 신뢰도 종목 허용 여부

RANKING_TRUST_ALLOW_LOW_IN_LIVE=false
  - pilot/live 모드에서 LOW 신뢰도 종목 허용 여부
  - false이면 buy_block_reason: trust_level_low_not_allowed

### 3-4. buy_block_reason 값 목록

ret_score_missing
prob_score_missing
tech_score_missing
fallback_count_exceeded
prob_score_raw_below_threshold
qual_score_missing_low_confidence
trust_level_low_not_allowed

### 3-5. TRUST_GATE_BLOCK 구조화 로그

trust_gate.py에서 buy_eligible=false 판정 시
아래 형식의 구조화 로그를 WARNING 레벨로 출력한다.

TRUST_GATE_BLOCK | date=2026-05-07 | code=005930 | name=삼성전자 |
final_score=72.3 | fallback_count=2 | prob_score_raw=48.0 |
block_reasons=fallback_count_exceeded,prob_score_raw_below_threshold |
score_trust_level=BLOCKED

필드: date, code, name (없으면 빈 문자열), final_score,
      fallback_count, prob_score_raw (없으면 null),
      block_reasons (쉼표 구분), score_trust_level

### 3-6. ranking_trust_summary.json 출력

ranking_builder.py 실행 완료 후
outputs/ranking_trust_summary.json에 저장한다.

포함 필드:
{
  "generated_at": "...",
  "as_of_date": "...",
  "total_ranked": 150,
  "buy_eligible_count": 98,
  "buy_blocked_count": 52,
  "score_trust_level": { "HIGH": 80, "MEDIUM": 18, "LOW": 10, "BLOCKED": 52 },
  "block_reason_counts": { "fallback_count_exceeded": 20, ... },
  "gate_enabled": true,
  "run_mode": "live",
  "settings": { ... }
}

---

## Step 4. 검증 요구사항

구현 완료 후 아래 7가지 케이스를 직접 실행하여 검증하라.

케이스 1: fallback_count=2 → buy_eligible=false, reason=fallback_count_exceeded
케이스 2: prob_score_raw=40 → buy_eligible=false, reason=prob_score_raw_below_threshold
케이스 3: ret_score_missing=true → buy_eligible=false, reason=ret_score_missing
케이스 4: prob_score_missing=true → buy_eligible=false, reason=prob_score_missing
케이스 5: qual_score_missing=true, 그 외 정상 → buy_eligible=true, score_trust_level=LOW
케이스 6: score_trust_level=LOW, ALLOW_LOW_IN_LIVE=false
          → buy_eligible=false, reason=trust_level_low_not_allowed
케이스 7: fallback_count=0, 모든 점수 정상, prob_score_raw=70
          → buy_eligible=true, score_trust_level=HIGH

---

## Step 5. 문서화

아래 두 문서를 갱신하라.
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/CONTEXT.md

포함 내용:
- Ranking Trust Gate의 목적
- final_score와 buy_eligible의 차이
- score_trust_level 기준
- BUY 제외 조건 목록
- 환경변수 목록 및 기본값
- ranking_trust_summary.json 확인 방법
- TRUST_GATE_BLOCK 로그 확인 방법
- 운영자가 매일 확인해야 할 컬럼 및 파일 목록
- ranking 행은 제거되지 않으며 buy_eligible 컬럼으로만 구분된다는 설명

---

## 주의사항

- 실계좌 주문 로직을 직접 실행하지 말 것
- KIS API 호출을 테스트 목적으로 실행하지 말 것
- 기존 운영 파일을 삭제하지 말 것
- final_score 기존 산식과 rank 순위 산정은 변경하지 말 것
- buy_eligible=false인 종목도 ranking 행에서 절대 제거하지 말 것
- Rule 자동매매 파일은 이번 과제에서 수정하지 말 것

---

## 완료 보고 형식

1. 변경/신설 파일 목록
2. 파일별 핵심 변경 내용 요약
3. 검증 케이스 7개 실행 결과
4. 기존 출력 파일/API 호환성 확인 결과
5. 영향 범위 요약
```

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
