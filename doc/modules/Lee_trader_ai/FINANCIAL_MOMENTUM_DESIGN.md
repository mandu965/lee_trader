# Financial Momentum 설계 문서

**작성일**: 2026-05-15
**상태**: 설계 확정 / 구현 미착수
**적용 대상**: Lee_trader_ai 한국 주식 추천 / 랭킹 / 자동매매

---

## 1. 목적

매출과 영업이익의 **증가 구간**을 추천 신뢰도 가산 요소로, **증가 둔화 및 감소 전환 구간**을 추천 감점·자동매매 주의 신호로 사용한다.

단순 재무 수치 표시가 아니라 **추세 모멘텀(증가·둔화·역전)을 분류하고 점수에 반영**하는 것이 핵심이다.

```text
매출 증가 + 영업이익 증가       → 실적 모멘텀 양호 → 추천 가산
증가율 둔화                     → 성장 둔화 → 감점 또는 주의
매출 + 영업이익 동시 감소       → 실적 훼손 → BUY 제한 후보
적자폭 축소 or 흑자 전환        → TURNAROUND 후보 → 별도 분류
```

최종적으로 Lee_trader는 아래 판단을 자동화한다.

```text
이 종목은 AI 점수는 높지만 실적이 꺾이고 있어 위험하다.
이 종목은 차트는 평범하지만 매출·영업이익이 동시 개선 중이어서 신뢰도가 높다.
이 종목은 실적이 나빠 보이지만 적자폭이 줄어들어 턴어라운드 후보다.
```

---

## 2. 적용 범위

### 포함

- 한국 주식 분기 재무 데이터 수집 및 저장 (`raw.financial_statement_quarterly`)
- 재무 모멘텀 feature 생성 (`feature.financial_momentum_quarterly`)
- 실적 구간 분류 (ACCELERATING / GROWING / SLOWING / WEAKENING / DECLINING / TURNAROUND)
- 점수화 (`financial_momentum_score`, `financial_risk_score`, `turnaround_score`)
- 기존 `final_score`에 overlay 방식으로 가감점 반영
- 랭킹 API, UI, acceptance report에 재무 정보 표시
- 자동매매 수량 축소 조건 (BUY gate 3단계 중 2단계)

### 제외 (초기)

- 업종별 별도 가중치 (Phase 4 이후 도입)
- 부채비율, FCF, ROE 기반 추가 스코어링 (1차 안정화 후 확장)
- 미국 주식 적용 (Project C와 별개)

---

## 3. 기존 점수 체계와의 통합 정책

### 3.1 현재 qual_score와의 중복 주의

현재 `qual_score`는 `op_margin`을 0.20 가중치로 이미 포함한다.

```text
현재 qual_score 구성 (ranking_formula.md 기준):
  roe           0.25
  op_margin     0.20  ← financial_momentum과 중복
  net_margin    0.20
  debt_ratio   -0.15
  ocf_to_assets 0.20
```

`financial_momentum_score`도 `op_margin`, `op_margin_change_yoy`를 핵심으로 사용하므로, 수준값(level)과 변화량(momentum)이 각각 반영되는 구조다. 의도적 이중 반영이지만 가중치 합산 시 동일 변수가 과도하게 반영되지 않도록 주의한다.

**결정된 통합 방식**: `financial_momentum_score`는 별도 overlay로 처리하며, `qual_score` 내 `op_margin` 가중치는 1차 안정화 전까지 현행 유지. Phase 6(백테스트) 이후 feature importance를 보고 `qual_score` 재구성 여부를 결정한다.

### 3.2 final_score 통합 방식

초기에는 기존 `final_score`를 건드리지 않고 overlay로 가감점만 부여한다.

```text
adjusted_final_score = final_score

if financial_momentum_phase == 'ACCELERATING':
    adjusted_final_score += 5
elif financial_momentum_phase == 'GROWING':
    adjusted_final_score += 3
elif financial_momentum_phase == 'SLOWING':
    adjusted_final_score -= 3
elif financial_momentum_phase == 'WEAKENING':
    adjusted_final_score -= 6
elif financial_momentum_phase == 'DECLINING':
    adjusted_final_score -= 10

if hard_fundamental_risk:
    adjusted_final_score -= 15

adjusted_final_score = clip(adjusted_final_score, 0, 100)
```

Phase 5(shadow ranking)에서 기존 rank와 adjusted rank를 비교한 후, 성과가 확인되면 가중치 방식으로 전환을 검토한다.

### 3.3 confidence_score 반영

```text
if financial_momentum_phase in ('ACCELERATING', 'GROWING'):
    confidence_score += 0.03 ~ 0.05
elif financial_momentum_phase == 'SLOWING':
    confidence_score -= 0.03
elif financial_momentum_phase == 'WEAKENING':
    confidence_score -= 0.05
elif financial_momentum_phase == 'DECLINING':
    confidence_score -= 0.08

if hard_fundamental_risk:
    confidence_score -= 0.10

confidence_score = clip(confidence_score, 0, 1)
```

---

## 4. 수집 데이터 설계

### 4.1 1차 적용 항목

```text
매출액         revenue
영업이익       operating_profit
당기순이익     net_income
```

파생 feature:

```text
revenue_yoy           매출 전년 동기 대비
op_profit_yoy         영업이익 전년 동기 대비
op_margin             영업이익률
op_margin_change_yoy  영업이익률 전년 동기 변화
```

### 4.2 2차 확장 항목 (1차 안정화 후)

```text
부채비율, 유보율, 영업활동현금흐름, FCF, ROE, ROA
```

### 4.3 수집 주기

```text
정기: 매주 1회
공시 시즌: 분기보고서·반기보고서·사업보고서 제출 기간 매일 1회
수동: 특정 종목 강제 갱신 지원
```

---

## 5. DB 스키마 설계

### 5.1 원천 재무제표 테이블

```sql
CREATE TABLE IF NOT EXISTS raw.financial_statement_quarterly (
    stock_code TEXT NOT NULL,
    corp_code  TEXT,
    company_name TEXT,

    fiscal_year INT  NOT NULL,
    quarter     TEXT NOT NULL,   -- Q1 / Q2 / Q3 / Q4 / ANNUAL
    report_code TEXT,

    revenue          NUMERIC,
    operating_profit NUMERIC,
    net_income       NUMERIC,
    total_assets     NUMERIC,
    total_liabilities NUMERIC,
    total_equity     NUMERIC,

    source_report_date DATE,   -- 재무 기준일 (분기 종료일)
    disclosed_at       DATE,   -- 실제 공시일 ← 백테스트 point-in-time 기준
    source_report_name TEXT,

    sector_code        TEXT,
    is_sector_exception BOOLEAN DEFAULT false,  -- 금융/바이오 등 특수업종

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),

    PRIMARY KEY (stock_code, fiscal_year, quarter)
);
```

> `source_report_date`(재무 기준일)와 `disclosed_at`(공시일)을 반드시 분리 저장한다.
> 백테스트에서는 `disclosed_at` 이후에만 해당 분기 feature를 사용한다.
> `is_sector_exception = true`인 종목은 momentum scoring을 별도 처리하거나 제외한다.

### 5.2 corp_code 매핑 테이블

OpenDART corp_code와 KRX stock_code는 1:1 대응이 아닐 수 있다. 지주사·자회사·상장폐지 종목을 포함한 운영 매핑 테이블을 별도 관리한다.

```sql
CREATE TABLE IF NOT EXISTS meta.dart_corp_mapping (
    stock_code   TEXT NOT NULL,
    corp_code    TEXT NOT NULL,
    corp_name    TEXT,
    listing_status TEXT,   -- LISTED / DELISTED / SUSPENDED
    is_holding_company BOOLEAN DEFAULT false,
    mapping_verified_at DATE,
    created_at   TIMESTAMP DEFAULT now(),
    updated_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (stock_code)
);
```

### 5.3 재무 모멘텀 feature 테이블

```sql
CREATE TABLE IF NOT EXISTS feature.financial_momentum_quarterly (
    stock_code TEXT NOT NULL,
    fiscal_year INT NOT NULL,
    quarter     TEXT NOT NULL,

    revenue          NUMERIC,
    operating_profit NUMERIC,
    net_income       NUMERIC,

    revenue_yoy          NUMERIC,
    revenue_qoq          NUMERIC,
    op_profit_yoy        NUMERIC,
    op_profit_qoq        NUMERIC,

    op_margin                NUMERIC,
    op_margin_change_yoy     NUMERIC,
    op_margin_change_qoq     NUMERIC,

    revenue_yoy_slowdown     BOOLEAN,
    op_profit_yoy_slowdown   BOOLEAN,
    revenue_slowdown_count   INT DEFAULT 0,
    op_profit_slowdown_count INT DEFAULT 0,

    revenue_negative_turn    BOOLEAN DEFAULT false,
    op_profit_negative_turn  BOOLEAN DEFAULT false,

    op_profit_turnaround_flag    BOOLEAN DEFAULT false,  -- 적자 → 흑자 전환
    op_profit_loss_expansion_flag BOOLEAN DEFAULT false, -- 적자폭 확대
    op_profit_loss_reduction_flag BOOLEAN DEFAULT false, -- 적자폭 축소

    financial_momentum_phase TEXT,   -- ACCELERATING/GROWING/SLOWING/WEAKENING/DECLINING/TURNAROUND
    financial_momentum_score NUMERIC,
    financial_risk_score     NUMERIC,
    turnaround_score         NUMERIC,

    fundamental_decline_flag   BOOLEAN DEFAULT false,
    hard_fundamental_risk      BOOLEAN DEFAULT false,

    is_sector_exception BOOLEAN DEFAULT false,
    disclosed_at        DATE,

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),

    PRIMARY KEY (stock_code, fiscal_year, quarter)
);
```

---

## 6. 핵심 Feature 정의

### 6.1 매출 YoY / QoQ

```text
revenue_yoy = 이번 분기 매출 / 전년 동기 매출 - 1
revenue_qoq = 이번 분기 매출 / 직전 분기 매출 - 1
```

### 6.2 영업이익 YoY / QoQ

적자·흑자 전환 시 단순 증가율 계산이 왜곡되므로 flag로 처리한다.

```text
전년 적자 → 올해 흑자:   op_profit_turnaround_flag = true, op_profit_yoy 계산 제외
전년 흑자 → 올해 적자:   op_profit_negative_turn = true, op_profit_yoy 계산 제외
전년 적자 → 적자폭 축소: op_profit_loss_reduction_flag = true
전년 적자 → 적자폭 확대: op_profit_loss_expansion_flag = true
```

### 6.3 영업이익률 및 변화

```text
op_margin            = operating_profit / revenue
op_margin_change_yoy = 이번 분기 op_margin - 전년 동기 op_margin
op_margin_change_qoq = 이번 분기 op_margin - 직전 분기 op_margin
```

### 6.4 둔화 판단

```text
revenue_yoy_slowdown   = revenue_yoy < previous_revenue_yoy
op_profit_yoy_slowdown = op_profit_yoy < previous_op_profit_yoy
revenue_slowdown_count   = 연속 둔화 분기 수 (rolling)
op_profit_slowdown_count = 연속 둔화 분기 수 (rolling)
```

---

## 7. 실적 구간 분류

### 분류 기준표

| Phase | 조건 요약 | 의미 |
|---|---|---|
| ACCELERATING | revenue_yoy > 0 AND op_profit_yoy > 0 AND 증가율 모두 개선 AND op_margin_change_yoy > 0 | 성장 가속 |
| GROWING | revenue_yoy > 0 AND op_profit_yoy > 0 AND op_margin_change_yoy >= 0 | 성장 지속 |
| SLOWING | revenue_yoy > 0 AND op_profit_yoy > 0 AND (증가율 중 하나 이상 둔화) | 성장 둔화 |
| WEAKENING | revenue_yoy > 0 AND (op_profit_yoy < 0 OR op_margin_change_yoy < -2.0) | 외형 성장, 수익성 훼손 |
| DECLINING | revenue_yoy < 0 AND op_profit_yoy < 0 | 매출·영업이익 동시 감소 |
| TURNAROUND | (YoY 기준 DECLINING이나) QoQ 개선 OR 적자폭 축소 OR 흑자 전환 | 회복 초기 가능성 |

> SLOWING 구간이 실전에서 가장 중요하다. 주가는 현재 실적 수치보다 **미래 증가율 둔화**에 먼저 반응할 수 있다.

### 분류 순서

조건 판정은 아래 순서로 적용한다. 먼저 해당하는 것이 채택된다.

```text
1. is_sector_exception → phase 계산 건너뜀 (NULL 또는 SECTOR_EXCEPTION)
2. op_profit_turnaround_flag AND revenue_qoq 개선 → TURNAROUND
3. revenue_yoy > 0 AND op_profit_yoy > 0 AND 증가율 모두 개선 → ACCELERATING
4. revenue_yoy > 0 AND op_profit_yoy > 0 AND op_margin 개선 → GROWING
5. revenue_yoy > 0 AND op_profit_yoy > 0 AND 둔화 → SLOWING
6. revenue_yoy > 0 AND (op_profit 역전 OR 마진 급락) → WEAKENING
7. revenue_yoy < 0 AND op_profit_yoy < 0 AND QoQ 개선 → TURNAROUND
8. revenue_yoy < 0 AND op_profit_yoy < 0 → DECLINING
```

---

## 8. 점수화 설계

### 8.1 financial_momentum_score (0~100, 높을수록 좋음)

```text
기본값: 50

+10  if revenue_yoy > 0
+15  if op_profit_yoy > 0
+10  if op_profit_yoy > revenue_yoy  (영업레버리지 효과)
+10  if op_margin_change_yoy > 0
-10  if revenue_yoy_slowdown
-15  if op_profit_yoy_slowdown
-15  if revenue_yoy < 0
-20  if op_profit_yoy < 0
-15  if op_margin_change_yoy < -2.0
-10  if revenue_slowdown_count >= 2
-15  if op_profit_slowdown_count >= 2

최종: clip(score, 0, 100)
```

Phase별 예상 분포:

```text
ACCELERATING: 80 ~ 100
GROWING:      65 ~ 80
SLOWING:      45 ~ 60
WEAKENING:    25 ~ 45
DECLINING:     0 ~ 30
TURNAROUND:   40 ~ 75
```

### 8.2 financial_risk_score (0~100, 높을수록 위험)

```text
기본값: 0

+20  if revenue_yoy < 0
+25  if op_profit_yoy < 0
+15  if op_margin_change_yoy < -2.0
+10  if revenue_slowdown_count >= 2
+15  if op_profit_slowdown_count >= 2
+20  if revenue_yoy < 0 AND op_profit_yoy < 0  (동시 감소 가중)
```

위험 등급:

```text
0 ~ 20   LOW
21 ~ 45  MEDIUM
46 ~ 70  HIGH
71 ~ 100 CRITICAL
```

### 8.3 turnaround_score (0~100)

```text
기본값: 0

+35  if op_profit_turnaround_flag (적자 → 흑자 전환)
+20  if op_profit_loss_reduction_flag (적자폭 축소)
+10  if revenue_qoq > 0
+20  if op_profit_qoq > 0
+15  if op_margin_change_qoq > 0
```

---

## 9. flag 정의

### 9.1 fundamental_decline_flag

```text
true 조건:
  revenue_yoy < 0
  OR op_profit_yoy < 0
  OR revenue_slowdown_count >= 2
  OR op_profit_slowdown_count >= 2
  OR op_margin_change_yoy < -2.0
```

추천 점수 감점 또는 UI 주의 표시 트리거.

### 9.2 hard_fundamental_risk

```text
true 조건:
  revenue_yoy < 0 AND op_profit_yoy < 0 AND op_margin_change_yoy < 0

  또는:
  op_profit_negative_turn = true AND revenue_yoy <= 0
```

BUY gate 제한 후보 트리거.

---

## 10. BUY Gate 설계

### 10.1 단계별 적용

```text
Phase 1 ~ 4:
  계산·표시만 수행, 매수에 영향 없음

Phase 5 (점수 overlay):
  adjusted_final_score에 반영, BUY 차단 없음

Phase 6 (백테스트 검증):
  성과 확인 후 진행 여부 판단

Phase 7 (BUY gate 적용):
  수량 축소 먼저 적용, 완전 차단은 마지막
```

### 10.2 수량 축소 (Phase 7 초기)

```text
if financial_momentum_phase == 'WEAKENING':
    order_size_multiplier = 0.7

if financial_momentum_phase == 'DECLINING':
    order_size_multiplier = 0.5

if hard_fundamental_risk:
    order_size_multiplier = 0.3
```

### 10.3 BUY 차단 조건 (Phase 7 후기, 보수적 기준)

```text
hard_fundamental_risk = true
AND confidence_score < 0.65
```

또는:

```text
financial_risk_score >= 75
AND final_score < 85
```

실적이 나쁘더라도 AI 신뢰도와 최종 점수가 충분히 높으면 차단하지 않는다.

---

## 11. 백테스트 요구사항

### 11.1 point-in-time 원칙

```text
재무제표 기준일(source_report_date)이 아닌 공시일(disclosed_at)부터 feature 사용.

예:
  2025Q1 기준일: 2025-03-31
  공시일: 2025-05-14
  → 2025-05-14 이후부터만 2025Q1 feature 사용
```

이 원칙을 어기면 백테스트 결과가 현실보다 과장된다.

### 11.2 검증 질문

```text
1. ACCELERATING / GROWING 종목의 이후 20일·60일 수익률이 유의미하게 좋은가?
2. SLOWING 구간 진입 후 주가 하락 확률이 높아지는가?
3. DECLINING 종목의 실제 하락 확률은?
4. WEAKENING 구간에서 수량 축소 시 MDD가 낮아지는가?
5. TURNAROUND 분류가 단순 DECLINING 제외보다 나은가?
```

### 11.3 비교군

```text
A: 기존 final_score만 사용
B: 기존 점수 + financial_momentum_score overlay
C: 기존 점수 + financial_risk 감점
D: 기존 점수 + hard_fundamental_risk BUY 제한
E: 기존 점수 + 수량 축소
```

평가 지표: 수익률, 승률, MDD, 상위 10종목 품질, BUY 차단 과도 여부.

---

## 12. UI / 리포트 표시 설계

### 12.1 종목 상세 표시

```text
실적 모멘텀: GROWING
매출 YoY: +12.4%
영업이익 YoY: +28.7%
영업이익률: 14.2%
영업이익률 변화: +2.1%p
재무 위험도: LOW

해석:
매출과 영업이익이 모두 증가하고 있으며,
영업이익 증가율이 매출보다 높아 수익성 개선 구간으로 판단됩니다.
추천 점수에 +3점 반영되었습니다.
```

위험 구간 표시:

```text
실적 모멘텀: DECLINING
매출 YoY: -7.8%
영업이익 YoY: -31.5%
영업이익률 변화: -4.2%p
재무 위험도: CRITICAL

해석:
매출과 영업이익이 동시에 감소하고 있습니다.
실적 모멘텀 훼손 구간으로 BUY gate 제한 후보입니다.
```

### 12.2 acceptance report 추가 섹션

```text
## 재무 모멘텀 점검

실적 가산 종목 수: N
실적 감점 종목 수: N
hard_fundamental_risk 종목 수: N
TURNAROUND 후보 수: N

[종목별 상세]
종목명: OOO
phase: GROWING
revenue_yoy: +8.2%
op_profit_yoy: +15.6%
op_margin_change_yoy: +1.4%p
score_adjustment: +3
reason: 매출·영업이익 증가, 수익성 개선
```

---

## 13. 환경변수

| 변수명 | 기본값 | 설명 | 안전 주의 |
|---|---|---|---|
| `FINANCIAL_FEATURE_ENABLED` | `0` | 재무 feature 계산 활성화 | - |
| `FINANCIAL_SHADOW_MODE` | `1` | shadow 모드 (계산만, 매수 미반영) | - |
| `FINANCIAL_SCORE_OVERLAY_ENABLED` | `0` | final_score overlay 적용 여부 | 랭킹 변동 주의 |
| `FINANCIAL_BUY_GATE_ENABLED` | `0` | BUY gate / 수량 축소 활성화 | 매수 차단 주의 |
| `FINANCIAL_WEIGHT` | `0.10` | 가중치 방식 전환 시 비중 | - |
| `FINANCIAL_HARD_RISK_BUY_BLOCK` | `0` | hard_fundamental_risk 시 BUY 차단 | 매수 차단 주의 |
| `FINANCIAL_RISK_SIZE_REDUCTION_ENABLED` | `0` | 수량 축소 활성화 여부 | 주문량 변동 |
| `FINANCIAL_OP_MARGIN_DROP_THRESHOLD` | `-2.0` | 영업이익률 급락 기준 (%p) | - |
| `FINANCIAL_SLOWDOWN_CONSECUTIVE_QUARTERS` | `2` | 둔화 연속 분기 기준 | - |
| `FINANCIAL_HARD_RISK_THRESHOLD` | `70` | financial_risk_score 위험 임계값 | - |
| `FINANCIAL_DATA_SOURCE` | `opendart` | 수집 데이터 소스 | - |
| `FINANCIAL_REFRESH_DAYS` | `7` | 정기 수집 주기 (일) | - |

운영 단계별 설정:

```env
# Phase 1~4: shadow 관찰
FINANCIAL_FEATURE_ENABLED=1
FINANCIAL_SHADOW_MODE=1
FINANCIAL_SCORE_OVERLAY_ENABLED=0
FINANCIAL_BUY_GATE_ENABLED=0

# Phase 5: overlay 적용
FINANCIAL_SCORE_OVERLAY_ENABLED=1

# Phase 7: BUY gate 적용
FINANCIAL_BUY_GATE_ENABLED=1
FINANCIAL_RISK_SIZE_REDUCTION_ENABLED=1
```

---

## 14. 개발 단계 (Phase)

| Phase | 목표 | 핵심 산출물 | 매수 영향 |
|---|---|---|---|
| 1 | OpenDART 분기 재무 데이터 수집 | `raw.financial_statement_quarterly`, `meta.dart_corp_mapping` | 없음 |
| 2 | 재무 feature 생성 | `feature.financial_momentum_quarterly` | 없음 |
| 3 | 실적 구간 분류 및 점수 계산 | phase, momentum_score, risk_score, turnaround_score | 없음 |
| 4 | 랭킹 API / UI / report 표시 | ranking 응답 payload 확장, UI 항목 추가 | 없음 |
| 5 | shadow ranking 비교 | `shadow_financial_adjusted_rank`, `rank_diff` | 없음 |
| 6 | 백테스트 검증 | 20d/60d/120d 수익률, MDD, 승률 비교 | 없음 |
| 7 | 점수 overlay 적용 | `adjusted_final_score` 운영 반영 | 랭킹 변동 |
| 8 | 수량 축소 / BUY gate 적용 | `order_size_multiplier`, buy_gate_status | 주문량 변동 |

---

## 15. 업종 예외 처리

아래 업종은 `is_sector_exception = true`로 표시하고, Phase 4 이전에 처리 정책을 별도 정의한다.

```text
금융업 (은행, 보험, 증권):
  매출/영업이익 개념이 다름
  → momentum scoring 제외 또는 별도 logic 적용

바이오/제약:
  파이프라인 가치가 재무보다 중요
  → momentum score 반영 비중 축소

지주사:
  연결 실적 왜곡 가능성
  → 별도 검토

건설:
  수주잔고 중요, 매출 인식 시점 왜곡 가능
  → QoQ 판단 시 주의
```

---

## 16. 핵심 주의사항

### 16.1 공시일 기준 준수

백테스트와 실운영 모두 `disclosed_at` 이후에만 해당 분기 feature를 사용한다. `source_report_date` 기준으로 적용하면 미래 정보 누수가 발생한다.

### 16.2 단순 감소 = DECLINING이 아닌 경우

```text
매출 YoY -3%, 영업이익 YoY -5%이지만
QoQ 개선, 마진 개선, 거래량 증가, 52주 저점 근처
→ TURNAROUND 후보로 분류
```

무조건 제외하지 않고 별도 분류해서 판단한다.

### 16.3 연속 둔화가 단회성보다 위험

`*_slowdown_count >= 2`는 일시적 노이즈가 아닌 추세 전환 신호다. 단회성 둔화보다 높은 감점을 적용한다.

---

## 17. 기존 개선과제와의 관계

[20260515_개선과제.md](20260515_개선과제.md)의 A-3(YoY 성장률 피처)와 본 설계의 관계:

```text
A-3: quality_builder.py에서 연간 YoY 계산 (revenue_growth_yoy, op_income_growth_yoy)
     → 모델 재학습 전 빠르게 반영 가능 (DART 연간 데이터 기존 수집 중)

본 설계 Phase 1~3: 분기 단위 더 정밀한 모멘텀 계산
     → 인프라 구축 소요, 별도 DART 분기 수집 필요

권장 순서:
  A-3 연간 YoY → 모델 재학습에 즉시 반영
  본 설계 Phase 1 → A-3 이후 병렬 진행
  Phase 4 이후 → 모델 재학습 재료로 분기 데이터 추가
```

---

## 18. 연관 문서

- [ranking_formula.md](../../python/ranking_formula.md) — 현재 final_score 산식 (overlay 통합 기준)
- [DB_SCHEMA.md](DB_SCHEMA.md) — US 재무 스키마 참고 (한국 설계 시 일관성 유지)
- [CONTEXT.md](CONTEXT.md) — 운영 경계 및 모듈 분리 원칙
- [20260515_개선과제.md](20260515_개선과제.md) — A-3 YoY 피처 관련 우선순위
- [feature_dictionary.md](../../feature_dictionary.md) — feature 명칭 규칙
