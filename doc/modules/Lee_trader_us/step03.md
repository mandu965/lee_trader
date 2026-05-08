# US Macro Overlay Phase 3 — 작업 완료 보고

작업일: 2026-05-08
작업자: Claude (Sonnet 4.6)
대상 환경: 로컬 (통신망 차단 상태) → 서버 배포 시 참고용
전제 조건: Phase 1~2 (step01.md) 완료 필요

---

## 작업 개요

Phase 1~2에서 수집된 미국 macro feature를 이용해,
한국 매수 후보(RULE / AI)에 overlay를 적용했을 경우의 **가상 결과를 로그 테이블에 저장**합니다.

실제 주문, 랭킹, 점수에는 어떠한 영향도 없습니다.

---

## 추가 / 수정된 파일 목록

| 파일 경로 | 역할 | 신규/수정 |
|---|---|---|
| `migrations/us_macro_overlay_phase3.sql` | DB 마이그레이션 (signal.kr_macro_overlay_log 생성) | 신규 |
| `python/compute_us_macro_overlay_shadow.py` | Phase 3 핵심 로직: overlay 계산 + DB 저장 | 신규 |
| `python/run_us_macro_overlay_shadow.py` | Phase 3 단독 실행 스크립트 | 신규 |

기존 파일은 일체 수정하지 않았습니다.

---

## 추가된 DB 테이블 DDL

마이그레이션 파일: `migrations/us_macro_overlay_phase3.sql`

서버에서 실행:
```bash
psql $DATABASE_URL -f migrations/us_macro_overlay_phase3.sql
```

### signal.kr_macro_overlay_log

```sql
CREATE TABLE IF NOT EXISTS signal.kr_macro_overlay_log (
    id                  bigserial    PRIMARY KEY,
    run_date            date         NOT NULL,
    us_trade_date       date,
    kr_apply_date       date         NOT NULL,
    macro_status        varchar(50),
    engine_type         varchar(20)  NOT NULL,    -- 'rule' | 'ai'
    code                varchar(10)  NOT NULL,
    name                text,
    sector              text,
    original_score      numeric,
    macro_adjustment    numeric,
    adjusted_score      numeric,
    buy_blocked_flag    boolean,
    is_buy_candidate    boolean,
    overlay_reason      text,
    shadow_mode         boolean      NOT NULL DEFAULT true,
    created_at          timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (run_date, engine_type, code)
);
```

컬럼 설명:

| 컬럼 | 설명 |
|---|---|
| `engine_type` | 'rule' = RULE 엔진 후보, 'ai' = AI 랭킹 후보 |
| `original_score` | RULE: rule_score_v2, AI: final_score |
| `macro_adjustment` | overlay로 조정된 점수 (양수/음수) |
| `adjusted_score` | original_score + macro_adjustment (0~100 범위 제한) |
| `buy_blocked_flag` | True = 이 overlay 조건에서 매수 차단 대상 |
| `is_buy_candidate` | 실제 entry_signal(RULE) 또는 top-N(AI) 여부 |
| `overlay_reason` | 적용된 룰 설명 문자열 |
| `shadow_mode` | 항상 TRUE — 실제 주문에 미반영 |

---

## 적용된 Overlay 룰 (Phase 3 초기 버전)

### Rule 1. Risk-Off Block

조건:
- `qqq_ret_1d <= US_MACRO_RISK_OFF_QQQ_RET` (기본 -0.015, -1.5%)
- `vix_ret_1d >= US_MACRO_VIX_SPIKE_RET` (기본 +0.10, +10%)

결과:
- `buy_blocked_flag = True`
- `macro_adjustment = US_MACRO_MAX_NEGATIVE_ADJUST` (기본 -10.0)
- `overlay_reason` 예시: `[RISK_OFF_BLOCK] QQQ=-2.10% ≤ -1.50%, VIX=+12.30% ≥ +10.00%`

### Rule 2. Semiconductor 가산점

조건:
- `semiconductor_ret_1d >= US_MACRO_SEMI_POSITIVE_RET` (기본 +0.01)
- `qqq_ret_1d >= 0`
- 국내 종목 sector에 "반도체" 포함

결과:
- `macro_adjustment += 3.0`
- `overlay_reason` 예시: `[SEMI_POSITIVE] 반도체섹터 + SMH=+1.50% ≥ +1.00%, QQQ=+0.80% ≥ 0`

### Rule 3. Semiconductor 감점

조건:
- `semiconductor_ret_1d <= US_MACRO_SEMI_NEGATIVE_RET` (기본 -0.01)
- 국내 종목 sector에 "반도체" 포함

결과:
- `macro_adjustment -= 5.0`
- `overlay_reason` 예시: `[SEMI_NEGATIVE] 반도체섹터 + SMH=-1.80% ≤ -1.00%`

### Rule 4. 전체 시장 보수

조건:
- `spy_ret_1d <= US_MACRO_RISK_OFF_SPY_RET` (기본 -0.012, -1.2%)

결과:
- `macro_adjustment -= 5.0`
- `overlay_reason` 예시: `[MARKET_WEAK] SPY=-1.50% ≤ -1.20%`

### Adjustment 상한/하한

- Rule 2~4 결과는 `US_MACRO_MAX_POSITIVE_ADJUST` (+5.0)와 `US_MACRO_MAX_NEGATIVE_ADJUST` (-10.0) 범위로 제한
- Rule 1 (block)은 무조건 `US_MACRO_MAX_NEGATIVE_ADJUST` 적용 (다른 룰 무시)

---

## 데이터 소스 (읽기 전용)

| 소스 | 테이블/파일 | 용도 |
|---|---|---|
| US macro | `signal.us_macro_feature_daily` | kr_apply_date 기준 최신 row |
| AI 후보 | `public.daily_ranking` | 최신 날짜 기준 top-N 종목 |
| RULE 후보 | `data/rule_signals.csv` | 최신 날짜의 entry_signal=True 종목 |

---

## Shadow Mode 실행 방법

### 전제조건 확인

```bash
# .env에서 반드시 확인
US_MACRO_ENABLED=1
US_MACRO_SHADOW_MODE=1
US_MACRO_ALLOW_REAL_APPLY=0    # Phase 3에서는 반드시 0
```

### DB 마이그레이션 (최초 1회)

```bash
# Phase 1~2 migration이 먼저 실행되어 있어야 함
psql $DATABASE_URL -f migrations/us_macro_overlay_phase3.sql
```

### 단독 실행

```bash
# 오늘 날짜 기준 실행
python python/run_us_macro_overlay_shadow.py

# 특정 날짜
python python/run_us_macro_overlay_shadow.py --kr-apply-date 2026-05-08

# DB에 쓰지 않고 로그만 확인
python python/run_us_macro_overlay_shadow.py --dry-run
```

### Docker 실행

```bash
docker compose run --rm python-pipeline python python/run_us_macro_overlay_shadow.py
```

### 전체 Phase 1~3 연속 실행

```bash
# Phase 1+2: 미국 데이터 수집 + feature 계산
python python/run_us_macro_overlay_scheduler.py

# Phase 3: shadow overlay 로그 생성
python python/run_us_macro_overlay_shadow.py
```

### 권장 스케줄 (한국 장 시작 전)

```cron
# KST 07:30 미국 데이터 수집 + feature 계산 (Phase 1+2)
30 7 * * 2-6 docker compose run --rm python-pipeline python python/run_us_macro_overlay_scheduler.py

# KST 08:50 shadow overlay 로그 생성 (Phase 3, 한국 장 시작 5분 전)
50 8 * * 2-6 docker compose run --rm python-pipeline python python/run_us_macro_overlay_shadow.py
```

---

## 생성되는 로그 예시

### 콘솔 출력 (정상 케이스, RISK_OFF 상황)

```
2026-05-08 08:50:01 [INFO] run_us_macro_overlay_shadow — ============================================================
2026-05-08 08:50:01 [INFO] run_us_macro_overlay_shadow — Phase 3: US Macro Overlay — SHADOW MODE
2026-05-08 08:50:01 [INFO] run_us_macro_overlay_shadow — ⚠ 실제 주문/랭킹/점수에는 영향을 주지 않습니다.
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — [Phase 3] Macro status for kr_apply_date=2026-05-08: RISK_OFF (us_trade=2026-05-07)
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — [Phase 3] Summary: US 2026-05-07 → status=RISK_OFF | SPY=-1.52% | QQQ=-2.10% | VIX=+12.30% | breadth=29% | top=Consumer Staples
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — [Phase 3] Candidates — AI: 30, RULE: 5
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — [Phase 3] Overlay applied: 35 rows | blocked: 35
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — ============================================================
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — ⚠ [SHADOW MODE] 이 결과는 실제 주문에 영향을 주지 않습니다.
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — US Macro Status  : RISK_OFF
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — Total candidates : 35
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — Buy blocked      : 35
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — Score adjusted + : 0
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — Score adjusted - : 0
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow — --- Blocked candidates ---
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow —   [RULE] 삼성전자 (005930) | original=78.5 → blocked | [RISK_OFF_BLOCK] QQQ=-2.10% ≤ -1.50%, VIX=+12.30% ≥ +10.00%
2026-05-08 08:50:02 [INFO] compute_us_macro_overlay_shadow —   [AI] SK하이닉스 (000660) | original=82.3 → blocked | [RISK_OFF_BLOCK] QQQ=-2.10% ≤ -1.50%, VIX=+12.30% ≥ +10.00%
```

### DB 로그 행 예시

```sql
SELECT run_date, us_trade_date, kr_apply_date,
       macro_status, engine_type, code, name, sector,
       original_score, macro_adjustment, adjusted_score,
       buy_blocked_flag, overlay_reason, shadow_mode
FROM signal.kr_macro_overlay_log
WHERE run_date = '2026-05-08'
ORDER BY engine_type, buy_blocked_flag DESC, original_score DESC;
```

| run_date | engine_type | code | name | original_score | macro_adjustment | adjusted_score | buy_blocked_flag | overlay_reason |
|---|---|---|---|---|---|---|---|---|
| 2026-05-08 | ai | 000660 | SK하이닉스 | 82.3 | -10.0 | 72.3 | true | [RISK_OFF_BLOCK] QQQ=-2.10% ≤ -1.50%, VIX=+12.30% ≥ +10.00% |
| 2026-05-08 | rule | 005930 | 삼성전자 | 78.5 | -10.0 | 68.5 | true | [RISK_OFF_BLOCK] QQQ=-2.10% ≤ -1.50%, VIX=+12.30% ≥ +10.00% |

---

## 테스트 방법

```bash
# 1. 테이블 생성 확인
psql $DATABASE_URL -c "\dt signal.*"

# 2. Dry-run (DB 미기록)
python python/run_us_macro_overlay_shadow.py --dry-run

# 3. 특정 날짜로 실행 후 결과 확인
python python/run_us_macro_overlay_shadow.py --kr-apply-date 2026-05-08

# 4. 생성된 overlay 로그 전체 확인
psql $DATABASE_URL -c "
SELECT run_date, engine_type, macro_status,
       count(*) as total,
       count(*) FILTER (WHERE buy_blocked_flag) as blocked,
       count(*) FILTER (WHERE macro_adjustment > 0) as boosted,
       count(*) FILTER (WHERE macro_adjustment < 0 AND NOT buy_blocked_flag) as penalized
FROM signal.kr_macro_overlay_log
GROUP BY run_date, engine_type, macro_status
ORDER BY run_date DESC, engine_type;"

# 5. 차단된 종목 확인
psql $DATABASE_URL -c "
SELECT engine_type, code, name, sector,
       original_score, macro_adjustment, adjusted_score,
       overlay_reason
FROM signal.kr_macro_overlay_log
WHERE run_date = CURRENT_DATE AND buy_blocked_flag = TRUE
ORDER BY engine_type, original_score DESC;"

# 6. 반도체 섹터 overlay 확인
psql $DATABASE_URL -c "
SELECT code, name, sector, original_score, macro_adjustment, overlay_reason
FROM signal.kr_macro_overlay_log
WHERE run_date = CURRENT_DATE AND sector LIKE '%반도체%'
ORDER BY macro_adjustment DESC;"
```

---

## 실제 매매 영향 여부

**Phase 3에서도 실제 매매에 영향 없습니다.**

근거:

1. `compute_us_macro_overlay_shadow.py`는 `daily_ranking` 테이블을 **읽기 전용**으로만 접근합니다.
2. `rule_signals.csv`를 **읽기 전용**으로만 접근합니다.
3. 쓰기는 `signal.kr_macro_overlay_log` 테이블에만 합니다 (기존 주문/랭킹 테이블 불변).
4. `submit_live_orders.py`, `build_trade_intents.py`, `ranking_builder.py`는 수정/import 하지 않았습니다.
5. `US_MACRO_ALLOW_REAL_APPLY=0`(기본값) 상태에서 스크립트가 실행 거부됩니다.
6. `shadow_mode = TRUE`가 모든 로그 행에 강제 저장됩니다.

---

## Phase 4에서 실반영 전 검증해야 할 항목

Phase 4 (실반영) 전환 전 반드시 아래를 확인하세요:

```text
[ ] 2~3개월 shadow 운영 후 overlay가 실제 수익률과 양의 상관관계를 보이는가?
[ ] Risk-Off block 조건이 한국 시장 하락과 실제로 일치하는가?
[ ] 반도체 overlay (+3/-5)가 반도체 종목 수익률에 유의미한 예측력을 가지는가?
[ ] US_MACRO_RISK_OFF_QQQ_RET / VIX_SPIKE_RET 임계값 최적화 백테스트 완료 여부
[ ] 미국 휴장일 처리 (macro_status=NO_DATA) 시 한국 장 운영 정책 확정
[ ] DATA_INCOMPLETE 상황에서의 fallback 정책 확정
[ ] Phase 5에서 adjusted_score를 실제 룰에 반영하는 코드 변경 범위 검토
[ ] US_MACRO_ALLOW_REAL_APPLY=1 전환 시 영향받는 코드 목록 확인
```

---

## 전체 Phase 연결 흐름

```
Phase 1~2 (run_us_macro_overlay_scheduler.py)
  → market.us_etf_daily_price (US 일봉 저장)
  → signal.us_macro_feature_daily (macro feature 생성)

Phase 3 (run_us_macro_overlay_shadow.py)
  ← signal.us_macro_feature_daily (읽기)
  ← public.daily_ranking (읽기)
  ← data/rule_signals.csv (읽기)
  → signal.kr_macro_overlay_log (로그 저장, 읽기 전용 참고용)

Phase 4~ (미구현, 실반영)
  ← signal.kr_macro_overlay_log (참고)
  → ranking_builder.py 또는 rule_signal_builder.py (수정 예정)
```
