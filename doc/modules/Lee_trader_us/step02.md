# US Macro Overlay Phase 2 — 작업 완료 보고

작업일: 2026-05-08
작업자: Claude (Sonnet 4.6)
대상 환경: 로컬 (통신망 차단 상태) → 서버 배포 시 참고용
전제 조건: Phase 1 (step01.md) 완료 후 실행

---

## 작업 개요

Phase 1에서 `market.us_etf_daily_price`에 수집된 미국 ETF/지수 데이터를 읽어
한국장 매매 판단에 사용할 **macro feature를 계산하고 DB에 저장**합니다.

실제 주문, 랭킹, 점수에는 어떠한 영향도 없습니다.

---

## 추가된 파일 (Phase 2 해당 파일만)

| 파일 경로 | 역할 | 신규/수정 |
|---|---|---|
| `python/compute_us_macro_feature_daily.py` | macro feature 계산 및 DB 저장 핵심 모듈 | 신규 |

Phase 1과 통합 실행 파일:

| 파일 경로 | 역할 |
|---|---|
| `python/run_us_macro_overlay_scheduler.py` | Phase 1 수집 → Phase 2 feature 계산 연속 실행 |
| `migrations/us_macro_overlay_phase1.sql` | signal.us_macro_feature_daily 테이블 포함 |

---

## 저장 대상 테이블 DDL

```sql
CREATE TABLE IF NOT EXISTS signal.us_macro_feature_daily (
    id                          bigserial    PRIMARY KEY,
    us_trade_date               date         NOT NULL,
    kr_apply_date               date         NOT NULL,
    spy_ret_1d                  numeric,
    spy_ret_5d                  numeric,
    qqq_ret_1d                  numeric,
    qqq_ret_5d                  numeric,
    semiconductor_ret_1d        numeric,
    semiconductor_ret_5d        numeric,
    vix_ret_1d                  numeric,
    dxy_ret_1d                  numeric,
    tnx_ret_1d                  numeric,
    sector_breadth              numeric,
    top_sector                  varchar(100),
    bottom_sector               varchar(100),
    sector_strength_rank        jsonb,
    risk_on_flag                boolean,
    risk_off_flag               boolean,
    vix_spike_flag              boolean,
    semiconductor_strength_flag boolean,
    macro_status                varchar(50)  NOT NULL,
    macro_summary               text,
    data_source                 varchar(50)  NOT NULL DEFAULT 'yfinance',
    missing_tickers             text[],
    created_at                  timestamptz  NOT NULL DEFAULT now(),
    updated_at                  timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (us_trade_date)
);
```

---

## 생성되는 Feature 목록

### 수익률 Feature

| Feature | 계산식 | 소스 티커 |
|---|---|---|
| `spy_ret_1d` | (당일 종가 / 전일 종가) - 1 | SPY |
| `spy_ret_5d` | (당일 종가 / 5거래일 전 종가) - 1 | SPY |
| `qqq_ret_1d` | (당일 종가 / 전일 종가) - 1 | QQQ |
| `qqq_ret_5d` | (당일 종가 / 5거래일 전 종가) - 1 | QQQ |
| `semiconductor_ret_1d` | (당일 종가 / 전일 종가) - 1 | SMH |
| `semiconductor_ret_5d` | (당일 종가 / 5거래일 전 종가) - 1 | SMH |
| `vix_ret_1d` | (당일 종가 / 전일 종가) - 1 | ^VIX |
| `dxy_ret_1d` | (당일 종가 / 전일 종가) - 1 | DX-Y.NYB |
| `tnx_ret_1d` | (당일 종가 / 전일 종가) - 1 | ^TNX |

### 섹터 Feature

| Feature | 설명 | 사용 티커 |
|---|---|---|
| `sector_breadth` | 1d 수익률 양수인 섹터 비율 (0~1) | XLK, XLF, XLE, XLV, XLI, XLY, XLP |
| `top_sector` | 1d 수익률 1위 섹터명 | 위 7개 |
| `bottom_sector` | 1d 수익률 최하위 섹터명 | 위 7개 |
| `sector_strength_rank` | 섹터별 1d 수익률 랭킹 JSON | 위 7개 |

`sector_strength_rank` JSON 구조 예시:
```json
[
  {"sector": "Technology", "ret_1d": 0.018},
  {"sector": "Financials", "ret_1d": 0.012},
  {"sector": "Consumer Discretionary", "ret_1d": -0.005},
  ...
]
```

### 플래그 Feature

| Feature | 조건 | 환경변수 |
|---|---|---|
| `risk_off_flag` | QQQ < RISK_OFF_QQQ_RET OR SPY < RISK_OFF_SPY_RET OR breadth < BREADTH_RISK_OFF | `US_MACRO_RISK_OFF_QQQ_RET`, `US_MACRO_RISK_OFF_SPY_RET`, `US_MACRO_SECTOR_BREADTH_RISK_OFF_RATIO` |
| `risk_on_flag` | NOT risk_off AND breadth >= BREADTH_RISK_ON AND SPY >= 0 AND QQQ >= 0 | `US_MACRO_SECTOR_BREADTH_RISK_ON_RATIO` |
| `vix_spike_flag` | vix_ret_1d > VIX_SPIKE_RET | `US_MACRO_VIX_SPIKE_RET` |
| `semiconductor_strength_flag` | semi_ret_1d > SEMI_POSITIVE_RET OR semi_ret_1d < SEMI_NEGATIVE_RET | `US_MACRO_SEMI_POSITIVE_RET`, `US_MACRO_SEMI_NEGATIVE_RET` |

### 메타 Feature

| Feature | 설명 |
|---|---|
| `us_trade_date` | 미국 거래일 (데이터 기준일) |
| `kr_apply_date` | 한국 적용일 (미국 거래일 다음 한국 개장일) |
| `macro_status` | RISK_ON / RISK_OFF / NEUTRAL / DATA_INCOMPLETE |
| `macro_summary` | 인간이 읽을 수 있는 한 줄 요약 |
| `missing_tickers` | 수집 실패한 티커 목록 (array) |

---

## macro_status 결정 로직

```
필수 티커(SPY, QQQ) 누락
  → DATA_INCOMPLETE

risk_off_flag = True
  → RISK_OFF

risk_on_flag = True
  → RISK_ON

그 외
  → NEUTRAL
```

### macro_summary 출력 예시

```
US 2026-05-07 → status=RISK_OFF | SPY=-1.52% | QQQ=-2.10% | VIX=+12.30% | breadth=29% | top=Consumer Staples
```

```
US 2026-05-06 → status=RISK_ON | SPY=+0.82% | QQQ=+1.15% | VIX=-3.20% | breadth=86% | top=Technology
```

---

## kr_apply_date 계산 방식

- `config/trading_calendar_kr.json`의 `closed_dates` (한국 공휴일) 참조
- 주말(토, 일) 자동 제외
- 미국 거래일 다음 첫 번째 KRX 개장일을 `kr_apply_date`로 설정

예시:
- 미국 거래일 2026-05-07(목) → kr_apply_date = 2026-05-08(금)
- 미국 거래일 2026-05-08(금) → kr_apply_date = 2026-05-11(월)

---

## 실행 방법

### DB 마이그레이션 (최초 1회, Phase 1과 동일 파일)

```bash
psql $DATABASE_URL -f migrations/us_macro_overlay_phase1.sql
```

### Phase 2만 단독 실행

```bash
# 오늘 날짜 기준 feature 계산 (price 데이터가 이미 있어야 함)
python python/compute_us_macro_feature_daily.py

# 특정 날짜
python python/compute_us_macro_feature_daily.py --date 2026-05-07

# DB에 쓰지 않고 계산 결과만 확인
python python/compute_us_macro_feature_daily.py --dry-run
```

### Phase 1+2 연속 실행 (권장)

```bash
python python/run_us_macro_overlay_scheduler.py
python python/run_us_macro_overlay_scheduler.py --date 2026-05-07 --lookback-days 30
```

---

## 테스트 방법

```bash
# 1. 최신 feature row 전체 확인
psql $DATABASE_URL -c "
SELECT us_trade_date, kr_apply_date,
       spy_ret_1d, qqq_ret_1d, vix_ret_1d,
       sector_breadth, top_sector, bottom_sector,
       risk_on_flag, risk_off_flag,
       vix_spike_flag, semiconductor_strength_flag,
       macro_status, macro_summary,
       missing_tickers
FROM signal.us_macro_feature_daily
ORDER BY us_trade_date DESC
LIMIT 5;"

# 2. RISK_OFF 발생 이력
psql $DATABASE_URL -c "
SELECT us_trade_date, kr_apply_date, spy_ret_1d, qqq_ret_1d, vix_ret_1d,
       sector_breadth, macro_summary
FROM signal.us_macro_feature_daily
WHERE macro_status = 'RISK_OFF'
ORDER BY us_trade_date DESC;"

# 3. DATA_INCOMPLETE 이력 (수집 실패 확인)
psql $DATABASE_URL -c "
SELECT us_trade_date, macro_status, missing_tickers
FROM signal.us_macro_feature_daily
WHERE macro_status = 'DATA_INCOMPLETE' OR missing_tickers IS NOT NULL
ORDER BY us_trade_date DESC;"

# 4. 섹터 랭킹 확인
psql $DATABASE_URL -c "
SELECT us_trade_date, sector_strength_rank
FROM signal.us_macro_feature_daily
ORDER BY us_trade_date DESC
LIMIT 1;"
```

---

## 실제 매매 영향 여부

**Phase 2에서는 실제 매매에 영향 없습니다.**

- `compute_us_macro_feature_daily.py`는 `market.us_etf_daily_price`를 읽고
  `signal.us_macro_feature_daily`에만 씁니다.
- `daily_ranking`, `rule_signals`, `order` 관련 테이블은 접근하지 않습니다.
- `US_MACRO_ALLOW_REAL_APPLY=0`(기본값) 상태에서는 실매매 연결이 코드 수준에서 불가능합니다.

---

## 다음 Phase

- Phase 3 (step03.md): `signal.us_macro_feature_daily`를 읽어 RULE/AI 후보에 Shadow overlay 적용 및 `signal.kr_macro_overlay_log`에 로그 저장
