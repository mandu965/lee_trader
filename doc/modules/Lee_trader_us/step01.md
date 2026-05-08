# US Macro Overlay Phase 1~2 — 작업 완료 보고

작업일: 2026-05-08
작업자: Claude (Sonnet 4.6)
대상 환경: 로컬 (통신망 차단 상태) → 서버 배포 시 참고용

---

## 작업 개요

미국 ETF/지수 데이터를 수집하고 macro feature를 생성하는 인프라를 구축했습니다.
국내 자동매매(주문/랭킹/점수)에는 어떠한 영향도 없습니다.

---

## 추가 / 수정된 파일 목록

| 파일 경로 | 역할 | 신규/수정 |
|---|---|---|
| `migrations/us_macro_overlay_phase1.sql` | DB 마이그레이션 (스키마 + 테이블 생성) | 신규 |
| `python/collect_us_macro_etf_daily.py` | Phase 1: 미국 ETF/지수 일봉 수집 | 신규 |
| `python/compute_us_macro_feature_daily.py` | Phase 2: macro feature 계산 및 DB 저장 | 신규 |
| `python/run_us_macro_overlay_scheduler.py` | Phase 1+2 통합 실행 스크립트 (standalone) | 신규 |
| `.env.example` | US macro 관련 환경변수 16개 추가 | 수정 |

기존 파일은 일체 수정하지 않았습니다.

---

## 추가된 DB 테이블 DDL

마이그레이션 파일 경로: `migrations/us_macro_overlay_phase1.sql`

서버에서 실행:
```bash
psql $DATABASE_URL -f migrations/us_macro_overlay_phase1.sql
```

### 신규 스키마

```sql
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS signal;
```

### market.us_etf_daily_price

미국 ETF/지수 일봉 OHLCV 저장 (국내 주식 테이블과 완전 분리)

```sql
CREATE TABLE IF NOT EXISTS market.us_etf_daily_price (
    id           bigserial PRIMARY KEY,
    trade_date   date         NOT NULL,
    ticker       varchar(20)  NOT NULL,
    ticker_label varchar(100),
    open         numeric,
    high         numeric,
    low          numeric,
    close        numeric      NOT NULL,
    volume       bigint,
    adj_close    numeric,
    data_source  varchar(50)  NOT NULL DEFAULT 'yfinance',
    created_at   timestamptz  NOT NULL DEFAULT now(),
    updated_at   timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (trade_date, ticker)
);
```

### signal.us_macro_feature_daily

US macro feature 일별 스냅샷 (1 row per US trading date)

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

## 추가된 환경변수

`.env.example` 파일 맨 아래에 추가됨. 기본값은 모두 Shadow Mode 기준.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `US_MACRO_ENABLED` | `1` | 전체 기능 on/off 마스터 스위치 |
| `US_MACRO_SHADOW_MODE` | `1` | 1 = 수집/계산만, 실매매 미반영 |
| `US_MACRO_DATA_SOURCE` | `yfinance` | 데이터 소스 어댑터 (yfinance / polygon / alpaca) |
| `US_MACRO_TICKERS` | _(비워두면 내장 기본값 사용)_ | 수집 티커 커스텀 목록 (콤마 구분) |
| `US_MACRO_RISK_OFF_QQQ_RET` | `-0.015` | QQQ 1d 수익률 risk-off 임계값 |
| `US_MACRO_RISK_OFF_SPY_RET` | `-0.012` | SPY 1d 수익률 risk-off 임계값 |
| `US_MACRO_VIX_SPIKE_RET` | `0.10` | VIX 급등 임계값 (+10%) |
| `US_MACRO_SEMI_POSITIVE_RET` | `0.01` | 반도체 강세 임계값 |
| `US_MACRO_SEMI_NEGATIVE_RET` | `-0.01` | 반도체 약세 임계값 |
| `US_MACRO_SECTOR_BREADTH_RISK_OFF_RATIO` | `0.3` | 섹터 breadth risk-off 판단 비율 |
| `US_MACRO_SECTOR_BREADTH_RISK_ON_RATIO` | `0.6` | 섹터 breadth risk-on 판단 비율 |
| `US_MACRO_MAX_POSITIVE_ADJUST` | `5.0` | (Phase 3+) 점수 상향 조정 상한 |
| `US_MACRO_MAX_NEGATIVE_ADJUST` | `-10.0` | (Phase 3+) 점수 하향 조정 하한 |
| `US_MACRO_STALE_DAYS_LIMIT` | `3` | 데이터 신선도 경고 기준 (일) |
| `US_MACRO_ALLOW_REAL_APPLY` | `0` | Phase 3+ 실적용 허용 여부 (현재 반드시 0) |

---

## 실행 방법

### 1. 서버 초기 설정 (최초 1회)

```bash
# DB 마이그레이션
psql $DATABASE_URL -f migrations/us_macro_overlay_phase1.sql

# pip 패키지 설치 (yfinance)
pip install yfinance
```

### 2. 수동 실행

```bash
# 오늘 날짜 기준으로 Phase 1+2 실행
python python/run_us_macro_overlay_scheduler.py

# 특정 날짜 + 히스토리 30일치 수집
python python/run_us_macro_overlay_scheduler.py --date 2026-05-07 --lookback-days 30

# DB에 쓰지 않고 동작 확인만 할 때
python python/run_us_macro_overlay_scheduler.py --dry-run --lookback-days 5
```

### 3. Docker 실행

```bash
docker compose run --rm python-pipeline python python/run_us_macro_overlay_scheduler.py
```

### 4. Phase 개별 실행

```bash
# Phase 1만 (수집)
python python/collect_us_macro_etf_daily.py --lookback-days 10

# Phase 2만 (feature 계산, 이미 price 데이터가 있을 때)
python python/compute_us_macro_feature_daily.py --date 2026-05-07
```

### 5. 스케줄 권장 (미국 장 마감 후 KST 07:30)

```cron
30 7 * * 2-6 docker compose run --rm python-pipeline python python/run_us_macro_overlay_scheduler.py
```

---

## 테스트 방법

```bash
# 1. 테이블 생성 확인
psql $DATABASE_URL -c "\dt market.*"
psql $DATABASE_URL -c "\dt signal.*"

# 2. Dry-run (DB 미기록 상태로 전체 흐름 확인)
python python/run_us_macro_overlay_scheduler.py --dry-run --lookback-days 5

# 3. 실 수집 후 가격 데이터 확인
python python/run_us_macro_overlay_scheduler.py --lookback-days 5
psql $DATABASE_URL -c "
  SELECT trade_date, ticker, close
  FROM market.us_etf_daily_price
  ORDER BY trade_date DESC, ticker
  LIMIT 30;"

# 4. macro feature 결과 확인
psql $DATABASE_URL -c "
  SELECT us_trade_date, kr_apply_date,
         spy_ret_1d, qqq_ret_1d, vix_ret_1d,
         sector_breadth, top_sector,
         risk_on_flag, risk_off_flag,
         macro_status, macro_summary
  FROM signal.us_macro_feature_daily
  ORDER BY us_trade_date DESC
  LIMIT 5;"

# 5. 수집 실패 티커 확인
psql $DATABASE_URL -c "
  SELECT us_trade_date, macro_status, missing_tickers
  FROM signal.us_macro_feature_daily
  WHERE macro_status = 'DATA_INCOMPLETE'
  ORDER BY us_trade_date DESC;"
```

---

## 실제 매매 영향 여부

**이번 Phase 1~2에서는 실제 매매에 영향 없습니다.**

구체적 근거:

- 신규 스크립트 3개는 기존 파일을 import하지 않습니다.
  - `run_daily_scheduler.py` 미수정
  - `run_live_auto_trade_cycle.py` 미수정
  - `rule_signal_builder.py` 미수정
  - `signal_builder.py` 미수정
  - `ranking_builder.py` 미수정
- 신규 테이블(`market.*`, `signal.*`)은 기존 한국 주식 테이블(`public.*`, `research.*`)과 완전히 분리됩니다.
- `US_MACRO_ALLOW_REAL_APPLY=0`(기본값)이 설정된 상태에서는 실매매 경로 연결이 코드 수준에서 차단됩니다.
- 기존 `.env.example` 외 다른 기존 파일은 수정하지 않았습니다.

---

## 아키텍처 설계 포인트

### Source Adapter 패턴

`BaseUSMacroAdapter` 추상 클래스를 상속하면 데이터 소스를 교체할 수 있습니다.
현재는 `YFinanceAdapter`만 구현되어 있습니다.

```
# Polygon 어댑터 추가 예시 (Phase 이후)
class PolygonAdapter(BaseUSMacroAdapter):
    def fetch_ohlcv(self, ticker, start_date, end_date) -> pd.DataFrame:
        ...
```

`US_MACRO_DATA_SOURCE=polygon`으로 설정하면 `_get_adapter()`에서 자동 선택됩니다.

### macro_status 결정 로직

```
if missing_required_tickers:      → DATA_INCOMPLETE
elif risk_off_flag (QQQ/SPY/breadth 기준):  → RISK_OFF
elif risk_on_flag (breadth + 상승 조건):    → RISK_ON
else:                             → NEUTRAL
```

### kr_apply_date 계산

`config/trading_calendar_kr.json` 의 `closed_dates`를 참조하여
미국 거래일 다음 첫 번째 한국 개장일을 계산합니다.

---

## 다음 Phase(3)에서 해야 할 작업

1. `signal.us_macro_feature_daily`에서 당일 `kr_apply_date` 기준 row 읽기
2. `macro_status` / `risk_on_flag` / `risk_off_flag` 를 `rule_signal_builder.py` 또는 `ranking_builder.py`에 overlay 반영
3. `US_MACRO_ALLOW_REAL_APPLY=1` 설정 + 실적용 Gate 코드 추가
4. Shadow 비교 리포트 구현 (overlay 적용 전후 점수 차이 분석)
5. 점수 조정 상한(`US_MACRO_MAX_POSITIVE_ADJUST`, `US_MACRO_MAX_NEGATIVE_ADJUST`) 적용 로직 구현
6. 충분한 shadow 검증 후 Phase 4 실적용 전환

---

## 수집 대상 티커 목록

| 티커 | 이름 | 역할 |
|---|---|---|
| SPY | S&P 500 ETF | 시장 전반 (필수) |
| QQQ | NASDAQ 100 ETF | 기술주/성장주 (필수) |
| DIA | Dow Jones ETF | 경기민감주 |
| XLK | Technology Sector | 섹터 |
| XLF | Financials Sector | 섹터 |
| XLE | Energy Sector | 섹터 |
| XLV | Healthcare Sector | 섹터 |
| XLI | Industrials Sector | 섹터 |
| XLY | Consumer Discretionary Sector | 섹터 |
| XLP | Consumer Staples Sector | 섹터 |
| SMH | Semiconductor ETF | 반도체 강도 |
| ^VIX | VIX Volatility Index | 리스크 지표 |
| DX-Y.NYB | US Dollar Index | 달러 강도 |
| ^TNX | 10-Year Treasury Yield | 금리 |
