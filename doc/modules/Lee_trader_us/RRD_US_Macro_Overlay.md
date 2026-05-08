# RRD — 미국 매크로 신호 기반 국내 RULE 강화

**Project**: Lee_trader Project B  
**Document Type**: RRD (Requirements / Roadmap / Risk Design)  
**Feature Name**: US Macro Overlay for KR RULE  
**Version**: v1.0 Final  
**Date**: 2026-05-08  
**Status**: 설계 확정 / Phase 1~2 개발 착수 가능  

---

## 0. 문서 목적

이 문서는 `PRD_B_US_Macro_Overlay.docx`와 회의 중 정리한 설계안을 비교하여 정리한 최종 RRD 문서다.

이 문서의 목적은 다음과 같다.

1. 미국 시장의 야간 흐름을 국내 자동매매 판단에 보조 신호로 반영하는 기능의 요구사항을 확정한다.
2. 실제 매매에 바로 반영하지 않고 Shadow Mode부터 시작하는 안전한 개발 순서를 정의한다.
3. DB, 환경변수, 스케줄러, 로그, UI, 알림, 백테스트, 실반영 기준을 하나의 실행 가능한 문서로 통합한다.
4. 향후 미국주식 자동추천 v1(Project C)로 확장할 수 있는 선행 인프라를 만든다.

---

## 1. 최종 결론

미국주식 자동매매를 바로 붙이는 것보다, 먼저 미국 시장 데이터를 국내 자동매매의 보조 신호로 활용하는 것이 더 현실적이다.

이번 Project B는 미국주식 실매매 프로젝트가 아니다.

핵심 목적은 다음과 같다.

```text
미국 야간 시장 데이터
→ ETF / 지수 / 리스크 지표 수집
→ macro feature 생성
→ 국내 RULE / AI 추천 점수 보조 보정
→ 다음날 한국장 매매 판단 강화
```

최종 결론은 다음과 같다.

```text
1. 미국 매크로 신호 기반 국내 RULE 강화 작업을 먼저 진행한다.
2. 실제 매매 반영은 금지하고, Phase 1~3은 반드시 Shadow Mode 중심으로 진행한다.
3. 미국 신호는 주 신호가 아니라 보조 신호로만 사용한다.
4. 룰은 3~5개만 단순하게 시작한다.
5. 로그 / UI / 알림에서 매수 차단 사유를 반드시 설명 가능하게 만든다.
6. 백테스트와 Shadow 운영 결과가 좋아야만 제한적으로 실반영한다.
7. 이후 미국주식 자동추천 v1(Project C)로 확장한다.
```

---

## 2. 프로젝트 범위

### 2-1. 포함 범위

이번 RRD의 포함 범위는 다음과 같다.

| 구분 | 내용 |
|---|---|
| 미국 ETF / 지수 데이터 수집 | SPY, QQQ, SMH, XLK, XLF, XLE, XLV, VIX, DXY, TNX 등 |
| Macro Feature 생성 | 수익률, 섹터 강도, VIX 급등, Risk-On / Risk-Off 판단 |
| 국내 RULE 보조 적용 | 점수 보정, 신규매수 차단 후보, 섹터별 가산 / 감점 |
| Shadow Mode | 실제 주문 영향 없이 적용 결과만 기록 |
| 로그 / UI / 알림 | 왜 매수 차단 또는 점수 보정이 발생했는지 설명 |
| 백테스트 검증 | 기존 전략 대비 overlay 적용 전략 비교 |
| 제한적 실반영 기준 | 검증 완료 후 일부 조건만 실제 RULE에 반영 |

### 2-2. 제외 범위

이번 RRD에서 제외하는 범위는 다음과 같다.

| 제외 항목 | 사유 |
|---|---|
| 미국주식 실매매 | 환율, 세금, 브로커, 체결 리스크가 크므로 제외 |
| 미국 개별 종목 추천 | Project C에서 별도 설계 |
| 미국 Paper Trading | Project C 이후 별도 진행 |
| 포지션 자동 확대 | 초기에는 리스크 방어 중심으로만 진행 |
| 복잡한 AI 모델 재학습 | v1에서는 rule-based overlay 중심 |

---

## 3. Project B와 Project C의 관계

이번 Project B는 미국주식 자동추천 v1(Project C)과 경쟁하는 구조가 아니다.

역할은 다음과 같이 분리한다.

| 구분 | Project B — US Macro Overlay | Project C — 미국주식 자동추천 v1 |
|---|---|---|
| 목적 | 국내 자동매매 보조 신호 강화 | 미국 종목 자체 추천 |
| 대상 | ETF / 지수 / 리스크 지표 | Nasdaq100 / S&P500 / ETF / 개별 종목 |
| 적용 시장 | 한국 주식 자동매매 | 미국 주식 추천 / Paper Trading |
| 실매매 리스크 | 낮음, 직접 주문 없음 | 높음, 주문 / 환율 / 세금 고려 필요 |
| 개발 순서 | 먼저 진행 | Project B 안정화 후 진행 |
| 인프라 관계 | 선행 인프라 구축 | B 인프라 재사용 및 확장 |

최종 구조는 다음과 같다.

```text
[미국 시장 데이터]
   ├─ Project B: 한국 시스템 강화용 macro overlay
   │      → 국내 매수 / 매도 룰 보조 보정
   │      → 현재 시스템 개선
   │
   └─ Project C: 미국 자동추천 v1
          → 미국 종목 랭킹
          → 백테스트 / Paper Trading
          → 미래 미국 실매매 확장
```

---

## 4. 우선순위

최종 우선순위는 다음과 같다.

| 순위 | 과제 | 시점 | 비고 |
|---|---|---|---|
| 1 | 미국 매크로 신호로 국내 자동매매 강화 | 즉시 | 이 문서의 범위 |
| 2 | 미국주식 자동추천 v1 설계 | Project B 운영 후 병행 | Project C |
| 3 | 미국주식 Paper Trading | 중기 | 실매매 전 검증 |
| 4 | 미국 실매매 검토 | 장기 | 가장 마지막 |

---

## 5. 전체 아키텍처

```text
[미국 시장 데이터 수집]
        ↓
[미국 ETF / 지수 / 리스크 데이터 저장]
        ↓
[Macro Feature 계산]
        ↓
[Risk-On / Risk-Off / Sector Strength 판단]
        ↓
[국내 추천 / RULE 후보에 Overlay 시뮬레이션]
        ↓
[Shadow Mode 로그 저장]
        ↓
[UI / 메일 / 운영 로그 표시]
        ↓
[백테스트 검증]
        ↓
[검증된 조건만 제한적 실반영]
```

---

## 6. 핵심 설계 원칙

### 6-1. 미국 신호는 주 신호가 아니다

미국 매크로 신호는 국내 종목을 직접 매수하게 만드는 주 신호가 아니다.

역할은 다음과 같다.

```text
기존 국내 추천 / RULE 판단 결과
+ 미국 macro overlay
= 보정된 판단 후보
```

즉, 미국 신호만으로 신규 매수를 만들지 않는다.

---

### 6-2. 감점과 차단을 가산보다 우선한다

이 기능의 핵심 목적은 수익 극대화보다 리스크 방어다.

따라서 초기에는 다음 원칙을 따른다.

```text
가산 룰: 보수적으로 적용
감점 룰: 명확한 경우 적용
차단 룰: 매우 강한 Risk-Off에서만 적용
```

---

### 6-3. Shadow Mode가 기본값이다

초기 환경변수 기본값은 반드시 다음과 같아야 한다.

```env
US_MACRO_ENABLED=true
US_MACRO_SHADOW_MODE=true
```

`US_MACRO_SHADOW_MODE=true` 상태에서는 실제 주문 후보, 주문 수량, 주문 실행 로직에 영향을 주면 안 된다.

---

### 6-4. 설명 가능성이 필수다

매수 차단 또는 점수 보정이 발생하면 반드시 이유가 남아야 한다.

예시:

```text
Risk-Off 판단:
QQQ -2.6%, VIX +12.4%, Sector Breadth 1/8
→ 신규 BUY 차단 후보 발생
→ Shadow Mode, 실제 주문 영향 없음
```

---

## 7. 데이터 수집 대상

### 7-1. v1 우선 수집 목록

개발 난이도와 안정성을 고려하여 v1에서는 다음 10개 중심으로 시작한다.

| 구분 | 티커 | 설명 | 우선순위 |
|---|---|---|---|
| 대표 지수 | SPY | S&P500 ETF | 필수 |
| 대표 지수 | QQQ | Nasdaq100 ETF | 필수 |
| 반도체 | SMH | VanEck Semiconductor ETF | 필수 |
| 기술주 | XLK | Technology Select Sector SPDR | 필수 |
| 금융 | XLF | Financial Select Sector SPDR | 필수 |
| 에너지 | XLE | Energy Select Sector SPDR | 필수 |
| 헬스케어 | XLV | Health Care Select Sector SPDR | 필수 |
| 변동성 | ^VIX | VIX Index | 필수 |
| 달러 | DX-Y.NYB | Dollar Index 대체 | 선택 / 검증 필요 |
| 금리 | ^TNX | 10Y Treasury Yield | 선택 / 검증 필요 |

### 7-2. 확장 수집 후보

| 티커 | 설명 | 활용 |
|---|---|---|
| DIA | Dow Jones ETF | 대형 우량주 분위기 |
| XLI | 산업재 | 조선 / 기계 / 산업재 참고 |
| XLY | 경기소비재 | 자동차 / 소비재 참고 |
| XLP | 필수소비재 | 방어주 참고 |
| SOXX | 반도체 ETF 대체 | SMH 대체 또는 보완 |

### 7-3. 티커 주의사항

VIX, DXY, TNX는 데이터 소스별 티커가 다를 수 있다.

v1에서는 yfinance 기준으로 다음 후보를 검증한다.

```text
VIX: ^VIX
DXY: DX-Y.NYB
TNX: ^TNX
```

티커 조회 실패 시 해당 지표는 feature 생성에서 제외하되, 전체 수집이 실패하지 않도록 한다.

---

## 8. Macro Feature 요구사항

### 8-1. 시장 방향성 Feature

| Feature | 설명 | 활용 |
|---|---|---|
| spy_ret_1d | SPY 1일 수익률 | 전체 시장 분위기 |
| spy_ret_5d | SPY 5일 수익률 | 단기 추세 |
| qqq_ret_1d | QQQ 1일 수익률 | 성장주 / 기술주 분위기 |
| qqq_ret_5d | QQQ 5일 수익률 | 성장주 단기 추세 |

판단 예시:

```text
QQQ 1일 수익률 <= -2.0%
→ 다음날 국내 성장주 신규매수 보수화

SPY 1일 수익률 <= -1.5%
→ 전체 신규매수 점수 감점 후보
```

---

### 8-2. 반도체 Feature

| Feature | 설명 | 활용 |
|---|---|---|
| semiconductor_ret_1d | SMH 또는 SOXX 1일 수익률 | 국내 반도체 가산 / 감점 |
| semiconductor_ret_5d | SMH 또는 SOXX 5일 수익률 | 반도체 단기 추세 |
| semiconductor_strength_flag | 반도체 강세 여부 | 섹터 룰 트리거 |

판단 예시:

```text
SMH 1일 수익률 >= +1.5%
AND QQQ 1일 수익률 >= 0%
→ 국내 반도체 후보 +3점

SMH 1일 수익률 <= -2.5%
→ 국내 반도체 후보 -5점
```

---

### 8-3. 섹터 강도 Feature

| Feature | 설명 | 예시 |
|---|---|---|
| top_sector | 당일 가장 강한 섹터 | semiconductor |
| bottom_sector | 당일 가장 약한 섹터 | financial |
| sector_breadth | 상승 섹터 수 / 전체 섹터 수 | 6/8 |
| sector_strength_rank | 섹터별 강도 순위 | XLK 1위, XLF 최하위 |

판단 예시:

```text
주요 섹터 ETF 8개 중 6개 상승
→ risk-on 후보

주요 섹터 ETF 8개 중 6개 이상 하락
→ risk-off 후보
```

---

### 8-4. 리스크 Feature

| Feature | 설명 | 활용 |
|---|---|---|
| vix_ret_1d | VIX 1일 변화율 | 변동성 급등 판단 |
| vix_spike_flag | VIX 급등 여부 | Risk-Off 차단 룰 |
| dxy_ret_1d | DXY 1일 변화율 | 환율 / 외국인 수급 부담 참고 |
| tnx_ret_1d | 미국 10년물 금리 변화율 | 성장주 부담 참고 |
| risk_off_flag | 종합 Risk-Off 여부 | 신규매수 차단 후보 |
| risk_on_flag | 종합 Risk-On 여부 | 제한적 가산 후보 |

---

## 9. 국내 시스템 반영 방식

초기 반영 방식은 다음 3가지다.

```text
A. 점수 보정
B. 신규매수 차단 후보 생성
C. 섹터별 가산 / 감점
```

초기 개발에서는 실제 주문 반영이 아니라 `Shadow Mode 로그 생성`만 수행한다.

---

### 9-1. 방식 A — 점수 보정

기존 국내 점수에 macro adjustment를 더해 adjusted_score를 계산한다.

예시:

```text
base_score = 78
macro_adjustment = -5
adjusted_score = 73
```

초기 보정 범위는 다음과 같다.

| 구분 | 값 |
|---|---|
| 최대 가산 | +5 |
| 최대 감점 | -10 |

---

### 9-2. 방식 B — 신규매수 차단 후보

강한 Risk-Off 조건에서는 신규매수 차단 후보를 기록한다.

단, Shadow Mode에서는 실제 주문 로직에 영향을 주지 않는다.

예시:

```text
QQQ <= -2.5%
AND VIX >= +10%
→ buy_blocked_flag = 'Y'
→ 실제 주문 영향 없음
```

---

### 9-3. 방식 C — 섹터별 가산 / 감점

미국 섹터 ETF와 국내 섹터를 매핑하여 일부 종목군만 보정한다.

| 미국 신호 | 국내 반영 대상 | 조치 |
|---|---|---|
| SMH / SOXX 강세 | 반도체 / 반도체 장비 / 소부장 | +3 |
| SMH / SOXX 약세 | 반도체 / 반도체 장비 / 소부장 | -5 |
| XLK 강세 | IT / 소프트웨어 / 성장주 | +2 |
| XLE 강세 | 정유 / 에너지 / 화학 | +2 |
| XLV 강세 | 제약 / 바이오 | +1 |
| XLI 강세 | 조선 / 기계 / 산업재 | +1 |
| XLF 약세 | 금융주 | -3 |

초기에는 반도체 섹터만 우선 연결해도 된다.

---

## 10. 초기 Rule 정의

아래 임계값은 초기 제안값이다.

Phase 4 백테스트에서 백분위수와 성과를 확인한 뒤 조정한다.

---

### 10-1. Risk-Off 차단 룰

```text
조건:
QQQ 1일 수익률 <= -2.5%
AND VIX 1일 상승률 >= +10%

Shadow 결과:
buy_blocked_flag = 'Y'
macro_adjustment = -10
overlay_reason = 'QQQ 급락 및 VIX 급등으로 Risk-Off 차단 후보'
```

---

### 10-2. 전체 시장 보수 룰

```text
조건:
SPY 1일 수익률 <= -1.8%
AND 주요 섹터 ETF 중 70% 이상 하락

Shadow 결과:
macro_adjustment = -5
buy_blocked_flag = 조건 강도에 따라 Y 또는 N
```

---

### 10-3. 반도체 가산 룰

```text
조건:
SMH 1일 수익률 >= +1.5%
AND QQQ 1일 수익률 >= 0%
AND 국내 종목이 반도체 섹터

Shadow 결과:
macro_adjustment = +3
```

---

### 10-4. 반도체 감점 룰

```text
조건:
SMH 1일 수익률 <= -2.5%
AND 국내 종목이 반도체 섹터

Shadow 결과:
macro_adjustment = -5
```

---

### 10-5. Risk-On 완화 룰

```text
조건:
SPY 1일 수익률 >= +1.0%
AND QQQ 1일 수익률 >= +1.0%
AND VIX 하락

Shadow 결과:
우량 Top 후보에 한해 macro_adjustment = +2
```

이 룰은 보수적으로 사용한다.

---

## 11. DB 설계

프로젝트의 기존 DB 네이밍 규칙에 맞춰 조정할 수 있다.

다만 국내 주식 데이터와 미국 매크로 데이터는 반드시 분리한다.

권장 schema는 다음과 같다.

```text
market.us_etf_daily_price
signal.us_macro_feature_daily
signal.kr_macro_overlay_log
```

---

### 11-1. 미국 ETF / 지수 일봉 테이블

```sql
CREATE TABLE IF NOT EXISTS market.us_etf_daily_price (
    trade_date        DATE          NOT NULL,
    ticker            VARCHAR(20)   NOT NULL,
    name              VARCHAR(100),
    open_price        NUMERIC(18,4),
    high_price        NUMERIC(18,4),
    low_price         NUMERIC(18,4),
    close_price       NUMERIC(18,4),
    adj_close_price   NUMERIC(18,4),
    volume            BIGINT,
    data_source       VARCHAR(50),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, ticker)
);
```

---

### 11-2. 미국 Macro Feature 테이블

`us_trade_date`와 `kr_apply_date`는 반드시 분리한다.

예시:

```text
미국 2026-05-07 장 마감 데이터
→ 한국 2026-05-08 장 시작 전 반영
```

```sql
CREATE TABLE IF NOT EXISTS signal.us_macro_feature_daily (
    us_trade_date               DATE     NOT NULL,
    kr_apply_date               DATE     NOT NULL,

    spy_ret_1d                  NUMERIC(10,4),
    spy_ret_5d                  NUMERIC(10,4),
    qqq_ret_1d                  NUMERIC(10,4),
    qqq_ret_5d                  NUMERIC(10,4),

    semiconductor_ret_1d        NUMERIC(10,4),
    semiconductor_ret_5d        NUMERIC(10,4),

    vix_ret_1d                  NUMERIC(10,4),
    dxy_ret_1d                  NUMERIC(10,4),
    tnx_ret_1d                  NUMERIC(10,4),

    sector_breadth              NUMERIC(10,4),
    top_sector                  VARCHAR(50),
    bottom_sector               VARCHAR(50),
    sector_strength_rank        TEXT,

    risk_on_flag                CHAR(1),
    risk_off_flag               CHAR(1),
    vix_spike_flag              CHAR(1),
    semiconductor_strength_flag CHAR(1),

    macro_status                VARCHAR(50),
    macro_summary               TEXT,

    data_source                 VARCHAR(50),
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (us_trade_date, kr_apply_date)
);
```

---

### 11-3. 국내 RULE 반영 로그 테이블

이 테이블은 운영 설명력의 핵심이다.

다음 질문에 답할 수 있어야 한다.

```text
왜 이 종목은 매수되지 않았는가?
왜 점수가 낮아졌는가?
미국 신호가 어떤 영향을 줬는가?
실제 반영이었는가, Shadow Mode였는가?
```

```sql
CREATE TABLE IF NOT EXISTS signal.kr_macro_overlay_log (
    kr_apply_date      DATE         NOT NULL,
    us_trade_date      DATE         NOT NULL,

    stock_code         VARCHAR(20)  NOT NULL,
    stock_name         VARCHAR(100),
    sector_name        VARCHAR(100),
    theme_name         VARCHAR(100),

    base_score         NUMERIC(10,4),
    macro_adjustment   NUMERIC(10,4),
    adjusted_score     NUMERIC(10,4),

    overlay_type       VARCHAR(50),
    overlay_reason     TEXT,

    risk_off_flag      CHAR(1),
    buy_blocked_flag   CHAR(1),
    shadow_mode_flag   CHAR(1),

    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (kr_apply_date, stock_code)
);
```

---

## 12. 환경변수 설계

`.env.example`에 다음 항목을 추가한다.

```env
# =========================================================
# US Macro Overlay for KR RULE
# =========================================================

# 기능 ON/OFF
US_MACRO_ENABLED=true
US_MACRO_SHADOW_MODE=true

# 데이터 소스
US_MACRO_DATA_SOURCE=yfinance
US_MACRO_TICKERS=SPY,QQQ,DIA,XLK,XLF,XLE,XLV,XLI,XLY,XLP,SMH,^VIX,DX-Y.NYB,^TNX

# 임계값 — 백테스트 후 조정
US_MACRO_RISK_OFF_QQQ_RET=-0.025
US_MACRO_RISK_OFF_SPY_RET=-0.018
US_MACRO_VIX_SPIKE_RET=0.10
US_MACRO_SEMI_POSITIVE_RET=0.015
US_MACRO_SEMI_NEGATIVE_RET=-0.025

# 점수 보정 범위
US_MACRO_MAX_POSITIVE_ADJUST=5
US_MACRO_MAX_NEGATIVE_ADJUST=-10

# Stale 데이터 처리
US_MACRO_STALE_DAYS_LIMIT=2

# 섹터 breadth 기준
US_MACRO_SECTOR_BREADTH_RISK_OFF_RATIO=0.30
US_MACRO_SECTOR_BREADTH_RISK_ON_RATIO=0.70

# 실제 반영 안전장치
US_MACRO_ALLOW_REAL_APPLY=false
```

운영 원칙:

```text
US_MACRO_SHADOW_MODE=true 이면 실제 주문 영향 없음
US_MACRO_ALLOW_REAL_APPLY=false 이면 실반영 절대 금지
```

---

## 13. 스케줄러 설계

한국 시간 기준으로 미국장은 야간에 끝난다.

따라서 한국장 시작 전까지 수집과 feature 생성을 완료해야 한다.

| 시간대(KST) | 작업 | 비고 |
|---|---|---|
| 06:00 ~ 07:00 | 미국 ETF / 지수 데이터 수집 | 비서머타임은 07:00 이후 안정적 |
| 07:00 ~ 07:20 | us_macro_feature_daily 생성 | feature 계산 및 DB 저장 |
| 07:20 ~ 08:00 | 국내 추천 / RULE 후보 overlay 시뮬레이션 | Shadow Mode 로그만 기록 |
| 08:00 ~ 08:30 | 당일 후보 / 차단 사유 생성 | UI / 메일 알림 포함 |
| 09:00 | 한국장 시작 | 실매매 전 판단 완료 |

서머타임 고려:

```text
미국 서머타임: 미국장 마감 KST 06:00
미국 비서머타임: 미국장 마감 KST 07:00

v1에서는 KST 07:00 이후 수집을 기본으로 한다.
v2에서 서머타임 자동 처리 및 장마감 완료 검증을 추가한다.
```

---

## 14. 개발 파일 구조

### 14-1. 권장 신규 파일 구조

미국 매크로 신호는 향후 미국 자동추천 v1과 연결될 수 있으므로 독립 모듈로 분리한다.

```text
python/
  collector/
    collect_us_macro_etf_daily.py
  features/
    compute_us_macro_feature_daily.py
  signal/
    apply_us_macro_overlay_to_kr.py
  scheduler/
    run_us_macro_overlay_scheduler.py

doc/modules/Lee_trader_us_macro/
  README.md
  CONTEXT.md
  FLOW.md
  ENV.md
  RRD.md
```

### 14-2. 기존 구조 최소 확장 방식

빠른 시작이 필요한 경우 다음 파일을 확장할 수 있다.

```text
compute_theme_etf_daily.py
signal_builder.py
run_daily_scheduler.py
schema.sql
.env.example
```

다만 장기적으로는 독립 모듈 분리를 권장한다.

---

## 15. Phase별 개발 계획

| Phase | 목표 | 기간 | 완료 기준 |
|---|---|---|---|
| Phase 1 | 미국 ETF / 지수 데이터 수집 | 1~2주 | 최근 1년치 적재 + 매일 자동 갱신 |
| Phase 2 | Macro Feature 생성 | 1주 | risk_on/off/semiconductor 플래그 생성 |
| Phase 3 | Shadow Mode 적용 | 1~2주 | 실제 주문 영향 없이 overlay 로그 확인 |
| Phase 4 | 백테스트 검증 | 2~4주 | 기존 전략 대비 MDD 개선 여부 확인 |
| Phase 5 | 제한적 실반영 | 검증 후 | 검증된 조건만 RULE에 반영 |

---

## 16. Phase 1 상세 — 미국 ETF / 지수 데이터 수집

### 목표

```text
미국 ETF / 지수 일봉 데이터를 매일 수집한다.
```

### 작업 항목

```text
1. 수집 대상 티커 목록 정의
2. market.us_etf_daily_price 테이블 생성
3. yfinance 기반 일봉 수집 구현
4. 중복 저장 방지 upsert 구현
5. 수집 실패 로그 기록
6. 티커별 조회 실패가 전체 실패로 번지지 않도록 처리
7. 최근 1년치 초기 적재 기능 구현
```

### 완료 기준

```text
최근 1년치 SPY, QQQ, SMH, XLK 등 데이터 적재 가능
매일 아침 자동 갱신 가능
수집 실패 시 로그 확인 가능
실제 주문 영향 없음
```

---

## 17. Phase 2 상세 — Macro Feature 생성

### 목표

```text
수집된 미국 데이터로 risk-on / risk-off / sector strength feature를 만든다.
```

### 작업 항목

```text
1. 1일 / 5일 수익률 계산
2. 섹터 ETF 상승 개수 계산
3. sector_breadth 계산
4. VIX 급등 여부 계산
5. 반도체 강세 여부 계산
6. risk_on_flag 계산
7. risk_off_flag 계산
8. us_macro_feature_daily 저장
9. macro_status / macro_summary 생성
```

### 완료 기준

```text
매일 us_trade_date / kr_apply_date 기준 macro feature 생성
risk_on_flag 생성
risk_off_flag 생성
semiconductor_strength_flag 생성
macro_summary 생성
실제 주문 영향 없음
```

---

## 18. Phase 3 상세 — Shadow Mode 적용

### 목표

```text
실제 매매에는 반영하지 않고, 적용했을 경우 결과만 기록한다.
```

### 작업 항목

```text
1. kr_apply_date 기준 최신 us_macro_feature_daily 조회
2. 기존 국내 추천 결과 또는 RULE BUY 후보 조회
3. 종목별 sector/theme 매핑 조회
4. macro_adjustment 계산
5. adjusted_score 계산
6. buy_blocked_flag 계산
7. overlay_reason 생성
8. signal.kr_macro_overlay_log 저장
9. UI / 메일 / 로그에 Shadow Mode 명시
10. 실제 주문 로직에는 adjusted_score를 사용하지 않도록 검증
```

### 완료 기준

```text
실제 주문 영향 없음
추천 점수 변경 예상 결과만 로그로 확인 가능
매수 차단 예상 사유 확인 가능
Shadow Mode 표시 확인 가능
```

---

## 19. Phase 4 상세 — 백테스트 검증

### 목표

```text
미국 macro overlay를 적용했을 때 국내 전략 성과가 개선되는지 검증한다.
```

### 비교 대상

```text
A. 기존 전략
B. overlay 적용 전략
```

### 검증 지표

| 지표 | 설명 |
|---|---|
| 누적 수익률 | 전체 성과 비교 |
| 평균 수익률 | 거래당 평균 성과 |
| 승률 | 수익 거래 비율 |
| MDD | 최대 낙폭 |
| 손실 거래 감소율 | Risk-Off 차단 효과 |
| 매매 횟수 | 기회 감소 여부 |
| 평균 보유 기간 | 전략 성격 변화 여부 |
| Top5 / Top10 성과 변화 | 추천 품질 변화 |
| 섹터별 성과 변화 | 반도체 등 섹터 룰 효과 |

### 핵심 질문

```text
1. Risk-Off 차단이 손실을 줄였는가?
2. Risk-Off 차단 때문에 좋은 매수 기회를 놓치지는 않았는가?
3. 반도체 가산점이 실제 수익률 개선으로 이어졌는가?
4. 전체 점수 보정이 기존 추천 랭킹을 과도하게 왜곡하지 않았는가?
5. 매매 횟수가 줄었을 때 수익률과 MDD가 같이 개선되었는가?
```

---

## 20. Phase 5 상세 — 제한적 실반영

### 실반영 조건

다음 조건을 모두 만족할 때만 실반영을 검토한다.

```text
1. 최소 2~3개월 Shadow Mode 운영
2. 백테스트에서 기존 전략 대비 MDD 개선 확인
3. 손실 거래 감소 효과 확인
4. 매매 기회 손실이 과도하지 않음
5. 로그 / UI에서 차단 사유 확인 가능
6. ON / OFF 환경변수로 즉시 비활성화 가능
7. US_MACRO_ALLOW_REAL_APPLY=true 설정을 별도로 요구
```

### 실반영 순서

```text
1단계: 강한 Risk-Off일 때 신규매수 차단만 반영
2단계: 반도체 급락 시 반도체 섹터 감점 반영
3단계: 반도체 강세 시 소폭 가산 반영
4단계: 포지션 크기 조절은 나중에 검토
```

---

## 21. UI / 알림 요구사항

### 21-1. Risk-Off 표시 예시

```text
미국 매크로 상태: Risk-Off

사유:
- QQQ -2.8%
- SPY -1.9%
- VIX +13.5%
- 주요 섹터 ETF 8개 중 7개 하락

오늘 신규매수 정책:
- 신규 BUY 차단 후보 발생
- 기존 보유종목 매도 룰은 정상 유지

반영 모드:
- Shadow Mode
- 실제 주문 영향 없음
```

### 21-2. 반도체 우호 표시 예시

```text
미국 매크로 상태: Semiconductor Positive

사유:
- SMH +2.1%
- QQQ +0.8%

반영:
- 반도체 섹터 후보 +3점

반영 모드:
- Shadow Mode
- 실제 주문 영향 없음
```

### 21-3. 알림 메시지 예시

```text
[US Macro Overlay] 2026-05-08

상태: Risk-Off

QQQ: -2.6%
SPY: -1.9%
VIX: +12.4%
Sector Breadth: 1/8 상승

조치:
- 신규 BUY 차단 후보 발생
- 국내 성장주 / 반도체 후보 감점
- 실제 주문 반영: Shadow Mode
- 실제 매매 영향 없음
```

Shadow Mode일 때는 반드시 `실제 매매 영향 없음` 문구를 표시한다.

---

## 22. 리스크 및 보완사항

| 리스크 | 내용 | 보완 방법 |
|---|---|---|
| 과최적화 | 미국 ETF 강세가 국내 관련주 상승을 보장하지 않음 | 룰 3~5개만 시작, 백테스트 필수 |
| 날짜 정합성 | 미국 기준일과 한국 적용일 혼동 가능 | us_trade_date / kr_apply_date 분리 |
| 휴장 문제 | 미국 / 한국 휴장일에 stale data 사용 가능 | STALE_DAYS_LIMIT 적용 |
| 실매매 영향 | 초기 잘못 반영 시 주문 오작동 가능 | Shadow Mode 필수, ALLOW_REAL_APPLY 별도 |
| 데이터 소스 | yfinance는 운영용 핵심 소스로 한계 가능 | adapter 구조, 추후 Polygon / Alpaca 전환 |
| 섹터 매핑 | 국내 섹터와 미국 ETF 매핑 오류 가능 | 반도체부터 시작, 점진적 확장 |
| 기회 손실 | Risk-Off 차단이 좋은 기회를 막을 수 있음 | 매매 횟수 / 기회 손실 백테스트 |

---

## 23. Codex 작업 지시 프롬프트 — Phase 1~2

```markdown
# 작업명: 미국 매크로 신호 기반 국내 RULE 강화 - Phase 1~2

현재 Lee_trader 프로젝트에 미국 시장 야간 흐름을 국내 자동매매 보조 신호로 활용하는 기능을 추가하려고 합니다.

목표는 미국주식 실매매가 아닙니다.
미국 지수 / 섹터 ETF / 리스크 지표를 수집하고, 다음날 한국장 매매 판단에 사용할 macro feature를 생성하는 것입니다.

## 반드시 먼저 확인할 파일

아래 파일이 존재하는지 확인하고, 있다면 먼저 읽어주세요.

- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/ENV.md
- compute_theme_etf_daily.py
- signal_builder.py
- run_daily_scheduler.py
- schema.sql
- .env.example

파일 경로가 다르면 프로젝트 내에서 유사한 역할의 파일을 찾아서 확인해주세요.

## 이번 작업 범위

이번 작업은 Phase 1~2까지만 진행합니다.

실제 매수 / 매도 로직에는 절대 반영하지 마세요.
국내 자동매매 주문 결과에 영향을 주면 안 됩니다.

이번 단계의 목표는 다음 두 가지입니다.

1. 미국 시장 데이터 수집
2. macro feature 생성 및 저장

## Phase 1. 미국 ETF / 지수 데이터 수집

다음 티커의 일봉 데이터를 수집하는 기능을 추가해주세요.

- SPY
- QQQ
- DIA
- XLK
- XLF
- XLE
- XLV
- XLI
- XLY
- XLP
- SMH 또는 SOXX
- ^VIX
- DX-Y.NYB
- ^TNX

데이터 소스는 우선 yfinance를 사용합니다.
단, 나중에 Polygon / Alpaca 등으로 교체 가능하도록 source adapter 형태를 고려해주세요.

수집 데이터는 국내 주식 데이터와 섞이지 않도록 별도 테이블 또는 별도 prefix / schema로 관리해주세요.

## Phase 2. 미국 Macro Feature 생성

수집된 데이터를 바탕으로 다음 feature를 생성해주세요.

- spy_ret_1d
- spy_ret_5d
- qqq_ret_1d
- qqq_ret_5d
- semiconductor_ret_1d
- semiconductor_ret_5d
- vix_ret_1d
- dxy_ret_1d
- tnx_ret_1d
- sector_breadth
- top_sector
- bottom_sector
- sector_strength_rank
- risk_on_flag
- risk_off_flag
- vix_spike_flag
- semiconductor_strength_flag
- us_trade_date
- kr_apply_date
- macro_status
- macro_summary

## DB 설계

기존 DB 스타일을 확인한 뒤, 가능하면 아래 목적의 테이블을 추가해주세요.

1. 미국 ETF / 지수 일봉 저장 테이블
2. 미국 macro feature daily 저장 테이블

권장 예시:

- market.us_etf_daily_price
- signal.us_macro_feature_daily

국내 주식 데이터와 섞이지 않도록 반드시 prefix 또는 schema를 분리해주세요.

## 환경변수 추가

.env.example에 다음 설정을 추가해주세요.

- US_MACRO_ENABLED
- US_MACRO_SHADOW_MODE
- US_MACRO_DATA_SOURCE
- US_MACRO_TICKERS
- US_MACRO_RISK_OFF_QQQ_RET
- US_MACRO_RISK_OFF_SPY_RET
- US_MACRO_VIX_SPIKE_RET
- US_MACRO_SEMI_POSITIVE_RET
- US_MACRO_SEMI_NEGATIVE_RET
- US_MACRO_MAX_POSITIVE_ADJUST
- US_MACRO_MAX_NEGATIVE_ADJUST
- US_MACRO_STALE_DAYS_LIMIT
- US_MACRO_SECTOR_BREADTH_RISK_OFF_RATIO
- US_MACRO_SECTOR_BREADTH_RISK_ON_RATIO
- US_MACRO_ALLOW_REAL_APPLY

기본값은 Shadow Mode 기준으로 설정해주세요.

## 주의사항

이번 작업에서는 실제 매수 / 매도 로직에 반영하지 마세요.
국내 자동매매 주문 결과에 영향을 주면 안 됩니다.

이번 단계에서는 다음만 수행합니다.

1. 미국 시장 데이터 수집
2. macro feature 생성
3. DB 저장
4. 실행 로그 출력
5. 테스트 방법 정리

실제 국내 RULE 반영은 다음 Phase에서 진행합니다.

## 완료 조건

작업 완료 후 다음 내용을 정리해주세요.

1. 추가 / 수정된 파일 목록
2. 추가된 DB 테이블 DDL
3. 추가된 환경변수
4. 실행 방법
5. 테스트 방법
6. 실제 매매 영향 여부
7. 다음 Phase에서 해야 할 작업

특히 실제 매매 영향 여부는 명확히 적어주세요.
이번 Phase에서는 실제 매매 영향이 없어야 합니다.
```

---

## 24. Codex 작업 지시 프롬프트 — Phase 3

```markdown
# 작업명: 미국 매크로 신호 기반 국내 RULE 강화 - Phase 3 Shadow Mode 적용

이전 Phase에서 미국 ETF / 지수 데이터 수집과 us_macro_feature_daily 생성을 완료했습니다.

이번 작업은 생성된 미국 macro feature를 국내 추천 / RULE 판단에 Shadow Mode로만 적용하는 것입니다.

중요:
이번 Phase에서도 실제 매수 / 매도 주문에는 영향을 주면 안 됩니다.

## 반드시 먼저 확인할 파일

아래 파일을 먼저 확인해주세요.

- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/ENV.md
- signal_builder.py
- run_daily_scheduler.py
- rule 관련 매수 후보 생성 파일
- 추천 랭킹 생성 파일
- 메일 / UI 출력 관련 파일
- schema.sql
- .env.example

경로가 다르면 유사 역할 파일을 찾아서 확인해주세요.

## 작업 목표

기존 국내 추천 결과 또는 RULE 매수 후보에 대해 미국 macro overlay를 적용했을 경우의 결과를 별도 로그로 남겨주세요.

실제 주문에는 반영하지 않습니다.

## 구현 내용

1. kr_apply_date 기준으로 최신 us_macro_feature_daily를 조회합니다.
2. 국내 추천 후보 또는 RULE BUY 후보를 조회합니다.
3. 각 종목에 대해 macro_adjustment를 계산합니다.
4. adjusted_score를 계산합니다.
5. buy_blocked_flag를 계산합니다.
6. overlay_reason을 생성합니다.
7. signal.kr_macro_overlay_log 또는 기존 네이밍에 맞는 로그 테이블에 저장합니다.
8. Shadow Mode 여부를 명확히 저장합니다.
9. UI / 메일 / 로그에 실제 주문 영향 없음 문구를 표시합니다.

## 반영 룰

초기 룰은 단순하게 구현해주세요.

### Risk-Off 차단 후보

조건:
- qqq_ret_1d <= US_MACRO_RISK_OFF_QQQ_RET
- vix_ret_1d >= US_MACRO_VIX_SPIKE_RET

결과:
- buy_blocked_flag = 'Y'
- macro_adjustment = US_MACRO_MAX_NEGATIVE_ADJUST
- overlay_reason에 QQQ 하락률과 VIX 상승률 기록

### 반도체 가산

조건:
- semiconductor_ret_1d >= US_MACRO_SEMI_POSITIVE_RET
- qqq_ret_1d >= 0
- 국내 종목이 반도체 섹터에 해당

결과:
- macro_adjustment = +3
- overlay_reason에 반도체 ETF 강세 사유 기록

### 반도체 감점

조건:
- semiconductor_ret_1d <= US_MACRO_SEMI_NEGATIVE_RET
- 국내 종목이 반도체 섹터에 해당

결과:
- macro_adjustment = -5
- overlay_reason에 반도체 ETF 약세 사유 기록

### 전체 시장 보수

조건:
- spy_ret_1d <= US_MACRO_RISK_OFF_SPY_RET

결과:
- macro_adjustment = -5
- overlay_reason에 SPY 약세 사유 기록

## 주의사항

- 실제 주문 후보를 바꾸지 마세요.
- 실제 order 로직에 adjusted_score를 사용하지 마세요.
- 기존 final_score는 변경하지 마세요.
- 별도 로그 테이블에만 기록하세요.
- Shadow Mode임을 UI / 메일 / 로그에 명확히 표시하세요.

## 완료 조건

작업 완료 후 다음을 정리해주세요.

1. 추가 / 수정된 파일 목록
2. 추가된 DB 테이블 DDL
3. Shadow Mode 실행 방법
4. 생성되는 로그 예시
5. 실제 주문 영향 여부
6. 다음 Phase에서 실반영 전 검증해야 할 항목
```

---

## 25. 최종 로드맵

| 단계 | 내용 | 시점 |
|---|---|---|
| 1 | 미국 ETF / 지수 데이터 수집 | 즉시 |
| 2 | 미국 macro feature 생성 | 1~2주차 |
| 3 | 국내 RULE Shadow Mode 적용 | 2~3주차 |
| 4 | 백테스트로 효과 검증 | 1~2개월 |
| 5 | 일부 조건만 실반영 | 2~3개월 |
| 6 | 미국주식 자동추천 v1 독립 구축 | B 안정화 후 |
| 7 | 미국 Paper Trading 검증 | C 구축 후 |
| 8 | 미국 실매매 검토 | 가장 마지막 |

---

## 26. 최종 검토 체크리스트

개발 착수 전 다음 항목을 확인한다.

```text
[ ] 기존 schema.sql에 market / signal schema가 존재하는가?
[ ] 기존 프로젝트에서 yfinance 의존성이 있는가?
[ ] compute_theme_etf_daily.py의 ETF / 테마 매핑 구조를 확인했는가?
[ ] signal_builder.py에서 점수 계산과 주문 후보 생성이 분리되어 있는가?
[ ] run_daily_scheduler.py에 미국 수집 스텝을 추가해도 기존 파이프라인에 영향이 없는가?
[ ] .env.example에 Shadow Mode 기본값이 들어갔는가?
[ ] Phase 1~2는 실제 주문 영향이 전혀 없는가?
[ ] Phase 3에서도 adjusted_score가 실제 주문에 사용되지 않는가?
[ ] UI / 메일에 Shadow Mode 문구가 표시되는가?
[ ] 수집 실패 시 전체 자동매매 파이프라인이 중단되지 않는가?
[ ] 미국 휴장 / stale data 처리 기준이 있는가?
```

---

## 27. 최종 결정

이번 Project B는 진행한다.

단, 다음 조건을 반드시 지킨다.

```text
1. Phase 1~2는 수집과 feature 생성까지만 진행한다.
2. Phase 3은 Shadow Mode 로그만 생성한다.
3. 실제 매매 영향은 Phase 5 전까지 금지한다.
4. 실반영 전에는 백테스트와 2~3개월 Shadow 운영 결과를 확인한다.
5. 미국주식 직접매매는 Project C 이후 가장 마지막에 검토한다.
```

최종 판단:

```text
미국주식 자동매매를 바로 붙이기 전에,
미국 매크로 신호 기반 국내 RULE 강화 작업을 먼저 진행하는 것이 가장 합리적이다.

이 작업은 현재 Lee_trader 시스템을 강화하면서,
향후 미국주식 자동추천 v1으로 확장할 수 있는 기반도 만들어준다.
```
