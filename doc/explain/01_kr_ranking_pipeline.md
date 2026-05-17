# 국내 주식 랭킹 산정 파이프라인

*작성 기준일: 2026-05-17*

---

## 개요

매일 장 마감 후 18:10에 자동으로 실행되는 배치 파이프라인입니다.  
외부 데이터 수집 → 피처 생성 → 모델 예측 → 최종 점수·순위 산출 순으로 진행됩니다.

```
fetch_market_data
fetch_top_universe
download_prices_kis     → clean_prices → create_adjusted_prices
fetch_fundamentals_dart
download_flows_kis
        ↓
quality_builder
feature_builder
        ↓
label_builder
model_train / model_predict
        ↓
ranking_builder         → ranking_final.csv
```

---

## 1. 유니버스 선정

**파일**: `python/fetch_top_universe.py`

### 1-1. 대상 시장

KOSPI와 KOSDAQ 전체에서 시가총액 상위 종목을 추출합니다.  
`TOP_N` 환경변수로 크기를 조정할 수 있으며, 기본값은 상위 400개입니다.

### 1-2. 데이터 소스 우선순위

| 순위 | 소스 | 비고 |
|---|---|---|
| 1 | 네이버 금융 크롤링 | 시총 순 정렬 페이지 파싱 |
| 2 | KIS API | `NAVER_FALLBACK_USE_KIS=1` 시 사용 |
| 3 | pykrx | soft-import, 보조 소스 |
| 4 | FinanceDataReader | soft-import, 최종 fallback |

### 1-3. 수집 정보

- 종목코드, 종목명, 상장 시장(KOSPI/KOSDAQ), 섹터, 상장일, 상장폐지일

### 1-4. 출력

- `data/universe.csv` — 유효 종목 목록
- DB `stocks` 테이블 upsert

---

## 2. 데이터 수집 인프라

### 2-1. 시장 지수 및 레짐 데이터

**파일**: `python/fetch_market_data.py`

KOSPI 지수 OHLCV와 외국인 투자 비율을 수집하여 당일 **시장 레짐**을 판단합니다.

**시장 레짐 판단 기준 (market_up = True 조건 모두 충족 시)**:

| 조건 | 기준값 | 의미 |
|---|---|---|
| `kospi_close > kospi_ma20` | — | KOSPI가 20일 이평선 위 |
| `volatility_5d < 0.03` | 3% | 5일 일간 표준편차 3% 미만 |
| `foreign_net_5d > 0` | — | 최근 5일 외국인 순매수 양수 |

**출력**: `data/market_status.csv`  
컬럼: `date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up`

---

### 2-2. 일별 주가 데이터

**파일**: `python/download_prices_kis.py` → `clean_prices.py` → `create_adjusted_prices.py`

KIS API를 통해 유니버스 전 종목의 일별 OHLCV를 수집합니다.

- `clean_prices.py`: 가격 이상치(스탑로스·갭) 제거
- `create_adjusted_prices.py`: 주식분할·배당 반영 수정 주가 생성
- 저장: DB `fact_price_daily` 테이블 + `data/prices_daily_adjusted.csv`

---

### 2-3. 기관·외국인 수급 데이터

**파일**: `python/download_flows_kis.py`

KIS API TR `FHKST01010200`을 통해 종목별 투자자 유형별 순매수 금액을 수집합니다.

**수집 대상 투자자 유형**:
- 외국인 (foreign)
- 기관 (institution)
- 개인 (individual)

**저장**: DB `flow_daily` 테이블  
컬럼: `date, code, foreign_net, institution_net, individual_net` (매수·매도 합계 포함)

**백필 도구**:
- `python/run_flow_backfill_local.py` — KIS API 기반, 최대 90일치
- `python/fetch_flow_pykrx.py` — pykrx 기반, 1년치 이상 (KRX 계정 필요)

---

### 2-4. 재무제표 데이터

**파일**: `python/fetch_fundamentals_dart.py`

DART 공시 API를 통해 상장사 연간 재무제표를 수집합니다.

**수집 항목**:
- 매출액, 영업이익, 당기순이익, 자본총계, 부채총계
- PER, PBR, ROE (산출 기준)

**처리 방식**:
1. DART 회사 코드 다운로드·캐시
2. 종목별 최근 연간 재무제표 조회 (보고서 코드 `11014`)
3. 기존 데이터와 병합 (point-in-time 기준)
4. `data/dart/fundamentals.csv` + DB `fundamentals` 테이블 저장

> **현재 수집 범위**: 연간 보고서만 수집 (최대 12개월 지연 내재).  
> 분기·반기(11011/11012/11013) 확장은 Phase B-1 과제로 예정.

---

## 3. 피처 빌딩

### 3-1. 기술적 피처

**파일**: `python/feature_builder.py` — `build_features()` 함수

일별 주가 데이터를 기반으로 종목별 기술적 지표를 계산합니다.

| 피처 그룹 | 피처명 | 설명 |
|---|---|---|
| **수익률** | `ret_1d, ret_5d, ret_10d, ret_60d, ret_120d` | 각 기간 로그수익률 |
| **모멘텀** | `mom_20` | 20일 모멘텀 |
| **이동평균** | `ma_5, ma_20, ma_60` | 단순 이동평균 |
| | `close_over_ma20` | 종가 / MA20 비율 |
| **변동성** | `vol_20, vol_60` | 20일·60일 일간 표준편차 |
| | `rsi_14` | RSI(14) |
| **거래량** | `volume_ratio_5d, volume_ratio_20d` | 평균 대비 거래량 비율 |
| **거래대금** | `value_ratio_5d, value_ratio_20d` | 평균 대비 거래대금 비율 |
| **복합 점수** | `volume_score, liquidity_score` | 거래량·유동성 백분위 복합 점수 |
| **중장기** | `high_52w_ratio` | 52주 신고가 대비 현재가 비율 |

---

### 3-2. 수급 피처

**파일**: `python/feature_builder.py` — `merge_flow()` 함수

`flow_daily` 테이블에서 외국인·기관 순매수를 롤링 합산합니다.

| 피처명 | 계산 방식 |
|---|---|
| `flow_foreign_net_5d` | 외국인 순매수 5영업일 누적 |
| `flow_foreign_net_20d` | 외국인 순매수 20영업일 누적 |
| `flow_inst_net_5d` | 기관 순매수 5영업일 누적 |
| `flow_inst_net_20d` | 기관 순매수 20영업일 누적 |

> 데이터 미보유 종목은 `NaN` 처리 — 점수 산출 시 중립값(50.0)으로 대체됩니다.

---

### 3-3. 재무 품질 피처

**파일**: `python/quality_builder.py`

DART 재무제표를 기반으로 종목별 재무 품질 점수를 생성합니다.

**주요 처리 단계**:
1. 일별 winsorize (이상치 제거)
2. 중앙값 기반 결측치 채우기
3. robust z-score 정규화
4. 일별 백분위 순위 변환

**출력 피처**:

| 피처명 | 설명 |
|---|---|
| `quality_score` | 재무 품질 종합 점수 (0~100) |
| `quality_factor_count` | 유효 재무 인자 수 |
| `quality_missing_ratio` | 결측 인자 비율 |
| `quality_score_confidence` | 점수 신뢰도 |
| `revenue_growth_yoy` | 매출액 YoY 성장률 |
| `op_income_growth_yoy` | 영업이익 YoY 성장률 |
| `roe_yoy` | ROE YoY 변화 |

---

### 3-4. 기타 피처

**파일**: `python/feature_builder.py`

| 피처 | 함수 | 설명 |
|---|---|---|
| `sector_rel_momentum_20d` | `merge_sector_rel_momentum()` | 종목 20일 수익률 - 동일 섹터 평균 수익률 |
| `fin_momentum_phase` 외 | `merge_financial_momentum()` | DART 기반 재무 모멘텀 단계 및 점수 |
| `short_ratio` 외 | `merge_short_interest()` | 공매도 잔고 비율 및 변화율 |

---

### 3-5. 피처 저장

- `data/features.csv` — 전체 피처 스냅샷 (80개+ 컬럼)
- DB `features` 테이블 upsert (기본 키: `date, code`)

---

## 4. 랭킹 점수 산출

### 4-1. 모델 예측값 생성

**파일**: `python/model_train.py`, `python/model_predict.py`

**알고리즘**: LightGBM (회귀 + 분류)

| 예측 타겟 | 모델 유형 | 의미 |
|---|---|---|
| `pred_return_60d` | 회귀 | 60일 후 예상 로그수익률 |
| `pred_return_90d` | 회귀 | 90일 후 예상 로그수익률 |
| `pred_mdd_60d` | 회귀 | 60일 내 최대 낙폭 예측 |
| `pred_mdd_90d` | 회귀 | 90일 내 최대 낙폭 예측 |
| `prob_top20_60d` | 분류 | 60일 후 상위 20% 진입 확률 |
| `prob_top20_90d` | 분류 | 90일 후 상위 20% 진입 확률 |

**학습 설정**:
- TimeSeriesSplit CV (N_SPLITS=3) — 미래 데이터 leakage 방지
- n_estimators=400, learning_rate=0.03 (회귀) / 0.05 (분류)
- 매일 배치에서 최신 데이터로 자동 재학습

---

### 4-2. 컴포넌트 점수

**파일**: `python/scoring/final_score.py`, `python/ranking_builder.py`

최종 점수는 6개 컴포넌트 점수의 가중 합산으로 계산됩니다.

| 컴포넌트 | 설명 | 핵심 입력 |
|---|---|---|
| `ret_score` | 수익률·예측 수익률 점수 | pred_return_60d/90d |
| `prob_score` | Top20 진입 확률 점수 | prob_top20_60d |
| `tech_score` | 기술적 점수 | RSI, 이동평균, 거래량 |
| `qual_score` | 재무 품질 점수 | quality_score |
| `flow_score` | 수급 점수 | flow_foreign/inst_net_5d |
| `risk_penalty` | 리스크 페널티 | pred_mdd_60d, vol_20 |

**flow_score 계산식**:
```
flow_score = 0.6 × percentile(flow_foreign_net_5d)
           + 0.4 × percentile(flow_inst_net_5d)
(0~100, 동일 날짜 내 상대 순위 / 데이터 없으면 50.0으로 neutral 처리)
```

---

### 4-3. 시장 레짐 및 가중치

**파일**: `python/scoring/final_score.py`

`market_status.csv`의 `market_up` 플래그를 기반으로 레짐을 결정하고,  
컴포넌트별 가중치를 동적으로 적용합니다.

| 레짐 | 조건 | ret | prob | tech | qual | flow | risk_pen |
|---|---|---|---|---|---|---|---|
| **Bull** | market_up=True, 강세 | 0.33 | 0.24 | 0.23 | 0.08 | 0.12 | 0.40 |
| **Neutral** | 중간 상태 | 0.28 | 0.23 | 0.21 | 0.18 | 0.10 | 0.65 |
| **Defensive** | market_up=False, 약세 | 0.23 | 0.19 | 0.16 | 0.34 | 0.08 | 0.80 |

> Defensive 레짐에서는 재무 품질(`qual_score`) 비중이 크게 높아지고,  
> 리스크 페널티 계수도 강화됩니다.

---

### 4-4. 최종 점수 버전

**파일**: `python/ranking_builder.py`

`SCORE_FORMULA_VERSION` 환경변수로 점수 공식을 선택합니다.

| 점수 컬럼 | 설명 |
|---|---|
| `final_score` | 레짐 인식 가중치 — **실제 운영 점수** |
| `final_score_v2` | 고정 가중치 — 비교·연구용 |
| `live_score` | 실제 정렬 키 (= final_score 또는 v3) |
| `live_rank` | 최종 순위 (1위 = 최고점) |

현재 적용 버전: `ranking_builder_v9_flow` (수급 피처 포함).

---

### 4-5. 출력

- `data/ranking_final.csv` — 전체 종목 점수·순위 스냅샷
- DB `ranking` 테이블 저장
- `outputs/operational_refresh_status.json` — 파이프라인 실행 결과 요약
