# Feature Gap Analysis

- generated_at: 2026-03-18 09:56:00
- basis: current code / current CSV headers / current DB schema
- scope: feature generation pipeline, market data, universe metadata, ranking final_score inputs

## 1. Pipeline Overview

현재 일일 파이프라인 순서는 `python/run_pipeline.py`의 `STEPS`에 정의되어 있습니다.

1. `python/fetch_market_data.py`
2. `python/fetch_top_universe.py`
3. `python/merge_universe.py`
4. `python/download_prices_kis.py`
5. `python/clean_prices.py`
6. `python/create_adjusted_prices.py`
7. `python/fetch_fundamentals_dart.py`
8. `python/quality_builder.py`
9. `python/feature_builder.py`
10. `python/label_builder.py`
11. `python/model_train.py`
12. `python/model_predict.py`
13. `python/ranking_builder.py`

핵심 해석:

- 가격/거래량 원천은 `download_prices_kis.py` -> `clean_prices.py` -> `create_adjusted_prices.py`를 거칩니다.
- quality 계열은 `fetch_fundamentals_dart.py` -> `quality_builder.py`를 거칩니다.
- 일봉 feature는 `feature_builder.py`에서 생성됩니다.
- 최종 점수는 `ranking_builder.py`에서 계산됩니다.

## 2. 거래량 관련 컬럼 생성/적재 위치

### 2.1 현재 실제 생성되는 컬럼

현재 feature 파이프라인에서 실제 생성/적재되는 거래량 관련 컬럼은 아래입니다.

| 컬럼 | 생성 파일 / 함수 | 저장 위치 | 비고 |
| --- | --- | --- | --- |
| `volume` | `python/download_prices_kis.py` -> `try_kis_download()`, `try_pykrx_download()` | `data/prices_daily_raw.csv`, DB `prices_raw` | 원천 일봉 거래량 |
| `volume` | `python/clean_prices.py` | `data/prices_daily_clean.csv`, DB `prices_clean` | 정제 후 유지 |
| `volume` | `python/create_adjusted_prices.py` -> `main()` | `data/prices_daily_adjusted.csv`, DB `prices_adjusted`, DB `fact_price_daily` | 수정주가 생성 후 유지 |
| `volume` | `python/feature_builder.py` -> `load_prices()` / `build_features()` | `data/features.csv`, DB `features` | feature 입력으로 사용 |
| `vol_ma_20` | `python/feature_builder.py` -> `build_features()` | `data/features.csv`, DB `features` | 20일 평균 거래량 |
| `vol_ratio_20` | `python/feature_builder.py` -> `build_features()` | `data/features.csv`, DB `features` | `volume / vol_ma_20` |
| `tech_volume_score` | `python/ranking_builder.py` -> `_compute_feature_based_tech_score()` | `data/ranking_final.csv`, DB `daily_ranking` | volume 기반 tech 내부 점수 |
| `liquidity_score` | `python/ranking_builder.py` -> `_compute_liquidity_score()` | `data/ranking_final.csv`, DB `daily_ranking` | `vol_ma_20` 우선, 없으면 `volume` 사용 |

### 2.2 현재 스키마에는 있으나 feature로 안 쓰이는 컬럼

| 컬럼 | 위치 | 실제 상태 | 판단 |
| --- | --- | --- | --- |
| `value` | DB `fact_price_daily` | `python/create_adjusted_prices.py`에서 `fact_df["value"] = pd.NA`로 적재 | 스키마만 있고 현재 비어 있음 |
| `market_cap` | DB `fact_price_daily` | `pd.NA`로 적재 | 현재 미사용 |
| `listed_shares` | DB `fact_price_daily` | `pd.NA`로 적재 | 현재 미사용 |

실측 확인:

- Postgres `fact_price_daily.value` non-null count: `0`

### 2.3 결론

- `volume` 계열은 이미 end-to-end로 존재합니다.
- `value`, `turnover`는 현재 feature 파이프라인에 **실질적으로 없음**으로 보는 것이 맞습니다.
- 거래대금/회전율 계열을 쓰려면 `download_prices_kis.py` 또는 `create_adjusted_prices.py`에서 실제 값을 채워야 합니다.

## 3. 외국인/기관 수급 데이터 존재 여부

### 3.1 외국인 수급

현재 코드 기준으로 존재하는 외국인 수급 계열은 **시장 레벨 proxy 한 개**입니다.

| 컬럼 | 생성 파일 / 함수 | 저장 위치 | 실제 의미 |
| --- | --- | --- | --- |
| `foreign_net_5d` | `python/fetch_market_data.py` -> `fetch_kospi_foreign_flow()` / `main()` | `data/market_status.csv`, DB `market_status`, `ranking_final.csv`의 `market_foreign_5d` | 종목별 수급이 아니라 KOSPI 레벨 5일 외국인 흐름 proxy |

구현 방식:

- `fetch_market_data.py`는 실제 외국인 순매수 원시 데이터가 아니라,
- 네이버 금융 KOSPI 시총 상위 종목의 외국인 보유비율 변화와 지수 거래량을 이용해 proxy series를 만든 뒤,
- 이를 `foreign_net_5d`로 저장합니다.

관련 함수:

- `python/fetch_market_data.py` `_fetch_naver_foreign_ratio_today()`
- `python/fetch_market_data.py` `fetch_kospi_foreign_flow()`
- `python/fetch_market_data.py` `main()`
- `python/ranking_builder.py` `_attach_market_columns()`

### 3.2 기관 수급

현재 코드/DB/CSV 기준으로 **기관 수급 컬럼은 없음**입니다.

실측 확인:

- Postgres 컬럼명 검색 `institution%`: `0`
- CSV 헤더 검색 결과 기관 수급 계열 컬럼 없음

### 3.3 종목별 수급

현재 파이프라인에는 종목별 외국인/기관 순매수, 수급 강도, 수급 누적 같은 컬럼이 없습니다.

즉 현재 상태는 아래와 같습니다.

- 시장 레벨 외국인 proxy: 있음
- 종목 레벨 외국인 수급: 없음
- 종목 레벨 기관 수급: 없음

## 4. Sector / Theme 관련 컬럼 존재 여부

### 4.1 Sector

`sector`는 현재 존재합니다.

| 위치 | 상태 |
| --- | --- |
| `data/universe.csv` | 있음 |
| DB `stocks` | 있음 |
| `data/ranking_final.csv` | 있음 |
| DB `daily_ranking` | 있음 |

생성/보강 경로:

- `python/fetch_top_universe.py`
  - `_naver_fetch_sector(code)`
  - `_classify_sector_by_name(name)`
  - 기존 `stocks`, `sectors.csv`, 네이버 fallback, 이름 기반 fallback을 통해 sector 보강
- `python/merge_universe.py`
  - `name`, `market`, `sector` 컬럼을 보존/병합
- `python/ranking_builder.py`
  - `universe.csv`를 metadata로 병합하여 ranking 결과에 포함

### 4.2 Theme

`theme`는 현재 코드/DB/CSV 기준으로 없습니다.

실측 확인:

- Postgres 컬럼명 검색 `theme`: `0`
- 주요 CSV 헤더(`features.csv`, `universe.csv`, `ranking_final.csv`)에 `theme` 없음
- 관련 코드 검색 결과 feature/ranking 경로에서 theme 생성 로직 없음

## 5. ranking_builder.py에서 final_score 계산에 사용되는 컬럼

### 5.1 direct inputs

`python/ranking_builder.py` `apply_default_ranking_scores()`에서 `final_score`는 직접적으로 아래 컬럼만 사용합니다.

```text
final_score =
    w_ret * ret_score
  + w_prob * prob_score
  + w_tech * tech_score
  + w_qual * qual_score
  + w_valuation * valuation_score
  - w_risk_penalty * risk_penalty
```

관련 함수:

- `python/ranking_builder.py` `compute_component_scores()`
- `python/ranking_builder.py` `_compute_risk_penalty()`
- `python/ranking_builder.py` `_ensure_regime_column()`
- `python/ranking_builder.py` `apply_default_ranking_scores()`

직접 입력 컬럼 목록:

| 구분 | 컬럼 |
| --- | --- |
| positive component | `ret_score` |
| positive component | `prob_score` |
| positive component | `tech_score` |
| positive component | `qual_score` |
| positive component | `valuation_score` |
| negative component | `risk_penalty` |
| weight selector | `regime` |
| weight output | `w_ret`, `w_prob`, `w_tech`, `w_qual`, `w_valuation`, `w_risk_penalty` |

중요:

- `safety_score`, `liquidity_score`는 **production final_score에 직접 더해지지 않습니다**.
- 현재 설계에서는 safety 계열 영향이 `risk_penalty`를 통해 우회 반영됩니다.
- `confidence_score`도 `final_score`에 곱하거나 빼지 않습니다.

### 5.2 component별 upstream source

| component | 생성 함수 | 실제 upstream 컬럼 |
| --- | --- | --- |
| `ret_score` | `_compute_ret_score()` | `pred_return_60d`, `pred_return_90d` |
| `prob_score` | `_compute_prob_score()` | `prob_top20_60d`, `prob_top20_90d` |
| `tech_score` | `_compute_tech_score()` / `_compute_feature_based_tech_score()` | `score_score` 또는 `close`, `ma_5`, `ma_20`, `ma_60`, `ret_5d`, `ret_10d`, `mom_20`, `rsi_14`, `vol_20`, `vol_60`, `vol_ma_20`, `vol_ratio_20`, `volume` |
| `qual_score` | `_compute_qual_score()` | `quality_score` |
| `valuation_score` | `_compute_valuation_score()` | 현재 valuation 입력 컬럼이 없어 neutral `50.0` fallback 가능 |
| `risk_penalty` | `_compute_risk_penalty()` | `pred_mdd_60d`, `pred_mdd_90d`, `vol_20`, `vol_60` |
| `regime` | `_ensure_regime_column()` / `detect_market_regime()` | `market_status.csv`, `close_over_ma20`, 지수 히스토리 |

## 6. 현재 존재하는 feature

### 6.1 price / volume / technical

- `close`
- `ret_1d`
- `ret_5d`
- `ret_10d`
- `mom_20`
- `ma_5`
- `ma_20`
- `ma_60`
- `close_over_ma20`
- `vol_20`
- `vol_60`
- `rsi_14`
- `volume`
- `vol_ma_20`
- `vol_ratio_20`

생성 파일 / 함수:

- `python/feature_builder.py` `build_features()`

### 6.2 quality

- `quality_score`
- `quality_factor_count`
- `quality_missing_ratio`
- `quality_score_confidence`

생성/병합 경로:

- `python/quality_builder.py`
- `python/feature_builder.py` `merge_quality()`

### 6.3 market / regime

- `market_up`
- `kospi_close`
- `kospi_ma20`
- `volatility_5d`
- `foreign_net_5d`
- `regime`
- `regime_reason`

생성 경로:

- `python/fetch_market_data.py`
- `python/ranking_builder.py` `_attach_market_columns()`
- `python/ranking_builder.py` `detect_market_regime()`

### 6.4 metadata

- `name`
- `market`
- `sector`
- `listed_at`
- `delisted_at`

생성 경로:

- `python/fetch_top_universe.py`
- `python/merge_universe.py`

## 7. 이번에 추가해야 하는 feature

아래는 현재 gap이 확인된 항목입니다.

| feature | 현재 상태 | 데이터 소스 확보 가능 여부 | 구현 난이도 | 메모 |
| --- | --- | --- | --- | --- |
| `value` 거래대금 | DB 스키마만 있고 실제 적재 없음 | 가능. 현재 raw 가격 + volume이 있으므로 최소 `close * volume` 근사 가능 | 낮음 | `fact_price_daily`와 `features.csv` 둘 다 반영 가능 |
| `turnover` 회전율 | 없음 | 부분 가능. `listed_shares`가 현재 비어 있어 먼저 채워야 함 | 중간 | `value`보다 선행 난이도 높음 |
| 종목별 `foreign_net_*` | 없음 | 현재 코드엔 없음. 신규 데이터 소스 필요 | 높음 | 시장 proxy만 존재 |
| 종목별 `institution_net_*` | 없음 | 현재 코드엔 없음. 신규 데이터 소스 필요 | 높음 | 기관 수급 경로 부재 |
| `theme` | 없음 | 현재 코드엔 없음. 신규 taxonomy/source 필요 | 중간~높음 | sector와 별도 설계 필요 |
| valuation raw inputs (`per`, `pbr`, `psr`, `ev_ebitda` 등) | final_score 입력은 있으나 실제 소스 빈약 | 일부 가능. fundamentals 파이프라인 확장 필요 | 중간 | 현재 `valuation_score`가 neutral fallback로 흐를 수 있음 |

## 8. 데이터 소스 확보 가능 여부

### 8.1 바로 확보 가능한 항목

- `value`
  - 현재 `close`와 `volume`이 이미 있습니다.
  - 최소 근사치는 `close * volume`으로 즉시 생성 가능합니다.
  - 구현 위치 후보:
    - `python/create_adjusted_prices.py`
    - `python/feature_builder.py`

### 8.2 추가 적재가 필요한 항목

- `turnover`
  - `listed_shares`가 필요하지만 현재 `fact_price_daily`에는 `pd.NA`로 저장됩니다.
  - 먼저 발행주식수 적재 경로를 만들어야 합니다.

- 종목별 외국인/기관 수급
  - 현재 파이프라인에는 해당 원천 적재 파일/테이블이 없습니다.
  - 새로운 fetch step과 저장 테이블이 필요합니다.

- `theme`
  - 현재 universe/stocks에도 theme taxonomy가 없습니다.
  - 별도 source file 또는 분류 규칙이 필요합니다.

## 9. 추천 구현 순서

### 1순위

1. `value` 거래대금 추가
2. `feature_builder.py`에서 `value_ma_20`, `value_ratio_20` 파생
3. `ranking_builder.py` liquidity/tech에 `value` 계열 선택 반영

이유:

- 현재 데이터만으로 바로 구현 가능
- liquidity 품질 개선 효과가 큼
- 난이도가 가장 낮음

### 2순위

1. `listed_shares` 또는 `market_cap` 실제 적재
2. `turnover` 생성
3. turnover 기반 liquidity 확장

이유:

- 거래량보다 왜곡이 적은 유동성 지표를 만들 수 있음
- 다만 선행 데이터 적재가 필요

### 3순위

1. 종목별 외국인 수급 적재
2. 종목별 기관 수급 적재
3. ranking / explain / diagnostics에 수급축 반영

이유:

- 신호 가치가 높을 수 있지만 현재 원천 데이터 경로가 전혀 없음
- 신규 fetch/저장/검증 범위가 큼

### 4순위

1. `theme` taxonomy 설계
2. universe/stocks에 `theme` 적재
3. sector-theme concentration 진단 추가

이유:

- UI/설명성 강화에는 유용
- final_score 직접 개선보다 우선순위는 낮음

## 10. 최종 결론

- 거래량 계열은 `volume`, `vol_ma_20`, `vol_ratio_20`까지만 운영 중입니다.
- `value`와 `turnover`는 현재 실질적으로 비어 있거나 없습니다.
- 외국인 수급은 시장 레벨 proxy만 있고, 종목별 외국인/기관 수급은 없습니다.
- `sector`는 이미 있고 `theme`는 없습니다.
- production `final_score`의 직접 입력은 `ret_score`, `prob_score`, `tech_score`, `qual_score`, `valuation_score`, `risk_penalty`, `regime`입니다.

현재 프로젝트 기준으로 가장 먼저 메우는 게 좋은 gap은 `value`와 `turnover` 쪽입니다.
