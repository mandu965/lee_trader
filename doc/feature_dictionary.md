# Feature Dictionary

*최종 수정: 2026-05-29 — A-3 merge 버그 수정 반영 (revenue_growth_yoy/op_income_growth_yoy 51%/45% 복구)*

---

## Liquidity / Volume Features

### `volume_ratio_5d`
- 정의: 당일 `volume / 최근 5일 평균 volume`
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 처리 규칙: 코드별 rolling mean, `0.0~5.0` clip, 일자별 winsorize
- 해석: `1.0` 초과 = 최근 평균 대비 거래량 증가

### `volume_ratio_20d`
- 정의: 당일 `volume / 최근 20일 평균 volume`
- 생성 파일: `feature_builder.py` — `build_features()`
- 해석: 중기 평균 대비 거래량 확장 여부

### `value_ratio_5d`
- 정의: `close * volume / 최근 5일 평균 거래대금`
- 해석: 가격 상승 + 거래량 증가가 동시에 반영된 단기 거래대금 확장 신호

### `value_ratio_20d`
- 정의: `close * volume / 최근 20일 평균 거래대금`
- 해석: 중기 기준 거래대금 확장 여부

### `volume_score`
- 정의: `0.40×percentile(volume_ratio_5d) + 0.40×percentile(volume_ratio_20d) + 0.20×percentile(volume)`
- 범위: `0~100`

### `liquidity_score`
- 정의: `0.35×percentile(vol_ma_20) + 0.25×percentile(volume) + 0.20×percentile(value_ratio_20d) + 0.20×percentile(value_ratio_5d)`
- 범위: `0~100`

---

## Momentum / Return Features

### `ret_1d`, `ret_5d`, `ret_10d`
- 정의: 1/5/10 영업일 수익률 (`close.pct_change(n)`)
- 생성 함수: `build_features()`

### `mom_20`
- 정의: 20 영업일 수익률 (`close.pct_change(20)`)
- 생성 함수: `build_features()`
- 주의: 컬럼명이 `ret_20d`가 아니라 `mom_20`

### `ret_60d`, `ret_120d`
- 정의: 60/120 영업일 수익률
- 목적: 중장기 모멘텀 피처 (A-1)

### `high_52w_ratio`
- 정의: `close / rolling_max(252일, min_periods=60)`
- 범위: `0.0~1.0` (1.0 = 52주 신고가)
- 목적: 52주 신고가 근접도 (A-2)

---

## Technical Features

### `ma_5`, `ma_20`, `ma_60`
- 정의: 5/20/60일 이동평균

### `close_over_ma20`
- 정의: `close / ma_20 - 1`

### `vol_20`, `vol_60`
- 정의: 20/60일 수익률 표준편차 (변동성)

### `rsi_14`
- 정의: 14일 RSI (0~100), EWM 방식

---

## Flow Features (수급)

> **상태: 구현 완료 (2026-05-13)**  
> `flow_daily` 테이블에서 외국인·기관 순매수를 rolling 합산.

### `flow_foreign_net_5d`
- 정의: 외국인 순매수 최근 5 영업일 합산 (원화)
- 생성 파일: `feature_builder.py` — `merge_flow()`

### `flow_foreign_net_20d`
- 정의: 외국인 순매수 최근 20 영업일 합산
- 생성 파일: `feature_builder.py` — `merge_flow()`

### `flow_inst_net_5d`
- 정의: 기관 순매수 최근 5 영업일 합산
- 생성 파일: `feature_builder.py` — `merge_flow()`

### `flow_inst_net_20d`
- 정의: 기관 순매수 최근 20 영업일 합산
- 생성 파일: `feature_builder.py` — `merge_flow()`

---

## Quality / Fundamental Features

### `quality_score`
- 정의: 재무 품질 composite (안정성·수익성·성장성)
- 생성 파일: `quality_builder.py`

### `revenue_growth_yoy`
- 정의: 매출액 전년 대비 성장률 (YoY %)
- 생성 파일: `feature_builder.py` — `merge_financial_momentum()` (primary, `revenue_yoy` 컬럼 rename) + `quality_builder.py` fallback (A-3)
- 커버리지: features 테이블 95,739행 (51.6%), 최근 일자 90.7%
- **2026-05-29 수정**: 이전엔 `merge_quality()`와 `merge_financial_momentum()`의 컬럼 충돌로 `pd.merge_asof`가 `_x`/`_y` 접미사를 만들어 0행이었음. `combine_first` coalesce로 해결.

### `op_income_growth_yoy`
- 정의: 영업이익 YoY 성장률 (%)
- 생성 파일: `feature_builder.py` — `merge_financial_momentum()` (primary, `op_profit_yoy` 컬럼 rename) + `quality_builder.py` fallback (A-3)
- 커버리지: features 테이블 84,305행 (45.5%), 최근 일자 73.5%
- **2026-05-29 수정**: 위와 동일 버그 해결.

### `roe_yoy`
- 정의: ROE YoY 변화 (pp)
- 생성 파일: `quality_builder.py` — `build_yoy_growth_features()` (A-3)
- 커버리지: features 117,120행 (63.2%) — 영향 없음 (`merge_financial_momentum`에서 처리 안 함)

---

## Sector Features

### `sector_rel_momentum_20d`
- 정의: `mom_20 - 동일 섹터 평균 mom_20` (날짜별 cross-sectional)
- 생성 파일: `feature_builder.py` — `merge_sector_rel_momentum()` (C-3)
- 주의: 섹터 내 종목 수 < 2이면 NaN

---

## Financial Momentum Features (분기 재무 모멘텀)

> **상태: 소스 완료 (Phase 1~4, 2026-05-15). 서버 실행 전까지 NaN.**  
> 생성 파일: `build_financial_momentum_features.py` (Phase 2) → `feature_builder.py` `merge_financial_momentum()` (Phase 3, point-in-time merge)

### `fin_momentum_phase`
- 정의: 매출·영업이익 추세 구간 분류 (TEXT)
- 값: `ACCELERATING` / `GROWING` / `SLOWING` / `WEAKENING` / `DECLINING` / `TURNAROUND` / `SECTOR_EXCEPTION` / `UNKNOWN`

### `fin_momentum_score`
- 정의: 재무 모멘텀 종합 점수 (0~100, base 50)

### `fin_risk_score`
- 정의: 재무 위험 점수 (0~100, base 0). 높을수록 위험

### `fin_turnaround_score`
- 정의: 흑자 전환 강도 점수 (0~100)

### `fin_hard_risk`
- 정의: 실적 훼손 위험 flag (0.0 / 1.0)
- 트리거: DECLINING 2분기 이상 + 영업이익률 급락 조합

---

## Short Interest Features (공매도 잔고)

> **상태: 소스 완료 (C-1, 2026-05-15). 서버에서 fetch_short_interest.py 실행 전까지 NaN.**  
> 생성 파일: `fetch_short_interest.py` → `feature_builder.py` `merge_short_interest()`

### `short_ratio`
- 정의: 공매도 잔고 비율 (%) = 공매도잔고 / 상장주식수 × 100
- 출처: pykrx `get_shorting_balance_by_ticker()`

### `short_ratio_5d_chg`
- 정의: 5 영업일 전 대비 `short_ratio` 변화량 (pp)
- 해석: 양수 = 공매도 급증, 하방 압력 신호

### `short_ratio_20d_avg`
- 정의: 20 영업일 이동 평균 `short_ratio`
- 해석: 추세 수준 파악

---

## 관련 문서

- `python/feature_builder.py` — 전체 feature 생성 파이프라인
- `python/quality_builder.py` — 재무 품질 + YoY 성장률
- `python/build_financial_momentum_features.py` — 분기 재무 모멘텀
- `doc/modules/Lee_trader_ai/FINANCIAL_MOMENTUM_DESIGN.md` — 재무 모멘텀 설계 상세
