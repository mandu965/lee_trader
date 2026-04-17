# Feature Dictionary

## Liquidity / Volume Features

### `volume_ratio_5d`
- 정의: 당일 `volume / 최근 5일 평균 volume`
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 처리 규칙:
  - 코드별 rolling mean 기준
  - `0.0 ~ 5.0` clip
  - 일자별 winsorize 적용
- 해석:
  - `1.0` 부근은 평시 수준
  - `1.0` 초과는 최근 평균 대비 거래량 증가
  - 단발 급등은 clip/winsorize로 과대 반영 방지

### `volume_ratio_20d`
- 정의: 당일 `volume / 최근 20일 평균 volume`
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 처리 규칙:
  - 코드별 rolling mean 기준
  - `0.0 ~ 5.0` clip
  - 일자별 winsorize 적용
- 해석:
  - 중기 평균 대비 거래량 확장 여부를 나타냄

### `value_ratio_5d`
- 정의: 당일 거래대금 근사치 `close * volume`을 최근 5일 평균 거래대금으로 나눈 값
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 처리 규칙:
  - 현재 원천 `value`가 비어 있어 `close * volume` 근사 사용
  - `0.0 ~ 5.0` clip
  - 일자별 winsorize 적용
- 해석:
  - 가격 상승과 거래량 증가가 동시에 반영된 단기 거래대금 확장 신호

### `value_ratio_20d`
- 정의: 당일 거래대금 근사치 `close * volume`을 최근 20일 평균 거래대금으로 나눈 값
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 처리 규칙:
  - 현재 원천 `value`가 비어 있어 `close * volume` 근사 사용
  - `0.0 ~ 5.0` clip
  - 일자별 winsorize 적용
- 해석:
  - 중기 기준 거래대금 확장 여부를 보여줌

### `volume_score`
- 정의:
  - `0.40 * percentile(volume_ratio_5d)`
  - `0.40 * percentile(volume_ratio_20d)`
  - `0.20 * percentile(volume)`
- 범위: `0 ~ 100`
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 해석:
  - 최근 평균 대비 거래량 증가를 점수화
  - 단순 절대 거래량뿐 아니라 최근 확장 강도를 같이 반영

### `liquidity_score`
- 정의:
  - `0.35 * percentile(vol_ma_20)`
  - `0.25 * percentile(volume)`
  - `0.20 * percentile(value_ratio_20d)`
  - `0.20 * percentile(value_ratio_5d)`
- 범위: `0 ~ 100`
- 생성 파일: [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 생성 함수: `build_features()`
- 해석:
  - 저유동성 종목을 감점하고,
  - 최근 거래대금/거래량이 충분한 종목을 상대적으로 높게 평가
- 주의:
  - production `ranking_builder.py`는 별도 `liquidity_score`를 다시 계산하므로,
    현재 이 컬럼은 feature 파이프라인/모델 입력 관점의 liquidity score입니다.

## Flow Features

### Status
- 아래 flow feature는 현재 **미구현** 상태입니다.
- 검토 문서: [flow_data_source_review.md](/d:/ai/Lee_trader/doc/flow_data_source_review.md)

미구현 항목:

- `foreign_net_buy_5d`
- `foreign_net_buy_20d`
- `institution_net_buy_5d`
- `institution_net_buy_20d`
- `foreign_flow_score`
- `institution_flow_score`
- `smart_money_score`

미구현 이유:

- 현재 프로젝트에는 종목별 외국인/기관 순매수 원천 데이터가 없습니다.
- `pykrx` 투자자 수급 함수는 로컬 런타임 테스트에서 안정적으로 동작하지 않았습니다.
- KIS는 가격 데이터만 구현돼 있고 투자자별 수급 endpoint는 아직 코드화되지 않았습니다.
