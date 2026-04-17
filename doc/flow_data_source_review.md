# Flow Data Source Review

- generated_at: 2026-03-18 10:15:00
- basis: current code, current DB/CSV, local runtime checks
- decision: per-stock foreign/institution flow features are **not added in this turn**

## Review Scope

검토 대상:

- KIS API
- `pykrx`
- 기존 DB
- 기존 CSV

검토 목적:

- 종목별 외국인/기관 순매수 데이터를 현재 프로젝트에서 안정적으로 확보할 수 있는지 확인
- 실제 사용 가능한 소스만 feature 파이프라인에 반영

## 1. Existing DB / CSV

### What exists

현재 프로젝트에 이미 존재하는 수급 관련 컬럼은 아래뿐입니다.

| 위치 | 컬럼 | 수준 | 해석 |
| --- | --- | --- | --- |
| `data/market_status.csv` | `foreign_net_5d` | 시장 레벨 | KOSPI 외국인 흐름 proxy |
| DB `market_status` | `foreign_net_5d` | 시장 레벨 | ranking regime용 |
| `data/ranking_final.csv` | `market_foreign_5d` | 시장 레벨 | `ranking_builder.py`에서 market status를 복사한 값 |

### What does not exist

현재 아래 데이터는 없습니다.

- 종목별 `foreign_net_buy`
- 종목별 `institution_net_buy`
- 종목별 수급 이력 테이블
- 종목별 수급 CSV

추가 확인 결과:

- SQLite `trades` 테이블: 없음
- Postgres `trades` 테이블: 있음
  - 하지만 컬럼은 `trade_id, date, side, code, name, market, sector, qty, price, amount, fee, memo, created_at`
  - 이는 **사용자 체결 기록**이지 시장 외국인/기관 수급 데이터가 아님

판단:

- 기존 DB/CSV는 종목별 외국인/기관 수급 feature의 소스로 사용할 수 없음

## 2. KIS API

### 현재 프로젝트 구현 상태

현재 KIS 연동 코드는 아래에만 있습니다.

- `python/download_prices_kis.py`
  - 일봉 OHLCV 조회
- `python/fetch_top_universe.py`
  - 종목/메타 정보 보강

현재 프로젝트 안에는 아래가 없습니다.

- 외국인/기관 투자자별 거래실적 조회 함수
- 수급 데이터를 저장하는 전용 테이블
- 수급 데이터를 feature로 변환하는 단계

### 환경 상태

- `.env`에는 `KIS_BASE_URL`, `KIS_APP_KEY`, `KIS_APP_SECRET`가 존재함
- 하지만 현재 코드에는 투자자별 순매수 endpoint를 호출하는 구현이 없음

판단:

- KIS는 **잠재적 소스**이지만, 현재 프로젝트 기준으로는 미구현 상태
- 공식 endpoint / 응답 스키마 검증 없이 바로 채택하면 추측 구현이 되므로 이번 턴에서는 채택하지 않음

## 3. pykrx

### 런타임 확인

`pykrx` 패키지는 설치돼 있고 아래 함수가 노출됩니다.

- `get_market_net_purchases_of_equities`
- `get_market_net_purchases_of_equities_by_ticker`
- `get_market_trading_value_by_investor`
- `get_market_trading_volume_by_investor`

### 실제 호출 결과

로컬 런타임 테스트 결과:

- `get_market_net_purchases_of_equities('20240304','20240308','KOSPI')`
  - 결과: `Empty DataFrame`
- `get_market_trading_value_by_investor('20240304','20240308','005930')`
  - 결과: `KeyError('거래대금')`
- `get_market_trading_volume_by_investor('20240304','20240308','005930')`
  - 결과: `KeyError('거래량')`

판단:

- 현재 환경의 `pykrx` 투자자 수급 함수는 안정적으로 동작하지 않음
- 반환 스키마가 깨져 있어 production feature source로 채택 불가

## 4. Source Decision

| 소스 | 상태 | 채택 여부 | 이유 |
| --- | --- | --- | --- |
| 기존 DB / CSV | 시장 레벨 proxy만 존재 | 미채택 | 종목별 외국인/기관 순매수 없음 |
| KIS API | 자격 증명은 있으나 투자자 수급 endpoint 미구현 | 미채택 | 현재 코드 기준으로 검증된 구현 없음 |
| `pykrx` | 패키지는 있으나 런타임 실패 | 미채택 | empty result / schema error |

## 5. Implementation Decision

이번 턴에서는 아래 feature를 **추가하지 않았습니다**.

- `foreign_net_buy_5d`
- `foreign_net_buy_20d`
- `institution_net_buy_5d`
- `institution_net_buy_20d`
- `foreign_flow_score`
- `institution_flow_score`
- `smart_money_score`

이유:

- 실제 사용 가능한 종목별 외국인/기관 수급 소스를 확인하지 못했기 때문
- 추정치나 시장 레벨 proxy를 종목 feature처럼 쓰면 설명성과 검증 가능성이 깨짐

## 6. Recommended Next Step

가장 안전한 다음 단계는 아래입니다.

1. KIS 공식 투자자별 거래실적 endpoint를 별도 검증
2. `python/fetch_flow_data.py` 같은 전용 수집 스텝 추가
3. DB `flow_daily` 같은 원천 테이블 신설
4. 그 뒤 `feature_builder.py`에서 5일/20일 누적 및 score 계산

## 7. Final Conclusion

현재 프로젝트 기준으로는:

- 시장 레벨 외국인 proxy는 있음
- 종목 레벨 외국인/기관 수급은 없음
- `pykrx`는 현재 런타임에서 신뢰할 수 없음
- KIS는 구현이 없어 이번 턴에 채택할 수 없음

따라서 이번 요청은 **1단계 점검 결과만 확정**, **2단계 feature 추가는 보류**가 맞습니다.
