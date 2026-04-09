# KIS 투자자별 수급 수집기 설계

- generated_at: 2026-03-18
- scope: KIS 공식 투자자별 수급 endpoint 기반 종목별 일별 수급 수집 설계
- status: design_only

## 1. 목적

현재 프로젝트에는 종목별 외국인/기관 순매수 원천이 없습니다. 기존 코드 기준으로는 시장 레벨 proxy만 존재합니다.

- 시장 레벨 proxy: `python/fetch_market_data.py`
- 종목별 수급 원천: 미구현

이번 설계는 추측 구현을 피하고, KIS 공식 샘플 코드에서 확인된 endpoint와 현재 프로젝트 구조를 기준으로 이후 구현 가능한 수준의 수집기 설계를 고정하는 것이 목적입니다.

## 2. 현재 프로젝트와의 연결 지점

### 2.1 기존 인증 재사용 가능성

현재 프로젝트는 이미 KIS REST 인증을 직접 구현해 사용 중입니다.

- 파일: `python/download_prices_kis.py`
- 인증 함수: `_kis_get_token(base_url, app_key, app_secret)`
- 가격 조회 함수: `_kis_fetch_daily_prices(...)`
- 사용 env:
  - `KIS_BASE_URL`
  - `KIS_APP_KEY`
  - `KIS_APP_SECRET`

따라서 수급 수집기는 KIS 공식 샘플의 `kis_auth.py`를 그대로 끌어오기보다, 현재 프로젝트의 `_kis_get_token(...)` 패턴을 공용 helper로 분리해 재사용하는 방향이 가장 자연스럽습니다.

권장 구조:

- 신규 공용 helper: `python/kis_client.py`
- 재사용 함수:
  - `get_kis_access_token()`
  - `kis_get(url_path, tr_id, params, *, tr_cont="")`

이렇게 하면 `download_prices_kis.py`와 향후 `download_flows_kis.py`가 동일 인증 경로를 쓰게 됩니다.

### 2.2 파이프라인 삽입 위치

현재 파이프라인 순서는 `python/run_pipeline.py`의 `STEPS`에 정의되어 있습니다.

핵심 관련 단계:

1. `download_prices_kis`
2. `clean_prices`
3. `create_adjusted_prices`
4. `feature_builder`

수급 수집기는 아래 둘 중 하나가 적절합니다.

- 권장: `create_adjusted_prices` 다음, `feature_builder` 이전
- 대안: `download_prices_kis` 바로 다음

권장안이 더 나은 이유:

- 가격/거래대금 feature와 같은 날짜 축으로 후처리하기 쉽습니다.
- `feature_builder.py`가 현재 `(date, code)` grain의 feature 병합 지점이기 때문입니다.

## 3. 대상 endpoint 목록

아래 목록은 모두 KIS 공식 샘플 코드 또는 공식 함수 래퍼에서 확인한 항목만 적었습니다.

### 3.1 1차 채택: 종목별 투자자매매동향(일별)

- 용도: 종목별 일별 외국인/기관 순매수의 기본 원천
- 공식 샘플 파일:
  - `_tmp_kis_openapi/examples_llm/domestic_stock/investor_trade_by_stock_daily/investor_trade_by_stock_daily.py`
  - `_tmp_kis_openapi/examples_llm/domestic_stock/investor_trade_by_stock_daily/chk_investor_trade_by_stock_daily.py`
  - `_tmp_kis_openapi/examples_user/domestic_stock/domestic_stock_functions.py`
- API URL:
  - `/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily`
- TR ID:
  - `FHPTJ04160001`
- 채택 판단:
  - 종목별
  - 일별
  - 외국인/기관 순매수 수량과 거래대금 필드가 공식 샘플에 명시됨
  - v1 수집 backbone으로 가장 적합

### 3.2 보조 검증용: 주식현재가 투자자

- 용도: 당일 snapshot 검증 또는 디버깅
- 공식 샘플 파일:
  - `_tmp_kis_openapi/examples_llm/domestic_stock/inquire_investor/inquire_investor.py`
  - `_tmp_kis_openapi/examples_user/domestic_stock/domestic_stock_functions.py`
- API URL:
  - `/uapi/domestic-stock/v1/quotations/inquire-investor`
- TR ID:
  - `FHKST01010900`
- 채택 판단:
  - historical daily backbone으로는 부적합
  - spot check / UI 실시간 보조용으로만 적합

### 3.3 보조 후보: 투자자동향 추정가집계

- 용도: 장중 추정 수급 참고
- 공식 샘플 파일:
  - `_tmp_kis_openapi/examples_llm/domestic_stock/investor_trend_estimate/investor_trend_estimate.py`
  - `_tmp_kis_openapi/examples_user/domestic_stock/domestic_stock_functions.py`
- API URL:
  - `/uapi/domestic-stock/v1/quotations/investor-trend-estimate`
- TR ID:
  - `HHPTJ04160200`
- 채택 판단:
  - 샘플 설명상 장중 추정치 성격
  - 일별 확정 수급 적재용 원천으로는 부적합
  - 실시간 보강이 필요해질 때 별도 검토

## 4. 인증 방식과 필요한 env 변수

### 4.1 인증 방식

현재 프로젝트 기준 인증 방식은 OAuth access token 발급 후 Bearer token을 붙이는 구조입니다.

- 토큰 발급 endpoint:
  - `POST /oauth2/tokenP`
- 현재 구현:
  - `python/download_prices_kis.py::_kis_get_token`

요청 body:

```json
{
  "grant_type": "client_credentials",
  "appkey": "<KIS_APP_KEY>",
  "appsecret": "<KIS_APP_SECRET>"
}
```

### 4.2 필요한 env 변수

현재 코드 기준 확정 env:

- `KIS_BASE_URL`
- `KIS_APP_KEY`
- `KIS_APP_SECRET`

선택 env 제안:

- `KIS_REQUEST_TIMEOUT_SEC`
- `KIS_MAX_RETRY`
- `KIS_FLOW_LOOKBACK_DAYS`

선택 env는 현재 코드에는 없고, 수집기 구현 시 추가하는 운영 편의 설정입니다.

## 5. 요청 파라미터 정의

## 5.1 `investor-trade-by-stock-daily`

공식 샘플 기준 파라미터:

- `FID_COND_MRKT_DIV_CODE`
  - 샘플 주석 확인값: `J:KRX`, `NX:NXT`, `UN:통합`
- `FID_INPUT_ISCD`
  - 6자리 종목코드
- `FID_INPUT_DATE_1`
  - 기준 일자, 예시 `20250812`
- `FID_ORG_ADJ_PRC`
  - 공식 샘플 예시는 공란 전달
  - 실제 business semantics는 현재 프로젝트에서 미확정
- `FID_ETC_CLS_CODE`
  - 공식 샘플 예시는 공란 전달
  - 실제 business semantics는 현재 프로젝트에서 미확정

v1 권장 파라미터 고정값:

- `FID_COND_MRKT_DIV_CODE="J"`
- `FID_INPUT_ISCD=<code>`
- `FID_INPUT_DATE_1=<yyyymmdd>`
- `FID_ORG_ADJ_PRC=""`
- `FID_ETC_CLS_CODE=""`

주의:

- 공식 샘플은 연속조회(`tr_cont`)를 사용합니다.
- 샘플 구현은 `res.getHeader().tr_cont in ["M", "F"]`인 경우 다음 페이지를 재호출합니다.
- 구현 시 이 동작을 그대로 반영해야 합니다.

## 5.2 `inquire-investor`

공식 샘플 기준 파라미터:

- `FID_COND_MRKT_DIV_CODE`
- `FID_INPUT_ISCD`

v1에서는 적재 backbone이 아니므로 직접 적재 대상에서 제외하고, 진단/샘플 검증용 호출만 허용하는 것이 적절합니다.

## 5.3 `investor-trend-estimate`

공식 샘플 기준 파라미터:

- `MKSC_SHRN_ISCD`

이 endpoint도 v1 적재 대상에서는 제외합니다.

## 6. 응답 필드 정의

## 6.1 v1 적재에 사용하는 확정 필드

아래 필드는 공식 checker 샘플에서 실제 컬럼명으로 확인됐고, v1 적재 표준화에 직접 사용 가능합니다.

출처:

- `_tmp_kis_openapi/examples_llm/domestic_stock/investor_trade_by_stock_daily/chk_investor_trade_by_stock_daily.py`

기본 식별/시세 필드:

- `stck_bsop_date`
- `stck_clpr`
- `stck_oprc`
- `stck_hgpr`
- `stck_lwpr`
- `acml_vol`
- `acml_tr_pbmn`
- `rprs_mrkt_kor_name`

외국인 순매수 필드:

- `frgn_ntby_qty`
- `frgn_ntby_tr_pbmn`

기관계 순매수 필드:

- `orgn_ntby_qty`
- `orgn_ntby_tr_pbmn`

v1 표준화 매핑:

| source field | standardized meaning |
| --- | --- |
| `stck_bsop_date` | 영업일 |
| `frgn_ntby_qty` | 외국인 순매수 수량 |
| `frgn_ntby_tr_pbmn` | 외국인 순매수 거래대금 |
| `orgn_ntby_qty` | 기관계 순매수 수량 |
| `orgn_ntby_tr_pbmn` | 기관계 순매수 거래대금 |

## 6.2 확인됐지만 v1 표준화 대상에서 제외하는 추가 필드

공식 checker 샘플에는 아래와 같은 세부 주체 필드도 보입니다.

- `frgn_reg_ntby_qty`
- `frgn_nreg_ntby_qty`
- `prsn_ntby_qty`
- `scrt_ntby_qty`
- `ivtr_ntby_qty`
- `bank_ntby_qty`
- `insu_ntby_qty`
- `mrbn_ntby_qty`
- `fund_ntby_qty`
- `etc_ntby_qty`
- `frgn_reg_ntby_pbmn`
- `frgn_nreg_ntby_pbmn`
- `prsn_ntby_tr_pbmn`
- `scrt_ntby_tr_pbmn`
- `ivtr_ntby_tr_pbmn`
- `bank_ntby_tr_pbmn`
- `insu_ntby_tr_pbmn`
- `mrbn_ntby_tr_pbmn`
- `fund_ntby_tr_pbmn`
- `etc_ntby_tr_pbmn`

이 필드들은 공식 샘플 코드에서는 확인됐지만, 현재 프로젝트의 v1 요구사항은 외국인/기관 중심입니다. 따라서 v1에서는 raw payload 보존만 하고, 정규화 테이블에는 `foreign`, `institution`만 적재하는 것이 적절합니다.

## 6.3 미확정 항목

다음 항목은 샘플 코드에서 값 전달 또는 컬럼 존재는 확인되지만, 현재 프로젝트 문서 수준에서 business semantics를 확정하지 않습니다.

- `FID_ORG_ADJ_PRC`
- `FID_ETC_CLS_CODE`
- `output1`과 `output2`의 역할 차이 전체

정책:

- 이 항목들은 설계 문서에서 미확정으로 유지
- 실제 구현 전 샌드박스 호출 1회와 응답 스키마 캡처 후 확정

## 7. 종목별/일자별 적재 grain 제안

권장 적재 grain:

- 1 row = `(date, code, investor_type)`

권장 investor_type 값:

- `foreign`
- `institution`

정규화 예시:

| date | code | investor_type | net_buy_amount | net_buy_volume |
| --- | --- | --- | ---: | ---: |
| 2026-03-18 | 005930 | foreign | 1234567890 | 54321 |
| 2026-03-18 | 005930 | institution | -234567890 | -12345 |

이 grain을 권장하는 이유:

- `feature_builder.py`에 `(date, code)` 단위로 pivot 후 병합하기 쉽습니다.
- investor type 확장이 쉽습니다.
- raw payload 구조 변화가 있어도 정규화 계층의 안정성이 높습니다.

## 8. 원천 테이블 스키마 제안

필수 스키마:

```sql
CREATE TABLE flow_daily (
    date date NOT NULL,
    code varchar(6) NOT NULL,
    investor_type varchar(32) NOT NULL,
    net_buy_amount numeric,
    net_buy_volume numeric,
    raw_payload_hash varchar(64) NOT NULL,
    collected_at timestamptz NOT NULL
);
```

권장 추가 컬럼:

- `source_endpoint varchar(128) not null`
- `market_div_code varchar(8) not null`
- `input_date varchar(8) not null`
- `tr_id varchar(32) not null`
- `raw_payload_json jsonb`
- `created_run_id bigint null`

권장 PK:

- `(date, code, investor_type)`

권장 index:

- `(code, date desc)`
- `(date, investor_type)`

권장 이유:

- raw hash와 raw payload를 같이 두면 재수집 결과 비교가 쉽습니다.
- 동일 날짜 재수집 시 upsert 기준이 명확합니다.

## 9. 재수집 정책

권장 재수집 정책은 아래와 같습니다.

### 9.1 최근 구간 rolling refresh

- 매 실행 시 최근 `20` 거래일 재수집
- 이유:
  - API 일시 실패 복구
  - 페이징 누락 복구
  - 장 마감 직후 미완결 데이터 보정

### 9.2 과거 구간 backfill

- 초기 적재 시 최근 `1~2년` 백필
- 이후에는 append + 최근 20거래일 refresh

### 9.3 upsert 정책

- key: `(date, code, investor_type)`
- 동일 key 재수집 시:
  - `raw_payload_hash`가 같으면 skip 가능
  - 다르면 값 갱신 후 `collected_at` 업데이트

## 10. 휴장일 / 결측 / 에러 처리 정책

## 10.1 휴장일

휴장일에는 synthetic 0 row를 만들지 않습니다.

정책:

- 호출 결과가 빈 데이터이고 해당 날짜가 전체 시장 휴장일로 판단되면 적재 생략
- 휴장 여부 판단은 기존 가격 수집 결과 또는 거래일 캘린더 기준으로 처리

## 10.2 종목별 결측

특정 종목만 빈 응답인 경우:

- 1차: warning log 남기고 적재 생략
- 2차: 최근 거래일 기준 가격 데이터 존재 여부와 대조
- feature 단계에서는 결측을 중립 처리

## 10.3 인증/네트워크 오류

오류 유형을 분리해야 합니다.

- 인증 실패
  - token 발급 실패
  - access token 만료
- HTTP 오류
  - 4xx
  - 5xx
- KIS business error
  - 샘플 기준 `res.isOK()` false
  - `res.getErrorCode()`, `res.getErrorMessage()` 로그 필요
- paging 오류
  - `tr_cont` 이어붙이기 누락

권장 처리:

- 종목 단위 실패는 전체 배치를 중단하지 않고 누적 실패 목록에 기록
- 인증 실패나 전체 rate limit은 배치 중단
- 실패 리포트는 별도 md/csv로 남김

## 11. `feature_builder.py`로 넘길 가공 인터페이스 제안

현재 `feature_builder.py`의 핵심 함수는 아래입니다.

- `load_prices()`
- `build_features(df)`
- `save_features(df_feat)`

수급 feature를 붙이려면 아래 구조가 적절합니다.

### 11.1 신규 로더 제안

- 파일: `python/feature_builder.py`
- 신규 함수 제안:
  - `load_flow_daily() -> pd.DataFrame`
  - `build_flow_features(flow_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame`

### 11.2 표준 wide 인터페이스

`flow_daily`를 `(date, code)` 기준으로 pivot:

- `foreign_net_buy_amount`
- `foreign_net_buy_volume`
- `institution_net_buy_amount`
- `institution_net_buy_volume`

그 다음 price/value 기준 상대화:

- `foreign_net_buy_5d`
- `foreign_net_buy_20d`
- `institution_net_buy_5d`
- `institution_net_buy_20d`
- `foreign_flow_score`
- `institution_flow_score`
- `smart_money_score`

권장 상대화 기준:

- 1순위: 당일 거래대금 대비
- 2순위: 필요 시 시가총액 대비

현재 프로젝트에는 `feature_builder.py` 내부에 `value_traded = close * volume`이 이미 계산됩니다. 따라서 v1에서는 추가 시가총액 테이블 없이도 거래대금 대비 정규화부터 시작할 수 있습니다.

### 11.3 결측 처리 원칙

- 수급 원천이 없는 종목: 0으로 강제하지 말고 `NaN -> 중립 score`
- 최근 5일/20일 rolling 합산 시 유효 관측 수가 부족하면 score는 중립값 사용

## 12. 구현 파일 제안

권장 신규 파일:

- `python/kis_client.py`
  - 토큰 발급, 공통 GET, 재시도, 헤더 처리
- `python/download_flows_kis.py`
  - 종목별 일별 수급 수집
- `python/check_flow_ingestion.py`
  - 적재 건수, 결측률, 최신 일자 검증

권장 기존 파일 수정:

- `python/run_pipeline.py`
  - `feature_builder` 이전에 `download_flows_kis` 추가
- `python/feature_builder.py`
  - flow 로더/feature 생성 함수 추가
- `schema.sql`
  - `flow_daily` 테이블 추가
- `doc/feature_dictionary.md`
  - flow feature 정의 확장

## 13. 구현 순서 체크리스트

1. `python/download_prices_kis.py`의 인증 함수를 공용 `kis_client.py`로 분리한다.
2. `investor-trade-by-stock-daily` 호출기와 paging 처리기를 구현한다.
3. raw 응답 1건을 저장해 `output1/output2` 구조와 실제 사용 필드를 확정한다.
4. `flow_daily` 원천 테이블을 생성한다.
5. universe 종목 기준 최근 20거래일 재수집 배치를 구현한다.
6. `(date, code, investor_type)` upsert를 구현한다.
7. 적재 검증 스크립트에서 일자별 row count와 결측 종목 수를 점검한다.
8. `feature_builder.py`에 flow pivot/rolling/score 계산을 추가한다.
9. `feature_dictionary.md`를 업데이트한다.
10. `run_pipeline.py`에 새 step을 삽입한다.
11. 샘플 종목 3개로 dry run 후 전체 universe로 확장한다.

## 14. 결론

현재 프로젝트에서 종목별 외국인/기관 수급을 추가하려면 KIS 공식 endpoint 중 아래 조합이 가장 타당합니다.

- 기본 원천: `/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily`
- 보조 검증: `/uapi/domestic-stock/v1/quotations/inquire-investor`
- 장중 보강 후보: `/uapi/domestic-stock/v1/quotations/investor-trend-estimate`

핵심 판단은 단순합니다.

- 일별 확정 원천은 `investor-trade-by-stock-daily`
- 현재 프로젝트 인증은 `download_prices_kis.py` 재사용이 가능
- 정규화 적재 grain은 `(date, code, investor_type)`가 가장 안정적
- `feature_builder.py`에는 거래대금 대비 상대화된 5일/20일 누적 수급 feature를 넘기는 구조가 적절

추가 확인이 필요한 부분은 응답의 `output1/output2` 세부 역할과 `FID_ORG_ADJ_PRC`, `FID_ETC_CLS_CODE`의 business semantics뿐입니다. 이 둘은 공식 샘플에서 값 전달은 확인됐지만, 현재 프로젝트 문서 수준에서는 미확정으로 유지하는 것이 맞습니다.
