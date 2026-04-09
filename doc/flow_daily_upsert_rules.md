# flow_daily Upsert Rules

- generated_at: 2026-03-18

## 범위

`python/download_flows_kis.py`의 1차 적재 규칙을 정리한 문서입니다.

원천 endpoint:

- `/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily`

정규화 대상 투자자 유형:

- `foreign`
- `institution`

## 정규화 규칙

### 1. source row 기준

raw payload의 `output2`를 우선 사용하고, 없으면 `output1`을 fallback 합니다.

### 2. date

- 원천 필드: `stck_bsop_date`
- `YYYYMMDD -> YYYY-MM-DD`로 변환해 `flow_daily.date`에 적재

### 3. foreign 매핑

- `net_buy_amount <- frgn_ntby_tr_pbmn`
- `net_buy_volume <- frgn_ntby_qty`

### 4. institution 매핑

- `net_buy_amount <- orgn_ntby_tr_pbmn`
- `net_buy_volume <- orgn_ntby_qty`

### 5. 결측 처리

- `None`, `""`, `"-"`는 결측으로 간주
- `0`은 실제 값으로 유지
- foreign / institution 4개 값이 모두 결측이면 raw는 저장 대상이지만 정규화 row는 만들지 않고 warning 로그를 남김

## raw_payload_json 원칙

`flow_daily.raw_payload_json`에는 의미 미확정 필드를 확정 해석하지 않고 아래만 저장합니다.

- `response_row`
- `mapping_scope = "foreign_institution_v1"`

즉 row 수준 raw 보존이 목적이며, 세부 투자자 의미는 이후 단계에서만 해석합니다.

## raw_payload_hash 규칙

- hash 대상: row 수준 raw JSON
- 알고리즘: `sha256`
- 직렬화: `sort_keys=True`

이 값은 재수집 시 동일 row 여부 확인용입니다.

## fetch_status 규칙

v1 상태값:

- `success`
- `skipped`
- `failed`

현재 구현은 실제 적재 row에 대해 우선 `success`를 사용합니다.

`skipped` / `failed`는 향후 per-row 오류 적재가 필요할 때 확장 가능한 상태값으로 남겨둡니다.

## is_partial_page 규칙

- 현재 page의 응답 헤더 `tr_cont`가 `M` 또는 `F`이면 `true`
- 마지막 page면 `false`

즉 해당 row가 연속조회 중간 page에서 나온 것인지 추적할 수 있습니다.

## upsert 규칙

키:

- `(date, code, investor_type)`

충돌 시 갱신 컬럼:

- `net_buy_amount`
- `net_buy_volume`
- `raw_payload_hash`
- `collected_at`
- `source_endpoint`
- `market_div_code`
- `input_date`
- `tr_id`
- `raw_payload_json`
- `created_run_id`
- `fetch_status`
- `error_code`
- `error_message`
- `is_partial_page`

## DB 사용 원칙

- Postgres 우선
- 프로젝트의 `python/db.py::get_engine()` 재사용
- 별도 CSV fallback은 추가하지 않음

DB 엔진이 없으면 적재를 건너뛰고 경고 로그만 남깁니다.

## 로그 분리

로그 prefix:

- `[fetch]` 수집 단계
- `[normalize]` 정규화 단계
- `[load]` 적재 단계
- `[summary]` 종목별 요약
- `[check]` 적재 후 구조 검증

## 적재 후 확인 포인트

정상 케이스에서는 `date, code` 기준으로 아래 구조가 나와야 합니다.

- `row_count = 2`
- `investor_types = foreign,institution`
