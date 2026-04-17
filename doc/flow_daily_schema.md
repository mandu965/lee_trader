# flow_daily Schema

- generated_at: 2026-03-18
- source_of_truth: `schema.sql`
- migration_status: no dedicated migration; apply via `schema.sql`

## 목적

`flow_daily`는 KIS 종목별 투자자 수급 원천을 저장하는 정규화 테이블입니다.

v1 기준 grain:

- `(date, code, investor_type)`

즉 같은 종목/같은 날짜라도 투자자 유형별로 별도 row를 가집니다.

## 컬럼 설계

### 필수 컬럼

| column | type | null | 설명 |
| --- | --- | --- | --- |
| `date` | `date` | no | 영업일 |
| `code` | `varchar(6)` | no | 6자리 종목코드 |
| `investor_type` | `varchar(32)` | no | v1 표준 투자자 유형 |
| `net_buy_amount` | `numeric` | yes | 순매수 거래대금 |
| `net_buy_volume` | `numeric` | yes | 순매수 수량 |
| `raw_payload_hash` | `varchar(64)` | no | raw payload 해시 |
| `collected_at` | `timestamptz` | no | 수집 시각 |

### 권장 컬럼

| column | type | null | 설명 |
| --- | --- | --- | --- |
| `source_endpoint` | `varchar(128)` | no | 원천 endpoint path |
| `market_div_code` | `varchar(8)` | no | KIS 시장 구분 코드 |
| `input_date` | `varchar(8)` | no | 호출에 사용한 입력 일자 |
| `tr_id` | `varchar(32)` | no | KIS TR ID |
| `raw_payload_json` | `jsonb` | yes | 응답 body 또는 raw payload |
| `created_run_id` | `bigint` | yes | 파이프라인 run 연계용 |

### 운영 보완 컬럼

| column | type | null | 설명 |
| --- | --- | --- | --- |
| `fetch_status` | `varchar(32)` | yes | `success`, `partial`, `error` 등 상태값 |
| `error_code` | `varchar(64)` | yes | KIS 오류 코드 또는 내부 오류 코드 |
| `error_message` | `text` | yes | 오류 메시지 |
| `is_partial_page` | `boolean` | no | 연속조회 중 부분 page 여부 |

## 제약과 인덱스

PK:

- `PRIMARY KEY (date, code, investor_type)`

인덱스:

- `idx_flow_daily_code_date_desc ON flow_daily(code, date DESC)`
- `idx_flow_daily_date_investor_type ON flow_daily(date, investor_type)`

의도:

- 종목별 시계열 조회 최적화
- 일자별 투자자 유형 집계 최적화
- upsert key를 PK와 일치시켜 적재 로직 단순화

## v1 투자자 유형

v1에서 정규화 대상으로 준비하는 값:

- `foreign`
- `institution`

세부 투자자 그룹은 raw payload에는 남기되, 정규화 테이블에서 바로 확장하지 않습니다.

## 적재/갱신 원칙

권장 upsert key:

- `(date, code, investor_type)`

권장 갱신 원칙:

- 동일 key 재수집 시 `raw_payload_hash`가 변경되면 row 갱신
- 부분 page 수집이면 `is_partial_page=true`
- 종목 단위 실패는 `fetch_status`, `error_code`, `error_message`로 추적 가능

## 구현 메모

이번 변경은 `schema.sql` 기준 정의만 추가한 것입니다. 별도 migration 파일은 없으므로 신규 환경은 `schema.sql` 재적용으로 반영하고, 기존 환경은 운영 절차에 맞춰 DDL을 수동 반영해야 합니다.
