# KIS Flow Dry Run

- generated_at: 2026-03-18
- status: initial implementation

## 구현 범위

신규 스크립트 `python/download_flows_kis.py`를 추가했습니다.

기본 원천 endpoint:

- `/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily`
- `TR_ID=FHPTJ04160001`

## 지원 인자

```powershell
python python/download_flows_kis.py `
  --codes 005930,000660,035420 `
  --start-date 20260313 `
  --end-date 20260318 `
  --dry-run `
  --save-raw
```

지원 옵션:

- `--codes`
- `--start-date`
- `--end-date`
- `--dry-run`
- `--save-raw`

## 생성 산출물

raw payload 저장 경로:

- `output/kis_flow_raw/*.json`

실패 리포트:

- `output/flow_ingestion_failures.csv`

raw JSON에는 아래를 함께 저장합니다.

- 요청 path / tr_id / params
- 응답 status code / headers
- 응답 body 전체

즉 `output1` / `output2` 구조를 가공하지 않고 그대로 확인할 수 있습니다.

## 에러 처리 정책

- 종목 단위 `KISBusinessError`: 실패 목록에 기록 후 다음 종목 계속
- `KISAuthError`: 배치 중단
- `KISHTTPError`: 배치 중단
- 기타 공통 예외: 배치 중단

## v1 정규화 범위

v1에서는 원천 raw 저장이 우선입니다.

동시에 메모리 상에서만 아래 투자자 유형을 준비합니다.

- `foreign`
- `institution`

사용 필드:

- `frgn_ntby_tr_pbmn`
- `frgn_ntby_qty`
- `orgn_ntby_tr_pbmn`
- `orgn_ntby_qty`
- `stck_bsop_date`

그 외 세부 투자자 필드는 raw payload에만 남기고 v1에서 확정하지 않습니다.

## 주의

공식 샘플에서 확정된 날짜 파라미터는 `FID_INPUT_DATE_1`뿐입니다. 따라서 v1 수집기는 날짜 범위를 한 번에 넘기지 않고, 지정한 기간을 평일 단위로 쪼개서 `FID_INPUT_DATE_1` 반복 호출 방식으로 구현했습니다.
