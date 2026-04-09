# KIS Client Refactor

- generated_at: 2026-03-18

## 변경 파일

- `python/kis_client.py`
- `python/download_prices_kis.py`

## 변경 내용

### 1. 공용 KIS 클라이언트 추가

신규 `python/kis_client.py`를 추가해 아래 기능을 공용화했습니다.

- access token 발급
- 기본 헤더 구성
- 공통 GET 호출
- timeout / retry / error handling

구성 요소:

- `KISClient`
- `KISError`
- `KISAuthError`
- `KISHTTPError`
- `KISBusinessError`

### 2. 인증 경로 통합

기존 `python/download_prices_kis.py` 안에 있던 `_kis_get_token(...)` 로직을 제거하고, `KISClient.from_env()` + `issue_access_token()` 경로로 통합했습니다.

재사용 env는 기존과 동일합니다.

- `KIS_BASE_URL`
- `KIS_APP_KEY`
- `KIS_APP_SECRET`

### 3. 공통 GET 호출 적용

기존 가격 조회는 `requests.get(...)`를 직접 호출했지만, 이제는 `KISClient.get(...)`을 사용합니다.

적용 endpoint:

- `/uapi/domestic-stock/v1/quotations/inquire-daily-price`

TR ID:

- `FHKST03010100`

### 4. 에러 처리 개선

아래 오류를 분리해 처리합니다.

- 인증 실패: `KISAuthError`
- HTTP 실패: `KISHTTPError`
- KIS business error: `KISBusinessError`

`download_prices_kis.py`는 기존 동작을 유지하기 위해 이 예외들을 잡아 `None` 반환 + warning log로 흡수합니다. 그래서 상위 fallback 흐름인 `pykrx -> demo prices`는 그대로 유지됩니다.

## 영향 범위

직접 영향:

- KIS 가격 수집 경로

간접 영향:

- 향후 `quotations` 계열 endpoint를 추가할 때 동일 클라이언트를 재사용 가능
- 수급 수집기 설계 문서의 `download_flows_kis.py` 구현 기반이 마련됨

영향 없음:

- CSV 저장 형식
- SQLite `prices_raw` 적재 방식
- pykrx fallback 경로
- demo price fallback 경로

## 검증

실행한 검증:

- `python -m py_compile python/kis_client.py python/download_prices_kis.py`
- import smoke test

검증 목적:

- import 에러 없음
- 기본 문법 오류 없음
- 기존 가격 수집 스크립트가 새 클라이언트를 참조할 수 있음
