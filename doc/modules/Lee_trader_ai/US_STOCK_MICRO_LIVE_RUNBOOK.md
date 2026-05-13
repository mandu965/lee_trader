# US Stock Micro Live Runbook

## 1. Phase 7-6 목적

Phase 7-6은 Micro Live 운영자가 주문 후보, 승인, 주문, 체결, reconciliation, Kill Switch, 차단 로그를 하루 기준으로 한 번에 점검하고 즉시 중단 기준을 판단할 수 있게 하는 운영 리포트 및 장애 대응 단계다.

이 단계는 조회/리포트/알림/중단 권고만 수행한다.

금지 사항:

- 신규 주문 생성 금지
- 주문 전송 금지
- 자동 재주문 금지
- 자동 청산 금지
- 자동 포지션 보정 금지

## 2. 일일 점검 명령어

운영 리포트:

```bash
python scripts/report_us_micro_live_operations.py \
  --trade-date 2026-05-16 \
  --account-id US_LIVE_TEST \
  --format console
```

markdown 리포트:

```bash
python scripts/report_us_micro_live_operations.py \
  --trade-date 2026-05-16 \
  --account-id US_LIVE_TEST \
  --format markdown
```

CSV 산출:

```bash
python scripts/report_us_micro_live_operations.py \
  --trade-date 2026-05-16 \
  --account-id US_LIVE_TEST \
  --format csv
```

일일 운영 점검 wrapper:

```bash
python scripts/run_us_micro_live_daily_check.py \
  --trade-date 2026-05-16 \
  --account-id US_LIVE_TEST \
  --execution-mode MOCK \
  --dry-run
```

reconciliation 확인:

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --dry-run
```

## 3. Health Status 기준

- `HEALTHY`: critical 0, error 0, mismatch 0
- `ATTENTION`: warning 존재, pending/expired approval 존재, stale open order 존재
- `DEGRADED`: error 존재, `ORDER_UNKNOWN` 또는 `SYNC_ERROR` 존재
- `CRITICAL`: reconciliation critical, duplicate order, cash/position mismatch, kill switch trigger condition

## 4. Action Required 기준

- Kill Switch active
- Pre-Trade Check ERROR 존재
- Approval PENDING 존재
- Approval EXPIRED 존재
- `SYNC_ERROR` 존재
- `ORDER_PARTIALLY_FILLED` 존재
- stale `ORDER_OPEN`
- `ORDER_REJECTED`
- `ORDER_UNKNOWN`
- reconciliation mismatch / critical
- daily risk usage 근접
- 동일 block reason 반복

## 5. 장애 분류 기준

- `INFO`: 안전 플래그 차단, 반복 block reason, 승인 대기
- `WARNING`: 승인 만료, stale open order, partial fill, 데이터 일부 누락
- `ERROR`: Pre-Trade Check ERROR, `SYNC_ERROR`, `ORDER_UNKNOWN`, mock/sandbox 조회 실패
- `CRITICAL`: reconciliation position/cash/fill mismatch, duplicate order, kill switch trigger condition

## 6. 장애 대응 절차

### 6-1. Kill Switch 활성 상태

증상:

- active kill switch row 존재

확인 명령어:

```bash
python scripts/report_us_micro_live_operations.py --trade-date YYYY-MM-DD --account-id US_LIVE_TEST --format console
```

확인 SQL:

```sql
SELECT *
FROM risk.us_stock_live_kill_switch
WHERE is_active = true;
```

즉시 조치:

- 활성화 이유 확인
- 의도된 중단인지 운영자 확인
- Micro Live 추가 진행 중단

금지 행동:

- 근거 없이 Kill Switch 해제 금지

후속 조치:

- 해제 전 block log / reconciliation / order 상태 확인

### 6-2. Pre-Trade Check ERROR

증상:

- block log에 error 성격 reason 존재

확인 명령어:

```bash
python scripts/report_us_micro_live_operations.py --trade-date YYYY-MM-DD --account-id US_LIVE_TEST --format markdown
```

확인 SQL:

```sql
SELECT *
FROM risk.us_stock_live_order_block_log
WHERE trade_date = DATE 'YYYY-MM-DD'
  AND account_id = 'US_LIVE_TEST'
  AND severity IN ('ERROR', 'CRITICAL');
```

즉시 조치:

- 원인 reason_code 확인
- 재실행보다 먼저 데이터/정책/kill switch 상태 확인

금지 행동:

- error 원인 미확인 상태에서 주문 생성 또는 전송 금지

후속 조치:

- 동일 error 반복 여부 확인

### 6-3. Approval 만료

증상:

- approval status `EXPIRED`

확인 SQL:

```sql
SELECT *
FROM risk.us_stock_live_order_approval
WHERE trade_date = DATE 'YYYY-MM-DD'
  AND account_id = 'US_LIVE_TEST'
  AND approval_status = 'EXPIRED';
```

즉시 조치:

- 기존 approval 재사용 금지
- Pre-Trade Check부터 다시 시작

금지 행동:

- 만료 approval로 주문 재진행 금지

후속 조치:

- approval 만료 시간 설정이 현실적인지 검토

### 6-4. Approval 승인 후 주문 생성 실패

증상:

- approval은 `APPROVED`인데 micro order가 없거나 `FAILED`

확인 SQL:

```sql
SELECT *
FROM risk.us_stock_live_order_approval
WHERE approval_status = 'APPROVED';
```

즉시 조치:

- micro order event log 확인
- 승인 시점 이후 정책/안전 플래그 변경 여부 확인

금지 행동:

- 승인만 보고 broker 주문이 나갔다고 가정 금지

후속 조치:

- approval event와 micro order event 연결 확인

### 6-5. Mock / Sandbox / Live 전송 실패

증상:

- `FAILED`, `LIVE_FAILED`, `REJECTED`, `LIVE_REJECTED`

확인 SQL:

```sql
SELECT *
FROM live.us_stock_micro_order_request
WHERE request_status IN ('FAILED', 'LIVE_FAILED', 'REJECTED', 'LIVE_REJECTED');
```

즉시 조치:

- response payload, reject reason 확인

금지 행동:

- 자동 재주문 금지

후속 조치:

- client 설정과 안전 플래그 확인

### 6-6. ORDER_UNKNOWN

증상:

- `request_status = ORDER_UNKNOWN`

즉시 조치:

- broker raw status 확인
- status mapper 업데이트 필요 여부 점검

금지 행동:

- unknown 상태를 filled/open으로 임의 해석 금지

후속 조치:

- `utils/us_order_status_mapper.py` 보완 검토

### 6-7. ORDER_PARTIALLY_FILLED

증상:

- 부분 체결 주문 존재

즉시 조치:

- 잔량/체결량/체결 금액 확인
- operator review 상태로 유지

금지 행동:

- 남은 수량 자동 top-up 금지

후속 조치:

- reconciliation으로 포지션 차이 확인

### 6-8. ORDER_OPEN 장기 지속

증상:

- open order가 지정 임계분 이상 유지

즉시 조치:

- stale 여부 확인
- broker/sandbox status 재조회 검토

금지 행동:

- 같은 symbol 중복 주문 금지

후속 조치:

- 주문 취소 정책은 별도 단계에서 검토

### 6-9. ORDER_REJECTED

증상:

- rejected order 존재

즉시 조치:

- reject reason 확인
- approval / precheck / order amount / broker 제한 확인

금지 행동:

- 같은 파라미터로 즉시 반복 주문 금지

후속 조치:

- 동일 reject 반복 시 policy 또는 client 설정 점검

### 6-10. Reconciliation MISMATCH

증상:

- reconciliation status `MISMATCH`

즉시 조치:

- order/fill/position 어느 유형인지 확인
- 수동 검증 대상으로 유지

금지 행동:

- mismatch를 자동 매매로 보정 금지
- 내부 DB를 broker 값으로 즉시 덮어쓰기 금지

후속 조치:

- 다음 운영 리포트까지 반복 여부 추적

### 6-11. Reconciliation CRITICAL

증상:

- internal expected position과 broker position 불일치
- cash/fill mismatch 중 critical 발생

확인 명령어:

```bash
python scripts/reconcile_us_micro_live.py --account-id US_LIVE_TEST --recon-date YYYY-MM-DD --execution-mode MOCK --dry-run
```

즉시 조치:

- Micro Live 중단
- ACCOUNT 또는 GLOBAL Kill Switch 활성화 검토
- 주문/체결/포지션 로그 확인

금지 행동:

- 자동 보정 주문 실행 금지
- 내부 DB를 broker 값으로 즉시 덮어쓰기 금지

후속 조치:

- 원인 분석 완료 전 Phase 8 진입 금지

### 6-12. Daily Risk Usage 한도 초과

증상:

- daily order / buy / failure count limit 근접 또는 초과

즉시 조치:

- 추가 진행 중단
- threshold 도달 원인 확인

금지 행동:

- 한도 우회 목적의 계정/정책 변경 금지

후속 조치:

- risk usage row와 block log 함께 검토

### 6-13. 데이터 누락

증상:

- ranking / approval / order / reconciliation 중 일부 섹션 비어 있음

즉시 조치:

- 테이블 존재 여부와 해당 날짜 데이터 존재 여부 확인

금지 행동:

- 데이터 없음 상태를 정상으로 단정 금지

후속 조치:

- upstream 단계 실행 누락 여부 확인

### 6-14. 브로커 API 장애

증상:

- status query 실패 반복
- sandbox/live client 응답 실패

즉시 조치:

- 외부 API 장애로 간주하고 조회 재시도 전에 상태 고정

금지 행동:

- 실패 상태에서 자동 주문 보정 금지

후속 조치:

- repeated failure면 Kill Switch 검토

## 7. Kill Switch 연동 정책

- critical 1건 이상: `Kill Switch Recommended = YES`
- reconciliation critical: `ACCOUNT` 또는 `GLOBAL` kill 추천
- duplicate order: `GLOBAL` kill 추천
- daily loss/usage 초과: `BUY` 또는 `GLOBAL` kill 추천
- Pre-Trade Check ERROR 반복: `BUY` 또는 `GLOBAL` kill 추천

기본 정책:

- 리포트는 추천만 수행
- 자동 활성화는 명시 옵션이 있을 때만 수행

## 8. 알림 옵션

관련 ENV:

- `US_MICRO_OPS_REPORT_NOTIFY_ENABLED`
- `US_MICRO_OPS_NOTIFY_ON_CRITICAL`
- `US_MICRO_OPS_NOTIFY_ON_ERROR`
- `US_MICRO_OPS_REPORT_OUTPUT_DIR`

알림 실패는 리포트 실패가 아니다.

## 9. Phase 7 완료 체크리스트

```text
[Phase 7 완료 체크리스트]

Mock/Sandbox:
- [ ] Mock 주문 생성/전송/거절/실패/취소 검증 완료
- [ ] Sandbox 주문 생성/전송/상태조회/취소 검증 완료
- [ ] Sandbox가 실계좌가 아님을 확인

Micro Live 안전:
- [ ] 수동 승인 필수
- [ ] Pre-Trade Check 주문 직전 재실행
- [ ] Kill Switch 정상 작동
- [ ] Limit 주문만 허용
- [ ] Market 주문 금지
- [ ] 1회 주문 금액 극소액
- [ ] 1일 신규 BUY 1건 이하
- [ ] SELL 자동화 비활성 또는 수동 승인

주문 상태:
- [ ] broker_order_id 저장 가능
- [ ] 주문 상태 조회 가능
- [ ] 체결 내역 저장 가능
- [ ] 부분 체결 상태 식별 가능
- [ ] 미체결 주문 식별 가능

Reconciliation:
- [ ] 주문 상태 대조 가능
- [ ] 체결 내역 대조 가능
- [ ] 포지션 대조 가능
- [ ] 현금/계좌 대조 기준 정의
- [ ] CRITICAL mismatch 발생 시 Kill Switch 추천 가능

운영:
- [ ] Micro Live 운영 리포트 생성 가능
- [ ] 장애 대응 Runbook 작성 완료
- [ ] Action Required 출력 가능
- [ ] 알림 옵션 준비
- [ ] 운영자가 수동으로 중단/확인 가능

안전:
- [ ] 자동 재주문 없음
- [ ] 자동 보정 주문 없음
- [ ] 기존 국내주식 실매매 로직 영향 없음
```

## 10. Phase 8 진입 조건

```text
[Phase 8 진입 조건]

성과/운영:
- [ ] Phase 4 Backtest / Forward Test 결과가 존재
- [ ] Phase 5 Paper Trading이 20~60거래일 이상 안정 운영
- [ ] Phase 7 Micro Live가 최소 10~20건 이상 극소액 주문으로 검증됨
- [ ] 주문 실패/거절/부분체결 대응 절차가 검증됨
- [ ] Reconciliation CRITICAL이 반복되지 않음
- [ ] Paper / Forward / Micro Live 결과 괴리 분석 완료

안전:
- [ ] Pre-Trade Check 안정적으로 동작
- [ ] Kill Switch 실제로 주문 차단 가능
- [ ] 수동 승인 플로우 정상 작동
- [ ] 주문 상태 동기화 정상 작동
- [ ] 계좌/포지션 대조 정상 작동
- [ ] 운영 리포트와 알림 정상 작동

자동화 제한:
- [ ] Phase 8 초기 자동화 범위는 BUY 1일 1건 이하
- [ ] 주문 금액은 여전히 소액
- [ ] SELL 자동화는 별도 검증 후 제한 적용
- [ ] Market 주문 금지
- [ ] Limit 주문만 허용
- [ ] 시장 급락일 BUY 금지
```

Phase 8은 전면 자동매매가 아니라 제한적 자동매매 운영화 단계입니다.
Phase 8 진입 전 Micro Live의 주문/체결/대조/장애 대응이 충분히 검증되어야 합니다.
