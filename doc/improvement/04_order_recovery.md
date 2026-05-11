# 4차 과제: submit_unknown + broker lookup 복구

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 없음 (1차와 독립적으로 진행 가능)
> 다음 과제: 없음 (독립 완결)

---

## 목적

자동매매에서 가장 위험한 상태는 단순 실패가 아니라
**"주문이 성공했는지 실패했는지 모르는 상태"** 이다.

현재 `rule_order_submitter.py`를 보면 타임아웃·OSError·5xx 등
"요청이 KIS에 도달했을 수 있는 예외"가 발생할 경우
`order_block_reason`에 `order_submit_failed_possibly_sent` 태그는 붙지만
`order_status`는 그냥 `"failed"`로 저장된다.

이 태그를 읽어서 다음 사이클에 broker lookup을 트리거하는
로직이 존재하지 않는다. 이 과제에서 그 흐름을 만든다.

---

## 현재 주문 상태값 구조 (소스 기준)

`rule_order_submitter.py`에서 사용 중인 상태값:

| 상태값 | 발생 시점 |
|---|---|
| `planned` | order_action 없는 항목 |
| `blocked` | guard에서 차단된 항목 |
| `debug_skipped` | debug_trade_mode에서 건너뜀 |
| `submitted` | KIS API 정상 응답 |
| `failed` | 예외 발생 (possibly_sent 포함하여 통합) |

`rule_order_fill_sync.py`에서 추가 전환되는 상태값:

| 상태값 | 발생 시점 |
|---|---|
| `unfilled` | fill check 후 미체결 확인 |
| `partial_filled` | 부분 체결 확인 |
| `canceled` | 미체결 취소 완료 |

---

## 핵심 문제점

`_is_retryable_order_error()`는 이미 타임아웃·5xx를
"KIS에 도달했을 수 있으므로 재시도 불가"로 정확히 분류한다.
그런데 이 경우 `order_status = "failed"`로 저장될 뿐,
다음 사이클에서 실제 접수 여부를 확인하는 흐름이 없다.

```python
# rule_order_submitter.py 현재 코드 (예외 처리 부분)
if not retryable and not isinstance(exc, (KISAuthError, ConnectionRefusedError)):
    failure_tag = "order_submit_failed_possibly_sent"  # 태그는 있음
row["order_status"] = "failed"                         # 하지만 상태는 failed로 동일
row["reconciliation_status"] = "submit_failed"
```

---

## 추가할 주문 상태값

```
submit_unknown      : 타임아웃·OSError·5xx 예외 발생.
                      요청이 KIS에 도달했을 수 있으므로
                      다음 사이클에서 반드시 broker lookup 필요.

reconcile_required  : lookup을 시도했으나 매칭 실패.
                      운영자 수동 확인 필요.
```

### 전체 상태값 목록 (변경 후)

```
planned             : order_action 없음
blocked             : guard 차단
debug_skipped       : debug 모드 건너뜀
submitted           : KIS 정상 접수 확인
failed              : 명확한 실패 (4xx, 인증 오류, ConnectionRefused)
submit_unknown      : 접수 여부 불명확 (타임아웃, OSError, 5xx)
unfilled            : fill check 후 미체결
partial_filled      : 부분 체결
canceled            : 취소 완료
reconcile_required  : broker lookup 시도했으나 매칭 실패, 수동 확인 필요
```

---

## 변경 방향

### Step 1. rule_order_submitter.py 수정

예외 처리 블록에서 `possibly_sent` 여부를 판별하여
`submit_unknown`과 `failed`를 분리한다.

판별 기준:

```python
# _is_retryable_order_error() 기준
# retryable=False AND (TimeoutError, OSError, KISHTTPError(5xx))
# → submit_unknown

# retryable=False AND (KISAuthError, ConnectionRefusedError, 4xx)
# → failed (기존 동일)
```

변경 후 코드 방향:

```python
except Exception as exc:
    retryable, _ = _is_retryable_order_error(exc)
    possibly_sent = (
        not retryable
        and not isinstance(exc, (KISAuthError, ConnectionRefusedError))
    )
    if possibly_sent:
        row["order_status"] = "submit_unknown"
        row["reconciliation_status"] = "submit_unknown_pending_lookup"
        if _flag("RULE_NOTIFY_ON_SUBMIT_UNKNOWN", "1"):
            notify_warning(
                title="주문 상태 불명확",
                message=f"code={code} side={side} qty={qty} 접수 여부 확인 필요",
                details={...}
            )
    else:
        row["order_status"] = "failed"
        row["reconciliation_status"] = "submit_failed"
```

### Step 2. rule_order_fill_sync.py 수정

`submit_unknown` 상태 항목을 broker lookup 대상에 포함한다.

현재 fill_sync는 `order_status == "submitted"` AND
`broker_order_id` 있는 항목만 조회한다.

`submit_unknown`은 `broker_order_id`가 없을 수 있으므로
KIS 당일 주문내역을 종목코드·날짜·수량으로 매칭하는
별도 lookup 함수를 추가한다.

```python
def _lookup_unknown_orders(items, client, account, as_of_date):
    """
    submit_unknown 상태 항목에 대해
    KIS 당일 주문내역을 조회하여 접수 여부를 확인한다.

    매칭 기준: code + side + ord_qty + as_of_date
    """
```

lookup 결과에 따른 상태 전환:

| 결과 | order_status | reconciliation_status |
|---|---|---|
| 매칭 1건 성공 | submitted | lookup_matched_submitted |
| 매칭 실패 | reconcile_required | lookup_no_match |
| 매칭 복수 | reconcile_required | lookup_ambiguous_match |
| 조회 API 실패 | submit_unknown 유지 | lookup_api_failed |

`reconcile_required` 상태가 다음 사이클에서도 해소되지 않으면
CRITICAL 알림 발송.

### Step 3. notifier 연결

```python
# submit_unknown 발생 시 (WARNING)
notify_warning(
    title="[RULE] 주문 상태 불명확 - 확인 필요",
    message=f"{as_of_date} | code={code} | side={side} | qty={qty}",
    details={
        "code": code,
        "name": name,
        "side": side,
        "order_qty": qty,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "as_of_date": as_of_date,
        "action_required": "다음 사이클 broker lookup 자동 시도. 미해소 시 수동 확인 필요."
    }
)

# reconcile_required 지속 시 (CRITICAL)
notify_critical(
    title="[RULE] 주문 미확인 지속 - 즉시 수동 확인 필요",
    message=f"{as_of_date} | code={code} | side={side} | broker lookup 실패",
    details={...}
)
```

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 수정 | python/rule_order_submitter.py | submit_unknown 상태 분리, notifier 연결 |
| 수정 | python/rule_order_fill_sync.py | submit_unknown lookup 흐름 추가 |

### 수정 금지 파일

- python/kis_client.py
- python/kis_live_account.py
- python/notifier.py (기존 notify_warning, notify_critical 그대로 사용)

---

## 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| RULE_UNKNOWN_ORDER_LOOKUP_ENABLED | true | submit_unknown broker lookup 활성화 |
| RULE_UNKNOWN_ORDER_MAX_LOOKUP_DAYS | 1 | lookup 대상 최대 경과일 수 |
| RULE_NOTIFY_ON_SUBMIT_UNKNOWN | true | submit_unknown 발생 시 WARNING 알림 |
| RULE_NOTIFY_ON_RECONCILE_REQUIRED | true | reconcile_required 지속 시 CRITICAL 알림 |

---

## 검증 케이스

| # | 조건 | 기대 결과 |
|---|---|---|
| 1 | TimeoutError 예외 발생 | order_status=submit_unknown |
| 2 | KISHTTPError(500) 발생 | order_status=submit_unknown |
| 3 | KISHTTPError(400) 발생 | order_status=failed (기존 동일) |
| 4 | KISAuthError 발생 | order_status=failed (기존 동일) |
| 5 | submit_unknown + lookup 1건 매칭 | order_status=submitted, broker_order_id 저장 |
| 6 | submit_unknown + lookup 매칭 없음 | order_status=reconcile_required |
| 7 | submit_unknown + lookup 복수 매칭 | order_status=reconcile_required |
| 8 | submit_unknown 발생 | notify_warning 발송 확인 |
| 9 | reconcile_required 다음 사이클 지속 | notify_critical 발송 확인 |

---

## 주의사항

- KIS API 호출을 테스트 목적으로 실행하지 말 것
- 실계좌 주문 로직을 직접 실행하지 말 것
- broker_order_id 없는 항목의 lookup은
  종목코드·날짜·수량 매칭으로 처리하되
  복수 매칭 시 reconcile_required로 보수적 처리할 것
- 기존 submitted 상태 항목의 fill sync 흐름은 변경하지 말 것
- failed 상태의 기존 동작은 변경하지 말 것

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
