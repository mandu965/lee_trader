# RULE 자동매매 미체결 원인 확인 목록

작성일: 2026-05-08
증상: RULE live 계좌 주문 미체결 / 주요 원인 코드 `order_qty_zero`

---

## 확인 순서

### 1. 환경변수 확인 (서버 `.env`)

```bash
grep RULE_AUTO_ADJUST .env
grep RULE_MAX_ORDER_AMOUNT .env
grep RULE_NEW_ENTRY_WEIGHT .env
grep RULE_MIN_CASH_WEIGHT .env
```

| 변수 | 기대값 | 이 값이면 문제 |
|---|---|---|
| `RULE_AUTO_ADJUST_MINIMUM_SHARES` | `1` (기본값) | `0`이면 고가 종목 qty=0 발생 |
| `RULE_MAX_ORDER_AMOUNT` | `1000000` (기본값) | 낮게 설정 시 price_too_high 발생 |
| `RULE_NEW_ENTRY_WEIGHT` | `0.05` | 변경 없으면 종목당 배분 10만원 |
| `RULE_MIN_CASH_WEIGHT` | `0.20` | 변경 없으면 가용매수력 160만원 |

---

### 2. order_preview 실제 값 확인

```bash
python3 - <<'EOF'
import json
d = json.load(open("outputs/rule_order_preview.json", encoding="utf-8-sig"))
for i in d.get("items", []):
    if i.get("side") == "BUY":
        print(
            i.get("code"), i.get("name"),
            "qty:", i.get("order_qty"),
            "amount:", i.get("order_amount"),
            "base_amount:", i.get("base_order_amount"),
            "block:", i.get("order_block_reason"),
        )
EOF
```

**확인 포인트:**
- `order_qty` 가 0이면 → preview 단계에서 이미 실패 (after-close 문제)
- `order_qty` 가 1 이상이면 → before-open 제출 단계 문제

---

### 3. portfolio_plan 실제 target_amount 확인

```bash
python3 - <<'EOF'
import json
d = json.load(open("outputs/rule_portfolio_plan.json", encoding="utf-8-sig"))
print("account_state:", d.get("account_state"))
for i in d.get("items", []):
    if i.get("portfolio_action") == "buy":
        print(
            i.get("code"), i.get("name"),
            "action:", i.get("portfolio_action"),
            "target_amount:", i.get("target_amount"),
            "expected_price:", i.get("expected_entry_price"),
        )
EOF
```

**확인 포인트:**
- `target_amount` 가 0이면 → portfolio_manager가 배분금액을 0으로 계산 (계좌 데이터 이상)
- `target_amount` 가 10만원이고 `expected_entry_price` 가 10만원 초과이면 → auto_adjust 미발동 의심

---

### 4. auto_adjust 발동 여부 직접 계산

위 2·3번 확인 후 아래 공식으로 수동 검증:

```
available_buying_power = cash - total_equity × min_cash_weight
                       = 200만 - 200만 × 0.20 = 160만원

base_order_amount = min(target_amount, available_buying_power)

required_amount = expected_entry_price × 1주

[auto_adjust 발동 조건]
  required_amount <= available_buying_power  AND
  required_amount <= RULE_MAX_ORDER_AMOUNT

→ 둘 다 True: order_amount = max(base_order_amount, required_amount) → qty = 1
→ 하나라도 False: price_too_high = True, order_amount = base_order_amount → qty = 0
```

---

### 5. 로그 확인 (있을 경우)

```bash
# after-close 로그
tail -100 logs/rule_after_close_scheduler.log | grep -E "order_qty|price_too_high|blocked"

# before-open 로그
tail -100 logs/rule_before_open_scheduler.log | grep -E "order_qty|blocked|submit"
```

---

## 예상 원인 요약

| 가능성 | 조건 | 확인 방법 |
|---|---|---|
| **A** `RULE_AUTO_ADJUST_MINIMUM_SHARES=0` | env 설정 오류 | 확인 #1 |
| **B** `target_amount=0` | 계좌 동기화 이상 | 확인 #3 |
| **C** `RULE_MAX_ORDER_AMOUNT` 낮게 설정 | 파일럿 설정 잔존 | 확인 #1 |

**결론 판단 기준:**
- 확인 #2에서 `order_qty=0` → after-close 단계 문제 → 확인 #1·#3으로 원인 확정
- 확인 #2에서 `order_qty≥1` → before-open 제출 단계 문제 (별도 분석 필요)
