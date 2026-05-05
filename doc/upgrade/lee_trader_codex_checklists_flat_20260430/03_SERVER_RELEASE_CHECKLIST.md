# Lee Trader Server Release Checklist

- Date: 2026-05-01
- Scope: Prompt 1 to Prompt 8 preview-only verification
- Server apply status: NOT APPLIED

## Current Recorded State

- `AUTO_TRADE_EXECUTE`: preview-only verification used
- `AUTO_TRADE_ALLOW_BUY`: preview-only verification used
- `GLOBAL_KILL_SWITCH`: tested with both `1` and `0`
- `RULE_KILL_SWITCH`: locally verified
- `RULE_ORDER_SUBMIT_ENABLED`: preview-only verification used
- `RULE_LIVE_ENABLED`: not changed by this verification

## Preview-Only Verification Completed

- [x] `common_live_risk_guard.py` self-test passed
- [x] `common_live_risk_guard.json` generated
- [x] `common_live_risk_guard_report.md` generated
- [x] `order_requests_preview.json` generated
- [x] `rule_order_preview.json` generated
- [x] `order_requests_preview_gks1.json` generated under safe stop state
- [x] `rule_order_preview_gks1.json` generated under safe stop state
- [x] `master_approved_orders.json` generated
- [x] `master_blocked_orders.json` generated
- [x] `master_risk_summary.json` generated
- [x] `master_risk_summary.md` generated
- [x] AI preview contains `entry_price_gate_status`
- [x] AI preview contains `entry_price_gate_reason`
- [x] RULE preview contains `common_risk_allowed`
- [x] RULE preview contains `common_risk_block_reasons`
- [x] RULE BUY blocked with `GLOBAL_KILL_SWITCH=1`
- [x] AI BUY mock blocked with `GLOBAL_KILL_SWITCH=1`
- [x] SELL / EXIT kept separate from BUY-only common guard
- [x] ops payload reflects stopped state with `GLOBAL_KILL_SWITCH=1`, `RULE_KILL_SWITCH=1`

## Not Yet Performed On Server

- [x] Docker container status check before market open
- [x] KIS API authentication check
- [x] account balance query check
- [x] `market_status` freshness check
- [x] `ranking/latest` freshness check
- [x] live holdings sync freshness check
- [x] live fills sync freshness check
- [x] explicit server-side kill switch review
- [ ] pilot/live submission check
- [ ] intraday monitoring after deploy

## Safe Pre-Release State

Use this state before any server-side validation:

```text
AUTO_TRADE_EXECUTE=0
AUTO_TRADE_ALLOW_BUY=0
RULE_ORDER_SUBMIT_ENABLED=0
GLOBAL_KILL_SWITCH=1
RULE_KILL_SWITCH=1
```

## Release Decision

- Result: hold server apply
- Reason:
  - local preview verification completed
  - server environment checks executed in safe state
  - docker services are up and KIS auth / balance query succeeded
  - current artifacts still show conservative BUY blocking, which is expected under the active stop controls
  - 2026-05-01 is Friday and current weekly BUY usage is `2,880,120`, above `GLOBAL_MAX_WEEKLY_BUY_AMOUNT=1,500,000`
  - because of that weekly cap, new BUY enablement on the current trading week is not an implementation issue and should not be forced open
  - AI live price lookup helper bug was fixed locally and preview regenerated
  - current safe-state preview still blocks all BUY candidates due intentional stop controls plus risk / entry gate conditions

## 2026-05-01 Server Verification Snapshot

- Docker containers confirmed up
  - `node-api`
  - `postgres`
  - `scheduler`
  - `scheduler-auto-buy`
  - `scheduler-live-account-sync`
  - `scheduler-rule-before-open`
  - `scheduler-rule-after-open`
- KIS verification
  - token issuance succeeded
  - live balance query succeeded
  - holdings rows: `12`
  - summary rows: `1`
- artifact freshness
  - `market_status.csv`: latest date `2026-04-30`
  - `ranking_final.csv`: latest date `2026-04-30`
  - `live_account_balance_summary.json`: `2026-05-01 04:45:15`
  - `live_order_fills.json`: `2026-05-01 04:45:16`
- safe-state preview summary
  - AI preview: `6` requests, BUY `3`, SELL `3`
  - RULE preview: BUY preview `2`, allowed `0`
  - master risk summary: approved `0`, blocked `5`
  - after fresh sync:
    - stale sync reasons cleared from AI BUY block reasons
    - safe-state aligned master risk summary remains approved `0`, blocked `5`
    - remaining BUY blocks are intentional stop / risk controls, not sync freshness failures
  - after date-basis correction:
    - `daily_buy_amount_limit_exceeded` cleared
    - `market_status_missing` cleared
    - `weekly_loss_pct_unavailable` cleared
    - remaining blockers are:
      - `global_kill_switch_on`
      - `weekly_buy_amount_limit_exceeded`
      - AI-only `entry_gap_*` and `market_defensive_mode`
      - RULE-only `rule_live_disabled`, `rule_order_submit_disabled`, `kill_switch_on`

## Next Release Gate

- Do not open live BUY on 2026-05-01 while weekly cap remains exceeded.
- Re-check on the next trading week after weekly BUY usage resets, or explicitly revise `GLOBAL_MAX_WEEKLY_BUY_AMOUNT` by policy decision.
- Even after weekly cap becomes available, release order should stay:
  1. review `GLOBAL_KILL_SWITCH`
  2. review `RULE_LIVE_ENABLED`
  3. review `RULE_ORDER_SUBMIT_ENABLED`
  4. review `AUTO_TRADE_ALLOW_BUY`
  5. review `AUTO_TRADE_EXECUTE`

## Emergency Stop Values

If any unexpected live-order risk is found, use:

```text
GLOBAL_KILL_SWITCH=1
AUTO_TRADE_EXECUTE=0
AUTO_TRADE_ALLOW_BUY=0
RULE_ORDER_SUBMIT_ENABLED=0
RULE_KILL_SWITCH=1
```
