# US Stock Micro Live

## 1. Phase 7-5 purpose

Phase 7-5 adds Micro Live reconciliation for broker order status, fills, positions, and optional cash/account checks.

Core rule:

`ORDER_FILLED` or a broker fill row alone does not finalize the internal position state.

Internal DB rows must be compared against broker-facing data first. Any mismatch is recorded for operator review. It is not auto-corrected by placing new orders.

## 2. Reconciliation concept

Phase 7-5 answers these questions:

- Does internal order status match broker order status?
- Do internal fill rows match broker fill rows?
- Does the internal expected position from fills match broker positions?
- Does internal cash, when available, stay within tolerance versus broker cash?
- If there is a mismatch, is it recorded instead of auto-fixed?

Phase 7-5 is a validation stage only.

It does not:

- create new orders
- submit orders
- auto-resubmit
- auto-adjust positions
- overwrite internal DB from broker state

## 3. Data sources

Internal reconciliation baseline:

- `live.us_stock_micro_order_request`
- `live.us_stock_micro_order_fill`

Broker-side adapters:

- `utils/us_broker_account_interface.py`
- `utils/us_mock_account_client.py`
- `utils/us_sandbox_account_client.py`
- `utils/us_live_account_client.py`
- `utils/us_order_client_interface.py`
- `utils/us_mock_order_client.py`
- `utils/us_sandbox_order_client.py`
- `utils/us_live_order_client.py`

Main reconciliation utility:

- `utils/us_micro_reconciliation.py`

## 4. Order-status comparison

Comparison fields:

- `micro_order_id`
- `broker_order_id`
- internal `request_status`
- broker raw status
- mapped broker status

Standard comparison:

- `ORDER_FILLED` vs `ORDER_FILLED`
- `ORDER_PARTIALLY_FILLED` vs `ORDER_PARTIALLY_FILLED`
- `ORDER_OPEN` vs broker open/pending/working
- `ORDER_CANCELED` vs broker canceled
- `ORDER_REJECTED` vs broker rejected

Typical mismatch reason codes:

- `order_status_mismatch`
- `broker_order_missing`
- `broker_status_unknown`
- `order_status_query_failed`

`ORDER_FILLED` mismatch is treated as `CRITICAL`.

## 5. Fill comparison

Comparison fields:

- `broker_fill_id`
- `filled_qty`
- `filled_price`
- `filled_amount_usd`
- `fill_time`

Tolerance:

- quantity: `US_MICRO_RECON_TOLERANCE_QTY`
- amount: `US_MICRO_RECON_TOLERANCE_AMOUNT_USD`

Typical reason codes:

- `fill_missing_internal`
- `fill_missing_broker`
- `fill_qty_mismatch`
- `fill_amount_mismatch`
- `fill_price_mismatch`

No fill mismatch triggers an automatic order.

## 6. Position comparison

Internal expected positions are reconstructed from fill rows only.

Position rule:

- BUY fill: `+filled_qty`
- SELL fill: `-filled_qty`
- symbol net quantity becomes expected internal quantity

Comparison fields:

- `symbol`
- `internal_qty`
- `broker_qty`
- `qty_diff`

Tolerance:

- `abs(qty_diff) <= US_MICRO_RECON_TOLERANCE_QTY`

Typical reason codes:

- `position_qty_mismatch`
- `position_missing_broker`
- `unexpected_broker_position`

Position mismatches are `CRITICAL` candidates and should be treated as a stop-and-review signal.

## 7. Cash and account comparison

Cash comparison is optional because early Micro Live may not yet have a complete internal cash ledger.

Comparison fields:

- `internal_cash_usd`
- `broker_cash_usd`
- `cash_diff_usd`

Tolerance:

- `abs(cash_diff_usd) <= US_MICRO_RECON_TOLERANCE_CASH_USD`

Typical reason codes:

- `cash_mismatch`
- `cash_query_failed`
- `account_snapshot_missing`
- `internal_cash_unavailable`

Live account query remains blocked by default.

## 8. Result tables

Phase 7-5 adds:

- `live.us_stock_micro_reconciliation_result`
- `live.us_stock_micro_reconciliation_event_log`

`live.us_stock_micro_reconciliation_result` stores:

- reconciliation run id and item id
- recon type
- symbol / order references
- internal vs broker quantity or amount values
- internal vs broker status
- recon status
- severity
- reason code / detail
- masked raw payload snapshots

Recon types:

- `ORDER_STATUS`
- `FILL`
- `POSITION`
- `CASH`
- `ACCOUNT_EQUITY`
- `SUMMARY`

Recon statuses:

- `MATCH`
- `MISMATCH`
- `MISSING_INTERNAL`
- `MISSING_BROKER`
- `UNKNOWN`
- `ERROR`

Severities:

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

`live.us_stock_micro_reconciliation_event_log` stores run-level events such as:

- `RECON_START`
- `RECON_COMPLETE`
- `RECON_ERROR`
- `ORDER_STATUS_CHECK`
- `FILL_CHECK`
- `POSITION_CHECK`
- `CASH_CHECK`
- `KILL_SWITCH_RECOMMENDED`
- `KILL_SWITCH_TRIGGERED`

## 9. Kill Switch integration

Phase 7-5 never auto-corrects by trading.

Instead:

- `CRITICAL` mismatches are saved in reconciliation results
- a `KILL_SWITCH_RECOMMENDED` event is logged
- an `ACCOUNT` kill switch may be activated only when explicitly requested

Kill-switch reason codes:

- `reconciliation_position_mismatch`
- `reconciliation_cash_mismatch`
- `reconciliation_fill_mismatch`
- `reconciliation_order_status_mismatch`

Default policy:

- recommendation only
- no auto activation without explicit trigger option
- no trade adjustment under any condition in this phase

## 10. Safety controls

Required defaults:

- `US_MICRO_RECON_ENABLED=false`
- `US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY=false`
- `US_MICRO_RECON_REAL_ORDER_BLOCKED=true`

Live account query requires all of:

- `US_MICRO_RECON_ENABLED=true`
- `US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY=true`
- `US_MICRO_RECON_REAL_ORDER_BLOCKED=true`
- explicit `--execution-mode LIVE`

Sensitive fields such as account keys and tokens must not be stored in logs or reconciliation payload columns.

## 11. Execution commands

Dry-run:

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --include-orders \
  --include-fills \
  --include-positions \
  --dry-run
```

Persist results:

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --include-orders \
  --include-fills \
  --include-positions
```

Markdown report:

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --format markdown
```

Output path:

- `output/us_stock_micro_live/recon_<ACCOUNT_ID>_<DATE>.md`

## 12. Report interpretation

Console and markdown reports summarize:

- match / mismatch counts
- missing internal / missing broker counts
- error / critical counts
- position mismatch rows
- kill-switch recommendation state

Interpretation:

- `MATCH`: internal and broker state are aligned within tolerance
- `WARNING`: review needed, but not necessarily a stop condition
- `ERROR`: broker query or reconciliation step failed
- `CRITICAL`: stop Micro Live review flow and investigate before any further progression

## 13. Phase 7-6 handoff

Phase 7-5 is the account / position / fill consistency verification stage.

Phase 7-6 will combine:

- order history
- fill history
- reconciliation results
- kill-switch events
- approval logs

into an operational report and incident-response workflow.

If reconciliation `CRITICAL` issues repeat, Micro Live should be stopped and the process should revert to the Phase 6 safety policy baseline.

## 14. Phase 7-6 operations report

Phase 7-6 adds a daily integrated operations report that combines:

- ranking candidates
- pre-trade block logs
- approvals
- micro orders
- fill rows
- reconciliation results
- kill-switch state
- daily risk usage
- action-required items

Main commands:

```bash
python scripts/report_us_micro_live_operations.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --format console
python scripts/report_us_micro_live_operations.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --format markdown
python scripts/run_us_micro_live_daily_check.py --trade-date 2026-05-16 --account-id US_LIVE_TEST --execution-mode MOCK --dry-run
```

Health states:

- `HEALTHY`
- `ATTENTION`
- `DEGRADED`
- `CRITICAL`

The report shows state and recommended action only.
It does not create or send any new order.

## 15. Phase 7 checklist

Phase 7 completion and Phase 8 gate criteria are documented in:

- `doc/modules/Lee_trader_ai/US_STOCK_MICRO_LIVE_RUNBOOK.md`
- `doc/modules/Lee_trader_ai/US_STOCK_LIVE_OPERATION_RUNBOOK.md`
