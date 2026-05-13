# US Stock Live Operation Runbook

## 1. Phase 7-5 scope

Phase 7-5 covers Micro Live reconciliation only.

It checks:

- order status alignment
- fill alignment
- position alignment
- optional cash/account alignment

It does not:

- create orders
- send orders
- auto-resubmit
- auto-adjust positions
- overwrite internal state from broker state

## 2. Safe defaults

Keep these defaults closed:

- `US_MICRO_RECON_ENABLED=false`
- `US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY=false`
- `US_MICRO_RECON_REAL_ORDER_BLOCKED=true`

Live account query must stay disabled unless an operator explicitly opens the gate for a controlled verification window.

## 3. Standard reconciliation flow

1. verify reconciliation env gates
2. generate `recon_run_id`
3. load internal micro-order and fill rows
4. query broker/mock/sandbox order status and fills
5. query broker/mock/sandbox positions
6. optionally query cash/account snapshot
7. compare internal vs broker state
8. store reconciliation results
9. store reconciliation event log
10. recommend or trigger kill switch if critical mismatches are found
11. print or save the reconciliation report

## 4. Commands

### Dry-run

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

### Persist run

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --include-orders \
  --include-fills \
  --include-positions
```

### Sandbox dry-run

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode SANDBOX \
  --include-orders \
  --include-fills \
  --include-positions \
  --dry-run
```

### Kill-switch trigger path

```bash
python scripts/reconcile_us_micro_live.py \
  --account-id US_LIVE_TEST \
  --recon-date 2026-05-16 \
  --execution-mode MOCK \
  --include-positions \
  --trigger-kill-on-critical
```

## 5. Result interpretation

- `MATCH`: state aligned
- `MISMATCH`: values disagree beyond tolerance
- `MISSING_INTERNAL`: broker state exists but internal counterpart does not
- `MISSING_BROKER`: internal state exists but broker counterpart does not
- `UNKNOWN`: comparison incomplete, usually due to unavailable internal reference
- `ERROR`: query or reconciliation failure

Severity guidance:

- `INFO`: aligned
- `WARNING`: review needed
- `ERROR`: reconcile step failed and needs investigation
- `CRITICAL`: Micro Live should be stopped until understood

## 6. Critical mismatch policy

Treat these as `CRITICAL`:

- broker position materially differs from expected internal position
- unexpected broker position exists
- order status says `ORDER_FILLED` on one side but not the other
- fill quantity mismatch
- cash difference exceeds tolerance

When this happens:

- save reconciliation result row
- save `KILL_SWITCH_RECOMMENDED` event
- only trigger kill switch if explicit option is provided

## 7. Kill Switch policy

Phase 7-5 supports:

- recommendation-only mode by default
- optional account-scoped kill-switch activation

It never supports:

- auto trading to repair mismatch
- broker-state overwrite of internal DB
- internal-state overwrite of broker position

## 8. Output and logs

Primary DB tables:

- `live.us_stock_micro_reconciliation_result`
- `live.us_stock_micro_reconciliation_event_log`

Markdown output:

- `output/us_stock_micro_live/recon_<ACCOUNT_ID>_<DATE>.md`

Sensitive credentials must not appear in:

- console output
- markdown reports
- stored raw payload columns

## 9. Phase 7-6 handoff

Phase 7-5 is the account / position / fill consistency verification stage.

Phase 7-6 will build the operating view that merges:

- orders
- fills
- reconciliation
- kill switch
- approvals

Repeated reconciliation `CRITICAL` outcomes mean Micro Live should be halted and the workflow should return to Phase 6 safety controls.

## 10. Phase 7-6 operations report

Phase 7-6 adds:

- `scripts/report_us_micro_live_operations.py`
- `scripts/run_us_micro_live_daily_check.py`

The operations report integrates:

- ranking
- pre-trade block logs
- approvals
- micro orders
- fills
- reconciliation
- kill switch
- daily risk usage
- action-required summary

Health levels:

- `HEALTHY`
- `ATTENTION`
- `DEGRADED`
- `CRITICAL`

Default behavior:

- report only
- no new order creation
- no order submission
- no auto re-order
- no auto position correction
- kill-switch activation remains opt-in

## 11. Phase 7 completion checklist

Refer to `US_STOCK_MICRO_LIVE_RUNBOOK.md` for the full Phase 7 completion checklist and the Phase 8 entry criteria.

Phase 8 is not full auto-trading.
It is a limited automation stage that only becomes eligible after Micro Live order, fill, reconciliation, and incident-response behavior have been sufficiently validated.
