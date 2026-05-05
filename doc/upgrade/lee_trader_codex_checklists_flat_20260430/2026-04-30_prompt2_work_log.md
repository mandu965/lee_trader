# Prompt 2 Work Log

- Date: 2026-04-30
- Scope: Prompt 2 only
- Target: `python/submit_live_orders.py`
- Status: LOCAL_TESTED

## Summary

Added an AI BUY entry price gate to preview generation and to the submit path re-check.
The gate compares previous close vs live price and blocks BUY when the configured gap threshold is exceeded or when live price is unavailable.
Existing order submission safety controls were left unchanged.

## Modified Files

- `python/submit_live_orders.py`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/02_CODEX_PROMPT_EXECUTION_BOARD.md`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/2026-04-30_prompt2_work_log.md`

## Added Environment Variables

- `ENTRY_GAP_BLOCK_UP_PCT` default `0.03`
- `ENTRY_GAP_HARD_BLOCK_UP_PCT` default `0.05`
- `ENTRY_GAP_BLOCK_DOWN_PCT` default `-0.04`
- `ENTRY_GAP_BLOCK_ON_LIVE_PRICE_MISSING` default `1`

## Added / Updated Preview Fields

- `previous_close`
- `live_price`
- `live_price_source`
- `entry_price_gap_pct`
- `entry_price_gate_status`
- `entry_price_gate_reason`

## Implementation Notes

- Reused the KIS current quote endpoint used elsewhere in the project.
- Preview BUY flow now fetches live price, evaluates the entry gate, and stores the result in preview payload fields.
- BUY remains preview-only when live price is unavailable or when gap thresholds are violated.
- Execution path re-fetches live price immediately before submit and re-applies the same gate.
- `AUTO_TRADE_EXECUTE`, `AUTO_TRADE_ALLOW_BUY`, and BUY approval checks were not weakened or bypassed.
- Markdown preview report was extended with entry gate columns.

## Local Test Commands

```powershell
.\.venv\Scripts\python.exe -m py_compile python\submit_live_orders.py
```

```powershell
@'
import sys
sys.path.insert(0, r'd:\ai\lee_trader\python')
import submit_live_orders as s
cases = [
    ('missing', 10000, None),
    ('up_soft', 10000, 10350),
    ('up_hard', 10000, 10500),
    ('down_block', 10000, 9600),
    ('ok', 10000, 10100),
]
for name, prev_close, live_price in cases:
    print(name, s._evaluate_entry_price_gate(previous_close=prev_close, live_price=live_price))
'@ | .\.venv\Scripts\python.exe -
```

```powershell
@'
import sys
from types import SimpleNamespace
import pandas as pd
sys.path.insert(0, r'd:\ai\lee_trader\python')
import submit_live_orders as s

class DummyClient:
    def issue_access_token(self):
        return None

s.KISClient.from_env = staticmethod(lambda: DummyClient())
s.resolve_account_env = lambda: SimpleNamespace(env_dv='paper')
s.inquire_balance = lambda client, account: (pd.DataFrame(), pd.DataFrame([{'dnca_tot_amt': 1000000, 'tot_evlu_amt': 5000000}]))
s.summarize_cash = lambda df: {'dnca_tot_amt': 1000000, 'tot_evlu_amt': 5000000}
s.inquire_psbl_order = lambda client, account, pdno, ord_unpr, ord_dvsn: pd.DataFrame([{'nrcvb_buy_qty': 100, 'max_buy_qty': 100}])
s.compute_market_order_preview_qty = lambda **kwargs: 10
ranking = pd.DataFrame([{'code':'005930','name':'Test','close':10000,'buy_rank':1}])
holdings = pd.DataFrame(columns=['code','qty','eval_amount','weight'])
intents_payload = {'asof_date':'2026-04-30','gate_status':'ok','intents':[{'intent_id':'i1','code':'005930','intent_type':'BUY','executable':True,'target_weight':0.1,'priority':1,'reason':'test'}]}
for snapshot in [
    {'live_price': None, 'previous_close': 10000, 'source': 'unavailable'},
    {'live_price': 10350, 'previous_close': 10000, 'source': 'kis_quote'},
    {'live_price': 10500, 'previous_close': 10000, 'source': 'kis_quote'},
    {'live_price': 9600, 'previous_close': 10000, 'source': 'kis_quote'},
    {'live_price': 10100, 'previous_close': 10000, 'source': 'kis_quote'},
]:
    s._fetch_live_price_snapshot = lambda client, code, snapshot=snapshot: {'live_price': snapshot['live_price'], 'previous_close': snapshot['previous_close'], 'live_price_source': snapshot['source']}
    payload = s.build_order_requests(intents_payload=intents_payload, holdings=holdings, ranking=ranking, ord_dvsn='01')
    print(payload['items'][0]['blocked_reason'], payload['items'][0]['entry_price_gate_reason'], payload['items'][0]['executable_now'])
'@ | .\.venv\Scripts\python.exe -
```

## Local Test Results

- `py_compile`: passed
- live price unavailable: blocked as `live_price_unavailable`
- previous close `10000`, live price `10350`: blocked as `entry_gap_up_blocked`
- previous close `10000`, live price `10500`: blocked as `entry_gap_up_hard_blocked`
- previous close `10000`, live price `9600`: blocked as `entry_gap_down_blocked`
- previous close `10000`, live price `10100`: allowed with `entry_gap_ok`
- execution re-check: preview-allowed BUY was skipped when the re-check live price became unavailable
- markdown preview render includes the new entry gate fields

## Diff Summary

- added entry gate env readers and KIS quote wrapper
- added shared entry gate evaluator
- stored entry gate fields in preview payload
- blocked BUY preview on missing / out-of-range live price
- re-checked entry gate immediately before actual BUY submit
- extended preview markdown columns without removing existing fields

## Remaining Risks

- live quote endpoint latency or broker-side failure will block BUY by design
- limit order mode still uses `reference_price`; this change adds gating, not repricing logic
- no real broker submit was executed in local verification

## Server Apply Notes

- verify KIS quote API access in the deploy environment before enabling production runs
- inspect downstream consumers of `order_requests_preview.md` for wider table width
- do not proceed to Prompt 3 wiring before this preview artifact is reviewed once on the target server
