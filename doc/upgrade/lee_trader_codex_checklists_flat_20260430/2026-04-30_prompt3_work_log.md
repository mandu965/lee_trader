# Prompt 3 Work Log

- Date: 2026-04-30
- Scope: Prompt 3 only
- Status: LOCAL_TESTED

## Summary

Connected RULE BUY order guarding to `common_live_risk_guard.evaluate_common_buy_guard()` without removing or weakening existing RULE block logic.
The common guard is now an additional BUY-only layer in both preview generation and submit-time validation.
SELL / EXIT paths keep their existing guard behavior and are not blocked by the BUY-only common risk layer.

## Modified Files

- `python/rule_account_guard.py`
- `python/rule_order_preview_builder.py`
- `python/rule_order_submitter.py`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/02_CODEX_PROMPT_EXECUTION_BOARD.md`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/2026-04-30_prompt3_work_log.md`

## Preview Field Changes

- `common_risk_allowed`
- `common_risk_block_reasons`
- `common_risk_snapshot`

## Implementation Notes

- Added `evaluate_rule_order_guard(order_context)` in `rule_account_guard.py`.
- Existing `assert_order_allowed(order_context)` remains available and now delegates to the new evaluator.
- Existing RULE guard reasons are still produced first and remain unchanged.
- Common live risk guard is evaluated only for `BUY`.
- `rule_order_preview_builder.py` now stores common guard result fields in `rule_order_preview.json`.
- `rule_order_submitter.py` now blocks BUY submission when preview already carries `common_risk_allowed=false`.
- `rule_order_submitter.py` also re-evaluates the common guard immediately before submit and blocks BUY when the re-check fails.

## Example Block Reasons

- Existing RULE reasons kept:
  - `paper_mode_no_order_submission`
  - `rule_live_disabled`
  - `rule_order_submit_disabled`
  - `kill_switch_on`
  - `buy_requires_strong_entry`
  - `market_defensive_mode`
  - `gap_risk_blocked`
  - `trading_value_failed`
  - `sector_limit_failed`
  - `cooldown_failed`
  - `cash_limit_failed`
  - `order_amount_exceeds_limit`
- Added BUY-only common reasons:
  - `global_kill_switch_on`
  - `holdings_sync_stale`
  - `fills_sync_stale`
  - `market_status_missing`
  - `daily_buy_amount_limit_exceeded`
  - `weekly_buy_amount_limit_exceeded`
  - `daily_loss_limit_reached`
  - `weekly_loss_limit_reached`

## Local Test Commands

```powershell
.\.venv\Scripts\python.exe -m py_compile python\rule_account_guard.py python\rule_order_preview_builder.py python\rule_order_submitter.py
```

```powershell
@'
import os
import sys
from types import SimpleNamespace
import pandas as pd
sys.path.insert(0, r'd:\ai\lee_trader\python')
import rule_account_guard as guard
import rule_order_preview_builder as preview_builder
import rule_order_submitter as submitter

orig_common = guard.evaluate_common_buy_guard
orig_resolve = submitter.resolve_rule_account_env
orig_kis = submitter.KISClient.from_env
orig_cash = submitter.order_cash

try:
    def fake_common_block(ctx):
        side = str(ctx.get('side') or '').upper()
        if side != 'BUY':
            return True, [], {'side': side, 'buy_allowed': True, 'block_reasons': [], 'bypass_reason': 'non_buy_side'}
        return False, ['global_kill_switch_on', 'daily_loss_limit_reached'], {'side': side, 'buy_allowed': False, 'block_reasons': ['global_kill_switch_on', 'daily_loss_limit_reached'], 'source': 'test'}

    def fake_common_allow(ctx):
        side = str(ctx.get('side') or '').upper()
        return True, [], {'side': side, 'buy_allowed': True, 'block_reasons': [], 'source': 'test'}

    plan = {
        'as_of_date': '2026-04-30',
        'account_state': {'total_equity': 5000000, 'cash': 2000000},
        'config': {'min_cash_weight': 0.2},
        'items': [
            {'code': '005930', 'name': 'BUYTEST', 'portfolio_action': 'buy', 'expected_entry_price': 10000, 'target_amount': 300000, 'current_amount': 0, 'signal_strength': 'strong_entry', 'market_defensive_mode': False, 'gap_risk_reason': 'none', 'trading_value_block_reason': 'none', 'sector_limit_pass': True, 'cooldown_pass': True, 'cash_limit_pass': True},
            {'code': '000660', 'name': 'SELLTEST', 'portfolio_action': 'exit', 'expected_entry_price': 90000, 'target_amount': 0, 'current_amount': 300000, 'signal_strength': 'strong_entry', 'market_defensive_mode': False, 'gap_risk_reason': 'none', 'trading_value_block_reason': 'none', 'sector_limit_pass': True, 'cooldown_pass': True, 'cash_limit_pass': True},
        ],
    }

    os.environ['RULE_KILL_SWITCH'] = '0'
    guard.evaluate_common_buy_guard = fake_common_block
    preview = preview_builder.build_rule_order_preview(plan, run_mode='paper')
    buy_item = next(row for row in preview['items'] if row['side'] == 'BUY')
    sell_item = next(row for row in preview['items'] if row['side'] == 'SELL')
    print('paper_mode', 'paper_mode_no_order_submission' in buy_item['order_block_reason'], buy_item['common_risk_allowed'], buy_item['common_risk_block_reasons'])
    print('sell_bypass', sell_item['common_risk_allowed'], sell_item['common_risk_block_reasons'])

    os.environ['RULE_KILL_SWITCH'] = '1'
    preview_kill = preview_builder.build_rule_order_preview(plan, run_mode='live')
    buy_kill = next(row for row in preview_kill['items'] if row['side'] == 'BUY')
    print('rule_kill', 'kill_switch_on' in buy_kill['order_block_reason'])
    os.environ['RULE_KILL_SWITCH'] = '0'

    guard.evaluate_common_buy_guard = fake_common_allow
    preview_allowed = preview_builder.build_rule_order_preview(plan, run_mode='live')

    guard.evaluate_common_buy_guard = fake_common_block
    class DummyClient:
        def issue_access_token(self):
            return None
    submitter.resolve_rule_account_env = lambda: SimpleNamespace()
    submitter.KISClient.from_env = staticmethod(lambda: DummyClient())
    submitter.order_cash = lambda *args, **kwargs: pd.DataFrame([{'ODNO': '123'}])
    market_snapshot = {'api_health_status': 'ok', 'api_failure_reason': None, 'snapshots': {'005930': {'market_data_available': True, 'open_price': 10000, 'actual_open_gap': 0.01}, '000660': {'market_data_available': True, 'open_price': 90000, 'actual_open_gap': 0.0}}}
    items, summary = submitter._submit_items(preview_allowed, market_snapshot)
    buy_submit = next(row for row in items if row['side'] == 'BUY')
    sell_submit = next(row for row in items if row['side'] == 'SELL')
    print('submit_recheck_buy', buy_submit['order_status'], buy_submit['common_risk_allowed'], buy_submit['common_risk_block_reasons'])
    print('submit_sell_not_removed', sell_submit['side'], sell_submit['common_risk_allowed'])
finally:
    guard.evaluate_common_buy_guard = orig_common
    submitter.resolve_rule_account_env = orig_resolve
    submitter.KISClient.from_env = orig_kis
    submitter.order_cash = orig_cash
    os.environ['RULE_KILL_SWITCH'] = '0'
'@ | .\.venv\Scripts\python.exe -
```

```powershell
@'
import os
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, r'd:\ai\lee_trader\python')
import rule_account_guard as guard

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    balance = root / 'balance.json'
    fills = root / 'fills.json'
    market = root / 'market_status.csv'
    holdings = root / 'holdings.csv'
    balance.write_text(json.dumps({'generated_at': '2026-04-30T09:00:00', 'summary_row': {'asst_icdc_erng_rt': 0.0}}), encoding='utf-8')
    fills.write_text(json.dumps({'generated_at': '2026-04-30T09:00:00', 'items': []}), encoding='utf-8')
    market.write_text('date,market_up\n2026-04-30,1\n', encoding='utf-8')
    holdings.write_text('code,qty\n005930,0\n', encoding='utf-8')
    ctx = {
        'account_id': 'RULE_ACCOUNT_01',
        'strategy_id': 'RULE_TREND_LIQUIDITY_V1',
        'engine_type': 'rule_based',
        'run_mode': 'live',
        'now': '2026-04-30T09:10:00',
        'as_of_date': '2026-04-30',
        'side': 'BUY',
        'code': '005930',
        'order_qty': 10,
        'order_amount': 300000,
        'reference_price': 10000,
        'signal_strength': 'strong_entry',
        'market_defensive_mode': False,
        'gap_risk_blocked': False,
        'trading_value_pass': True,
        'sector_limit_pass': True,
        'cooldown_pass': True,
        'cash_limit_pass': True,
        'holdings_csv': holdings,
        'balance_json': balance,
        'fills_json': fills,
        'market_status_csv': market,
        'weekly_loss_pct': 0.0,
    }
    os.environ['RULE_LIVE_ENABLED'] = '1'
    os.environ['RULE_ORDER_SUBMIT_ENABLED'] = '1'
    os.environ['GLOBAL_KILL_SWITCH'] = '1'
    os.environ['RULE_KILL_SWITCH'] = '0'
    print('global_only', guard.evaluate_rule_order_guard(ctx)[1])
    os.environ['GLOBAL_KILL_SWITCH'] = '0'
    os.environ['RULE_KILL_SWITCH'] = '1'
    print('rule_only', guard.evaluate_rule_order_guard(ctx)[1])
    os.environ['RULE_KILL_SWITCH'] = '0'
'@ | .\.venv\Scripts\python.exe -
```

## Local Test Results

- compile passed
- `GLOBAL_KILL_SWITCH=1` with valid mock data blocked RULE BUY and produced `global_kill_switch_on`
- `RULE_KILL_SWITCH=1` preserved existing `kill_switch_on`
- paper mode preview preserved `paper_mode_no_order_submission`
- common guard block was recorded in:
  - `common_risk_allowed=false`
  - `common_risk_block_reasons=[...]`
  - `common_risk_snapshot={...}`
- SELL / EXIT candidates kept `common_risk_allowed=true` and empty common block reasons
- submit-time re-check blocked BUY when common guard changed from allow at preview time to block at submit time

## Diff Summary

- added `evaluate_rule_order_guard()` with BUY-only common guard integration
- kept `assert_order_allowed()` as compatibility wrapper
- added common guard fields to RULE preview items
- blocked BUY submit when preview common risk is already false
- re-ran common guard immediately before BUY submit

## Remaining Risks

- common guard depends on live sync / market status artifacts and will conservatively block BUY when those artifacts are stale or missing
- submit-time SELL can still be blocked by existing RULE controls such as live-disable flags; this is expected and separate from common BUY risk
- no real broker submit was executed in local verification

## Server Apply Notes

- verify `market_status.csv`, holdings sync, and fills sync freshness on the target server before enabling RULE live BUY
- expect BUY volume to drop if common guard artifacts are stale; that is the intended conservative behavior
- review downstream consumers of `rule_order_preview.json` so the added `common_risk_*` fields are surfaced where operators need them
