# Lee Trader Daily Codex Work Log

- Date: 2026-04-30
- Operator: Lee
- Work context: local / preview verification
- Target prompts: 1, 2, 3 integration verification

## Goal

Verify that Prompt 1 to Prompt 3 changes conservatively block new BUY orders for both AI and RULE flows before any live submission is allowed.

## Safety State Used For Preview Verification

- `AUTO_TRADE_EXECUTE=0`
- `AUTO_TRADE_ALLOW_BUY=0`
- `RULE_ORDER_SUBMIT_ENABLED=0`
- `GLOBAL_KILL_SWITCH=1`

No live order submission was executed.

## Files Updated Today

- `python/submit_live_orders.py`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/01_DAILY_CODEX_WORK_LOG_TEMPLATE.md`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/02_CODEX_PROMPT_EXECUTION_BOARD.md`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/03_SERVER_RELEASE_CHECKLIST.md`

## Verification Commands

```powershell
.\.venv\Scripts\python.exe python\common_live_risk_guard.py --self-test
```

```powershell
$env:AUTO_TRADE_EXECUTE='0'
$env:AUTO_TRADE_ALLOW_BUY='0'
$env:RULE_ORDER_SUBMIT_ENABLED='0'
$env:GLOBAL_KILL_SWITCH='1'
.\.venv\Scripts\python.exe python\submit_live_orders.py --out-json outputs\order_requests_preview_gks1.json --out-md outputs\order_requests_preview_gks1.md
.\.venv\Scripts\python.exe python\rule_order_preview_builder.py --out-json outputs\rule_order_preview_gks1.json
```

```powershell
$env:AUTO_TRADE_EXECUTE='0'
$env:AUTO_TRADE_ALLOW_BUY='0'
$env:RULE_ORDER_SUBMIT_ENABLED='0'
$env:GLOBAL_KILL_SWITCH='0'
.\.venv\Scripts\python.exe python\submit_live_orders.py --out-json outputs\order_requests_preview_gks0.json --out-md outputs\order_requests_preview_gks0.md
.\.venv\Scripts\python.exe python\rule_order_preview_builder.py --out-json outputs\rule_order_preview_gks0.json
```

## Verification Results

- Prompt 1 self-test passed
  - `outputs/common_live_risk_guard_self_test.json`
  - `passed=true`
  - `scenario_count=6`
- Prompt 1 outputs confirmed
  - `outputs/common_live_risk_guard.json`
  - `outputs/common_live_risk_guard_report.md`
- AI preview fields confirmed
  - `entry_price_gate_status`
  - `entry_price_gate_reason`
- RULE preview fields confirmed
  - `common_risk_allowed`
  - `common_risk_block_reasons`
  - `common_risk_snapshot`
- RULE BUY with `GLOBAL_KILL_SWITCH=1`
  - blocked
  - `global_kill_switch_on` present
- RULE BUY with `GLOBAL_KILL_SWITCH=0`
  - `global_kill_switch_on` removed from common block reasons
  - BUY still blocked by other conservative conditions in current local artifacts
- AI real-data preview on 2026-04-30
  - no BUY candidates were present
  - only 4 SELL/TRIM preview rows were produced
- AI BUY mock verification
  - detected a missing common guard connection during integration verification
  - added BUY common guard to `submit_live_orders.py`
  - re-verified `GLOBAL_KILL_SWITCH=1` blocks AI BUY
  - with `GLOBAL_KILL_SWITCH=0`, the global kill reason disappears and other conservative common guard reasons remain based on current local data
- SELL / EXIT separation
  - AI SELL/TRIM preview rows remain present
  - RULE SELL mock remains outside BUY-only common risk blocking

## Outputs Generated / Checked

- `outputs/common_live_risk_guard.json`
- `outputs/common_live_risk_guard_report.md`
- `outputs/common_live_risk_guard_self_test.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_preview.md`
- `outputs/order_requests_preview_gks1.json`
- `outputs/order_requests_preview_gks1.md`
- `outputs/order_requests_preview_gks0.json`
- `outputs/order_requests_preview_gks0.md`
- `outputs/rule_order_preview.json`
- `outputs/rule_order_preview_gks1.json`
- `outputs/rule_order_preview_gks0.json`

## Risks / Notes

- Current local sync artifacts are stale enough to keep BUY blocked even when `GLOBAL_KILL_SWITCH=0`.
- That behavior is consistent with the conservative design.
- `submit_live_orders.py` preview triggers local/web display sync side effects after preview generation.
- No server deployment was performed.

## Next Step

Server-side pre-release verification against fresh holdings, fills, market status, and ranking payloads.

---

# Lee Trader Daily Codex Work Log

- Date: 2026-05-01
- Operator: Lee
- Work context: server pre-release verification / preview-only
- Target prompts: 1 to 8 integrated safe-state validation

## Goal

Verify that the server environment is healthy and that AI / RULE preview outputs remain conservatively blocked under safe stop controls before any live-order enablement is considered.

## Safety State Used For Server Verification

- `AUTO_TRADE_EXECUTE=0`
- `AUTO_TRADE_ALLOW_BUY=0`
- `RULE_ORDER_SUBMIT_ENABLED=0`
- `GLOBAL_KILL_SWITCH=1`
- `RULE_KILL_SWITCH=1`

No live order submission was executed.

## Files Updated Today

- `python/submit_live_orders.py`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/01_DAILY_CODEX_WORK_LOG_TEMPLATE.md`
- `doc/upgrade/lee_trader_codex_checklists_flat_20260430/03_SERVER_RELEASE_CHECKLIST.md`

## Verification Commands

```powershell
docker compose ps
```

```powershell
@'
import sys
sys.path.insert(0, r'd:\ai\lee_trader\python')
from kis_client import KISClient
from kis_live_account import inquire_balance, resolve_account_env
client = KISClient.from_env()
token = client.issue_access_token()
account = resolve_account_env()
out1, out2 = inquire_balance(client, account)
print({
    'token_ok': bool(token),
    'env_dv': account.env_dv,
    'account': f'{account.cano}-{account.acnt_prdt_cd}',
    'holdings_rows': int(len(out1)),
    'summary_rows': int(len(out2)),
})
'@ | .\.venv\Scripts\python.exe -
```

```powershell
.\.venv\Scripts\python.exe python\submit_live_orders.py --out-json outputs\order_requests_preview_gks1.json --out-md outputs\order_requests_preview_gks1.md
.\.venv\Scripts\python.exe python\rule_order_preview_builder.py --out-json outputs\rule_order_preview_gks1.json
.\.venv\Scripts\python.exe python\master_risk_manager.py
```

## Verification Results

- Docker containers up
  - `node-api`
  - `postgres`
  - `scheduler`
  - `scheduler-auto-buy`
  - `scheduler-live-account-sync`
  - `scheduler-rule-before-open`
  - `scheduler-rule-after-open`
- KIS API authentication succeeded
- live account balance query succeeded
  - account environment: `real`
  - holdings rows: `12`
  - balance summary rows: `1`
- freshness checks
  - `data/market_status.csv`: latest date `2026-04-30`
  - `data/ranking_final.csv`: latest date `2026-04-30`
  - `outputs/live_account_balance_summary.json`: generated `2026-04-30 20:33:15`
  - `outputs/live_order_fills.json`: generated `2026-04-30 18:00:17`
- AI safe-state preview
  - `outputs/order_requests_preview_gks1.json`
  - request count `6`
  - BUY `3`, SELL `3`
  - BUY blocked reasons:
    - `052020`: `entry_gap_down_blocked`
    - `214150`: `global_kill_switch_on;holdings_sync_stale;fills_sync_stale;market_defensive_mode;daily_buy_amount_limit_exceeded;weekly_buy_amount_limit_exceeded;weekly_loss_pct_unavailable`
    - `058470`: `entry_gap_up_hard_blocked`
- RULE safe-state preview
  - `outputs/rule_order_preview_gks1.json`
  - universe items `204`
  - BUY preview count `2`
  - order allowed count `0`
- master integration preview
  - `outputs/master_risk_summary.json`
  - `ai_buy_candidates=3`
  - `rule_buy_candidates=2`
  - `approved_count=0`
  - `blocked_count=5`
  - blocked reason counts:
    - `entry_price_gate_blocked=2`
    - `common_risk_blocked=3`
- ops payload
  - `outputs/auto_trading_ops_status.json` generated
  - payload present for UI / API consumption

## Outputs Generated / Checked

- `outputs/order_requests_preview_gks1.json`
- `outputs/order_requests_preview_gks1.md`
- `outputs/rule_order_preview_gks1.json`
- `outputs/master_approved_orders.json`
- `outputs/master_blocked_orders.json`
- `outputs/master_risk_summary.json`
- `outputs/master_risk_summary.md`
- `outputs/auto_trading_ops_status.json`

## Additional Refresh On 2026-05-01

- refreshed live sync artifacts
  - `python/sync_live_account_holdings.py`
  - `python/sync_live_order_fills.py`
  - `python/sync_auxiliary_payloads.py`
- refreshed outputs
  - `outputs/live_account_balance_summary.json` at `2026-05-01 04:45:15`
  - `outputs/live_order_fills.json` at `2026-05-01 04:45:16`
  - `outputs/order_requests_preview_gks1.json` at `2026-05-01 04:45:29`
  - `outputs/rule_order_preview_gks1.json` at `2026-05-01 04:45:25`
  - `outputs/master_risk_summary.json` at `2026-05-01 04:45:24`
  - `outputs/live_trade_review_report.json` at `2026-05-01 04:45:27`
- post-refresh findings
  - AI BUY stale sync reasons no longer appeared
  - AI BUY remaining blocks:
    - `entry_gap_down_blocked`
    - `entry_gap_up_hard_blocked`
    - `global_kill_switch_on`
    - `market_defensive_mode`
    - `daily_buy_amount_limit_exceeded`
    - `weekly_buy_amount_limit_exceeded`
    - `weekly_loss_pct_unavailable`
  - RULE BUY remaining blocks:
    - `global_kill_switch_on`
    - `market_status_missing`
    - `daily_buy_amount_limit_exceeded`
    - `weekly_buy_amount_limit_exceeded`
    - `weekly_loss_pct_unavailable`
  - master risk summary after refresh:
    - `ai_buy_candidates=0`
    - `rule_buy_candidates=2`
    - `approved_count=0`
    - `blocked_count=2`

## Final Guard Review On 2026-05-01

- added RULE live account env verification
  - `KIS_RULE_CANO` present
  - `KIS_RULE_ACNT_PRDT_CD` present
  - live-mode RULE preview account profile validated
- fixed guard date basis
  - BUY day / week amount checks now use execution date
  - market status freshness continues to use signal date
- added weekly loss derivation
  - source: `research.live_position_snapshot`
  - `week_start_total_assets=3144804.0`
  - `weekly_loss_pct=-0.018866994572634733`
- refined blocker state after re-run
  - `daily_buy_amount_limit_exceeded` cleared
  - `market_status_missing` cleared
  - `weekly_loss_pct_unavailable` cleared
  - remaining blockers:
    - AI: `entry_gap_down_blocked`, `entry_gap_up_hard_blocked`, `global_kill_switch_on`, `market_defensive_mode`, `weekly_buy_amount_limit_exceeded`
    - RULE live: `rule_live_disabled`, `rule_order_submit_disabled`, `kill_switch_on`, `global_kill_switch_on`, `weekly_buy_amount_limit_exceeded`
- policy conclusion
  - `2026-05-01` is Friday
  - current weekly BUY usage from `2026-04-27` to `2026-04-30` is `2,880,120`
  - current `GLOBAL_MAX_WEEKLY_BUY_AMOUNT=1,500,000`
  - therefore no live BUY open should be attempted in the current trading week without an explicit policy change

## Risks / Notes

- Server apply remains blocked by policy, not by runtime failure.
- Current safe-state preview intentionally blocks every BUY candidate.
- `GLOBAL_KILL_SWITCH=1` and `RULE_KILL_SWITCH=1` are still active.
- Freshness is acceptable for preview verification, but market-open deployment should re-check holdings / fills / market status on the same trading day.

## Next Step

If server deployment is still planned, re-run the same checks on the target trading morning and only then consider staged release of `GLOBAL_KILL_SWITCH`, `AUTO_TRADE_ALLOW_BUY`, `RULE_ORDER_SUBMIT_ENABLED`, and `AUTO_TRADE_EXECUTE`.
