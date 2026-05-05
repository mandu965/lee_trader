# RULE Live Transition: Operation Commands

## Preconditions
- Windows PowerShell
- Docker Desktop running
- project root: `d:\ai\lee_trader`
- `.env` remains safe by default

## 1. Paper after-close build
```powershell
$env:RULE_TRADING_RUN_MODE="paper"
$env:RULE_LIVE_ENABLED="0"
$env:RULE_ORDER_SUBMIT_ENABLED="0"
docker compose run --rm --no-deps scheduler-rule-after-close python python/run_rule_after_close_cycle.py
```

## 2. After-close local script alternative
```powershell
$env:RULE_TRADING_RUN_MODE="paper"
$env:RULE_LIVE_ENABLED="0"
$env:RULE_ORDER_SUBMIT_ENABLED="0"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_rule_after_close.ps1
```

## 3. Before-open paper execution
```powershell
$env:RULE_TRADING_RUN_MODE="paper"
$env:RULE_LIVE_ENABLED="0"
$env:RULE_ORDER_SUBMIT_ENABLED="0"
docker compose run --rm --no-deps scheduler-rule-before-open python python/run_rule_before_open_cycle.py
```

## 4. After-open cycle
- `paper` mode:
  - no meaningful after-open action is needed
- `pilot/live` mode:
```powershell
docker compose run --rm --no-deps scheduler-rule-after-open python python/run_rule_after_open_cycle.py
```

## 5. Pilot mode env setup
- Keep `.env` safe.
- For a temporary PowerShell session only:
```powershell
$env:RULE_TRADING_RUN_MODE="pilot"
$env:RULE_LIVE_ENABLED="1"
$env:RULE_ORDER_SUBMIT_ENABLED="1"
$env:RULE_MAX_ORDER_AMOUNT="10000"
$env:RULE_MAX_DAILY_ORDER_AMOUNT="30000"
$env:RULE_PILOT_MAX_ORDER_AMOUNT="10000"
$env:RULE_PILOT_MAX_ORDER_QTY="1"
```
- Optional rollback-safe stop:
```powershell
$env:RULE_KILL_SWITCH="1"
```

## 6. Pilot before-open submit
```powershell
docker compose run --rm --no-deps scheduler-rule-before-open python python/run_rule_before_open_cycle.py
```

## 7. Pilot 1-week test procedure
1. Day before market open:
   - run after-close in `pilot`
   - inspect `outputs/rule_portfolio_plan.json`
   - inspect `outputs/rule_order_preview.json`
2. Before market open:
   - confirm KST time is inside the configured window
   - confirm holiday file does not block the day
   - run before-open cycle once
3. After market open:
   - run after-open fill sync once
   - inspect `outputs/rule_execution_results.json`
   - inspect `outputs/rule_account_live_state.json`
4. Repeat for 5 trading days.
5. Do not raise limits during the same week unless there is a documented reason.

## 8. Live mode transition conditions
- all pilot days completed without unexpected abort
- fill sync reconciles submitted orders
- live account snapshot succeeds
- operator understands every blocked order reason
- amount and position limits are intentionally raised, not left at defaults by accident

## 9. Live mode switch
```powershell
$env:RULE_TRADING_RUN_MODE="live"
$env:RULE_LIVE_ENABLED="1"
$env:RULE_ORDER_SUBMIT_ENABLED="1"
```
- Then run:
```powershell
docker compose run --rm --no-deps scheduler-rule-before-open python python/run_rule_before_open_cycle.py
docker compose run --rm --no-deps scheduler-rule-after-open python python/run_rule_after_open_cycle.py
```

## 10. Revert to paper
```powershell
$env:RULE_TRADING_RUN_MODE="paper"
$env:RULE_LIVE_ENABLED="0"
$env:RULE_ORDER_SUBMIT_ENABLED="0"
$env:RULE_KILL_SWITCH="0"
```

## 11. Emergency stop and rollback
- Immediate stop:
```powershell
$env:RULE_KILL_SWITCH="1"
$env:RULE_ORDER_SUBMIT_ENABLED="0"
$env:RULE_LIVE_ENABLED="0"
```
- Then revert to paper:
```powershell
$env:RULE_TRADING_RUN_MODE="paper"
docker compose run --rm --no-deps scheduler-rule-before-open python python/run_rule_before_open_cycle.py
```

## 12. Docker scheduler services
- Start rule schedulers:
```powershell
docker compose up -d scheduler-rule-after-close
docker compose up -d scheduler-rule-before-open
docker compose up -d scheduler-rule-after-open
```
- Check logs:
```powershell
docker logs lee_trader_scheduler_rule_after_close --tail 200
docker logs lee_trader_scheduler_rule_before_open --tail 200
docker logs lee_trader_scheduler_rule_after_open --tail 200
```
- Check scheduler status files:
```powershell
Get-Content outputs/rule_after_close_scheduler_status.json
Get-Content outputs/rule_before_open_scheduler_status.json
Get-Content outputs/rule_after_open_scheduler_status.json
```
