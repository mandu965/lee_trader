$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

python python/rule_signal_builder.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/rule_backtest.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/rule_portfolio_manager.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/rule_order_preview_builder.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/rule_daily_report.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
