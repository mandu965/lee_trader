$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$runMode = $env:RULE_TRADING_RUN_MODE
if ([string]::IsNullOrWhiteSpace($runMode)) { $runMode = "paper" }
if ($runMode.ToLower() -eq "paper") {
  python python/rule_execution_simulator.py
} else {
  python python/rule_order_fill_sync.py
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
