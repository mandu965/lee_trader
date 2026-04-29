$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

python python/rule_execution_simulator.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
