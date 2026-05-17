param(
    [string]$TaskName = "LeeTraderUSDailyPipeline",
    [string]$StartTime = "06:30"
)

$ErrorActionPreference = "Stop"

$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "run_us_daily_pipeline.ps1")).Path
$workingRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$powerShellExe = (Get-Command powershell.exe).Source

$action = New-ScheduledTaskAction `
    -Execute $powerShellExe `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $scriptPath) `
    -WorkingDirectory $workingRoot

$trigger = New-ScheduledTaskTrigger -Daily -At $StartTime

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "US stock daily pipeline: price incremental + relative strength + rule scores. Runs at 06:30 KST (= 17:30 ET) after US market close."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

Write-Host ("Registered scheduled task '{0}' at {1} KST (= {2} ET)." -f $TaskName, $StartTime, "17:30")
Write-Host ("Script: {0}" -f $scriptPath)
Write-Host ("Log dir: {0}" -f (Join-Path $workingRoot "logs\scheduler\us"))
Write-Host ""
Write-Host "To run manually now:"
Write-Host ("  powershell -ExecutionPolicy Bypass -File `"{0}`"" -f $scriptPath)
