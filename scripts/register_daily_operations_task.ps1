param(
    [string]$TaskName = "LeeTraderDailyOperations",
    [string]$StartTime = "17:00"
)

$ErrorActionPreference = "Stop"

$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "run_daily_operations.ps1")).Path
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
    -Description "Runs Lee_trader daily pipeline and operational refresh at 17:00."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

Write-Host ("Registered scheduled task '{0}' at {1}." -f $TaskName, $StartTime)
Write-Host ("Script: {0}" -f $scriptPath)
