$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "logs\scheduler\us"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir ("us_daily_{0}.log" -f $timestamp)

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$tradeDate = (Get-Date).ToString("yyyy-MM-dd")

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    $startedAt = Get-Date
    Write-Host ("[{0}] START {1}" -f $startedAt.ToString("yyyy-MM-dd HH:mm:ss"), $Name)
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw ("Step failed: {0} (exit={1})" -f $Name, $LASTEXITCODE)
    }
    $endedAt = Get-Date
    Write-Host ("[{0}] OK {1} elapsed={2:N1}s" -f $endedAt.ToString("yyyy-MM-dd HH:mm:ss"), $Name, ($endedAt - $startedAt).TotalSeconds)
}

Push-Location $repoRoot
try {
    Start-Transcript -Path $logPath -Force | Out-Null
    Write-Host ("[US_SCHEDULER] Started trade_date={0}" -f $tradeDate)

    Invoke-Step -Name "1_pipeline_incremental" -Action {
        & $python python/us/run_us_daily_pipeline.py --incremental
    }

    Write-Host ("[US_SCHEDULER] DONE all steps completed trade_date={0}" -f $tradeDate)
    exit 0
}
catch {
    Write-Host ("[US_SCHEDULER] FAILED {0}" -f $_)
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    Pop-Location
}
