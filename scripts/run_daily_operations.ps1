$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "logs\\scheduler"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir ("daily_operations_{0}.log" -f $timestamp)

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
    Write-Host ("[{0}] OK {1}" -f $endedAt.ToString("yyyy-MM-dd HH:mm:ss"), $Name)
}

Push-Location $repoRoot
try {
    Start-Transcript -Path $logPath -Force | Out-Null

<<<<<<< HEAD
    Invoke-Step -Name "manual_close_batch" -Action {
        python python/run_manual_close_batch.py
=======
    Invoke-Step -Name "docker compose build python-pipeline" -Action {
        docker compose build --no-cache --progress=plain python-pipeline
    }

    Invoke-Step -Name "docker compose up python-pipeline" -Action {
        docker compose up -d python-pipeline
    }

    Invoke-Step -Name "run_operational_refresh" -Action {
        python python/run_operational_refresh.py
    }

    Invoke-Step -Name "docker compose up node-api" -Action {
        docker compose up -d --build node-api
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
    }

    Write-Host ("[{0}] DONE daily operations completed" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
    Pop-Location
}
