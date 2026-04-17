param(
    [string]$ProjectRoot = ".",
    [string]$Destination = "C:\ai\Lee_trader_home.zip",
    [switch]$IncludeData
)

# Helper: UTF-8 no BOM write
function Write-NoBom {
    param([string]$Path, [string]$Content)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

$root = (Resolve-Path $ProjectRoot).Path
Write-Host "Project root: $root"

# Build exclusion regex list
$exclude = @(
    "\\.git(\\|/|$)",
    "node(\\|/)node_modules(\\|/|$)",
    "(\\|/|^)logs(\\|/|$)",
    "__pycache__(\\|/|$)",
    "(\\|/|^)\.env$"  # .env always excluded
)

if (-not $IncludeData) {
    $exclude += "(\\|/|^)data(\\|/|$)"
    Write-Host "Excluding data/ directory (use -IncludeData to include)"
} else {
    Write-Host "Including data/ directory"
}

# Collect files with exclusions
$files = Get-ChildItem -Path $root -Recurse -File | Where-Object {
    $p = $_.FullName
    -not ($exclude | ForEach-Object { if ($p -match $_) { $true } })
}

if (-not $files -or $files.Count -eq 0) {
    Write-Error "No files to archive after applying exclusions."
    exit 1
}

# Ensure destination directory exists
$destDir = Split-Path -Parent $Destination
if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
}

# Create archive
if (Test-Path $Destination) {
    Remove-Item $Destination -Force
}
Compress-Archive -Path $files.FullName -DestinationPath $Destination -Force
Write-Host "Archive created: $Destination"

# Summary
Write-Host ""
Write-Host "Summary:"
Write-Host " - Files archived: $($files.Count)"
Write-Host " - Excluded patterns:"
$exclude | ForEach-Object { Write-Host "   * $_" }
Write-Host ""
Write-Host "NOTE:"
Write-Host " - .env is intentionally excluded. On the target machine, copy .env.example to .env and fill in real secrets."
Write-Host " - After copying the zip to home PC, extract and run: docker compose up --build -d"
