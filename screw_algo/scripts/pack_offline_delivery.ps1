# Pack screw_algo for offline delivery (excludes .env and runs/obb)
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File screw_algo\scripts\pack_offline_delivery.ps1

param(
    [string]$OutZip = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PkgRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ProjectRoot = (Resolve-Path (Join-Path $PkgRoot "..")).Path

if (-not (Test-Path (Join-Path $PkgRoot "pipeline.py"))) {
    throw "screw_algo/pipeline.py not found. Run from project root that contains screw_algo."
}

$DistDir = Join-Path $ProjectRoot "dist"
$Staging = Join-Path $DistDir ("screw_algo_staging_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if (-not $OutZip) {
    $OutZip = Join-Path $DistDir "ScrewInspect-screw_algo-offline.zip"
} else {
    $OutZip = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutZip)
    $parent = Split-Path -Parent $OutZip
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

New-Item -ItemType Directory -Path $Staging -Force | Out-Null
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

$DestPkg = Join-Path $Staging "screw_algo"
New-Item -ItemType Directory -Path $DestPkg -Force | Out-Null

robocopy $PkgRoot $DestPkg /E /XD "runs\obb" "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" /XF ".env" "*.pyc" /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

if (Test-Path $OutZip) {
    Remove-Item $OutZip -Force
}
Compress-Archive -Path $DestPkg -DestinationPath $OutZip -CompressionLevel Optimal

Remove-Item $Staging -Recurse -Force

$sizeMb = [math]::Round((Get-Item $OutZip).Length / 1MB, 2)
Write-Host ""
Write-Host ("Offline zip: " + $OutZip + " (" + $sizeMb + " MB)")
Write-Host "Excluded: .env, runs/obb/, __pycache__"
Write-Host "See screw_algo/DELIVERY.md"
