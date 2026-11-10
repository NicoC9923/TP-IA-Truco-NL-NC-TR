# Waits for run 2 to finish successfully, then runs the pre-registered acceptance test once (RULE.txt).
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$done = 'training_output/phase2-run2/done.json'
while (-not (Test-Path $done)) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 5
$d = Get-Content $done -Raw | ConvertFrom-Json
if ($d.train -ne 0) { Write-Host "Training did not finish cleanly; acceptance NOT run."; exit 1 }
Write-Host "=== Acceptance test $(Get-Date -Format s) ==="
& python -u proyecto/phase2_acceptance.py --candidate champions/phase2-run2/reigning_champion.pt --baseline champions/phase2/reigning_champion.pt --run-dir training_output/phase2-run2 --held-out-manifest experiments/phase2-run2/held-out.json --direct-seed 60001 --held-out-seed 60002 --pairs 1000 --workers 12 --output experiments/phase2-run2/acceptance
Write-Host "=== Acceptance exit $LASTEXITCODE $(Get-Date -Format s) ==="
