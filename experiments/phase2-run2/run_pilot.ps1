# Phase 2 run 2 throughput pilot (user-approved 2026-11-09). Measurement only: its output is never used for selection.
# 2 generations of 28,560 matches (check after generation 2, six-member panel), seeds 61001 / 62000.
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = 'training_output/phase2-run2-pilot'
if (Test-Path $root) { throw "Output already exists: $root" }
New-Item -ItemType Directory -Force $root | Out-Null
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PilotAwake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][PilotAwake]::SetThreadExecutionState([uint32]2147483649)
$codes = @{}
try {
    $argsTrain = @('proyecto/phase2_trainer.py','--output',$root,'--champions-dir',"$root/champions",
        '--generations','2','--matches-per-generation','28560','--matches-per-round','240','--workers','12',
        '--seed','61001','--check-seed-base','62000','--target','mixed','--lr','0.0003','--minibatch','512',
        '--temperature','0.05','--profile-prior','5','--patience','40','--time-cap-seconds','43200',
        '--panel-path','experiments/phase2-run2/panel.json','--no-panel-rotation',
        '--held-out-manifest','experiments/phase2-run2/held-out.json',
        '--league-shares','experiments/phase2-run2/league-shares.json','--timing-log')
    Write-Host "=== Phase 2 run 2 pilot $(Get-Date -Format s) ==="
    & python -u @argsTrain
    $codes['train'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Pilot failed' }
} finally {
    $codes['finished'] = (Get-Date -Format s)
    $codes | ConvertTo-Json | Set-Content -Encoding utf8 "$root/done.json"
    [void][PilotAwake]::SetThreadExecutionState([uint32]2147483648)
}
