# Phase 2 run 2 (12 h cap, unattended). Settings: RULE.txt in this folder. Resume: -Resume <checkpoint path>.
param([string]$Resume)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = 'training_output/phase2-run2'
if (-not $Resume -and ((Test-Path $root) -or (Test-Path 'champions/phase2-run2'))) { throw "Output already exists: $root or champions/phase2-run2" }
New-Item -ItemType Directory -Force $root | Out-Null
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Run2Awake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][Run2Awake]::SetThreadExecutionState([uint32]2147483649)
$codes = @{}
if (Test-Path "$root/done.json") { Remove-Item "$root/done.json" }
try {
    $argsTrain = @('proyecto/phase2_trainer.py','--output',$root,'--champions-dir','champions/phase2-run2',
        '--generations','1000','--matches-per-generation','57120','--matches-per-round','240','--workers','12',
        '--seed','57001','--check-seed-base','58000','--target','mixed','--lr','0.0003','--minibatch','512',
        '--temperature','0.05','--profile-prior','5','--patience','40','--time-cap-seconds','43200',
        '--panel-path','experiments/phase2-run2/panel.json','--no-panel-rotation',
        '--held-out-manifest','experiments/phase2-run2/held-out.json',
        '--league-shares','experiments/phase2-run2/league-shares.json','--timing-log')
    if ($Resume) { $argsTrain += @('--resume',$Resume) }
    Write-Host "=== Phase 2 run 2 $(Get-Date -Format s) ==="
    & python -u @argsTrain
    $codes['train'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} finally {
    $codes['finished'] = (Get-Date -Format s)
    $codes | ConvertTo-Json | Set-Content -Encoding utf8 "$root/done.json"
    [void][Run2Awake]::SetThreadExecutionState([uint32]2147483648)
}
