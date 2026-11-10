# Phase 2 run 3 (12 h cap, unattended). Settings: RULE.txt in this folder. Resume: -Resume <checkpoint path> [-NewMatchesPerGeneration N].
param([string]$Resume, [int]$NewMatchesPerGeneration = 0)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = 'training_output/phase2-run3'
if (-not $Resume -and ((Test-Path $root) -or (Test-Path 'champions/phase2-run3'))) { throw "Output already exists: $root or champions/phase2-run3" }
New-Item -ItemType Directory -Force $root | Out-Null
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Run3Awake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][Run3Awake]::SetThreadExecutionState([uint32]2147483649)
$codes = @{}
if (Test-Path "$root/done.json") { Remove-Item "$root/done.json" }
$m = '124560'
if ($NewMatchesPerGeneration -gt 0) { $m = "$NewMatchesPerGeneration" }
try {
    $argsTrain = @('proyecto/phase2_trainer.py','--output',$root,'--champions-dir','champions/phase2-run3',
        '--generations','1000','--matches-per-generation',$m,'--matches-per-round','240','--workers','12',
        '--seed','65001','--check-seed-base','66000','--target','mixed','--lr','0.0003','--minibatch','512',
        '--temperature','0.02','--profile-prior','5','--patience','40','--time-cap-seconds','43200',
        '--panel-path','experiments/phase2-run2/panel.json','--no-panel-rotation',
        '--held-out-manifest','experiments/phase2-run2/held-out.json',
        '--league-shares','experiments/phase2-run2/league-shares.json','--timing-log',
        '--panel-pairs','300','--head-to-head-pairs','300','--pool-recent','8','--pool-diverse','4',
        '--encoder-version','3','--warm-start','champions/phase2/reigning_champion.pt',
        '--target-blend','0.5','--equity-table','experiments/phase2-run3/equity-table.json','--check-selection','greedy')
    if ($Resume) { $argsTrain += @('--resume',$Resume) }
    if ($NewMatchesPerGeneration -gt 0) { $argsTrain += @('--change-matches-per-generation') }
    Write-Host "=== Phase 2 run 3 $(Get-Date -Format s) ==="
    & py -u @argsTrain
    $codes['train'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} finally {
    $codes['finished'] = (Get-Date -Format s)
    $codes | ConvertTo-Json | Set-Content -Encoding utf8 "$root/done.json"
    [void][Run3Awake]::SetThreadExecutionState([uint32]2147483648)
}
