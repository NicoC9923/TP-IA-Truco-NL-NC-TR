# Phase 2 real run 1 (#29). Settings: see RULE.txt in this folder.
param([string]$Resume)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = 'training_output/phase2'
if (-not $Resume -and (Test-Path $root)) { throw "Output already exists: $root" }
New-Item -ItemType Directory -Force $root | Out-Null
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Run1Awake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][Run1Awake]::SetThreadExecutionState([uint32]2147483649)
$codes = @{}
if (Test-Path "$root/done.json") { Remove-Item "$root/done.json" }
try {
    $argsTrain = @('proyecto/phase2_trainer.py','--output',$root,'--seed','54001','--check-seed-base','55000',
        '--workers','12','--target','mixed','--temperature','0.05','--profile-prior','5',
        '--matches-per-round','240','--matches-per-generation','57120','--generations','1000',
        '--patience','40','--time-cap-seconds','28800')
    if ($Resume) { $argsTrain += @('--resume',$Resume) }
    Write-Host "=== Phase 2 run 1 $(Get-Date -Format s) ==="
    & python -u @argsTrain
    $codes['train'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} finally {
    $codes['finished'] = (Get-Date -Format s)
    $codes | ConvertTo-Json | Set-Content -Encoding utf8 "$root/done.json"
    [void][Run1Awake]::SetThreadExecutionState([uint32]2147483648)
}
