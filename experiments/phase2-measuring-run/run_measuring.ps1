param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = if ($Smoke) { 'training_output/phase2_measuring_smoke_4' } else { 'training_output/phase2_measuring' }
if ((Test-Path $root) -and (-not $Smoke -or (Get-ChildItem $root -Force))) { throw "Output already exists: $root" }
New-Item -ItemType Directory -Force $root | Out-Null
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class MeasuringAwake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][MeasuringAwake]::SetThreadExecutionState([uint32]2147483649)
$codes = @{}
try {
    foreach ($s in @('A','B')) {
        $seed = if ($s -eq 'A') { '42001' } else { '42002' }
        $base = if ($s -eq 'A') { '43000' } else { '45000' }
        $argsTrain = @('proyecto/phase2_trainer.py','--output',"$root/$s",'--champions-dir',"$root/$s/champions",'--panel-path','experiments/phase2-selection-panel/panel-v0-measuring.json','--seed',$seed,'--check-seed-base',$base,'--workers','12','--target','mixed','--temperature','0.1','--matches-per-round','240','--matches-per-generation','240','--generations','1000','--time-cap-seconds','480')
        if ($s -eq 'B') { $argsTrain += @('--auxiliary','--card-prediction-situations','400') }
        if ($Smoke) {
            $seed = if ($s -eq 'A') { '24001' } else { '24002' }
            $base = if ($s -eq 'A') { '30000' } else { '32000' }
            $argsTrain = @('proyecto/phase2_trainer.py','--output',"$root/$s",'--champions-dir',"$root/$s/champions",'--panel-path','experiments/phase2-selection-panel/panel-v0-measuring.json','--seed',$seed,'--check-seed-base',$base,'--workers','1','--target','1','--temperature','0.1','--matches-per-round','4','--matches-per-generation','4','--generations','3','--panel-pairs','2','--head-to-head-pairs','2','--time-cap-seconds','100')
            if ($s -eq 'B') { $argsTrain += @('--auxiliary','--card-prediction-situations','4') }
        }
        Write-Host "=== Segment $s ==="
        & python -u @argsTrain
        $codes[$s] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "Segment $s failed" }
    }
    foreach ($s in @('A','B')) {
        $argsBench = @('proyecto/bench_phase2.py','bench','--output',"$root/bench_$s.json",'--workers','12','--matches','240')
        if ($Smoke) { $argsBench = @('proyecto/bench_phase2.py','bench','--output',"$root/bench_$s.json",'--workers','2','--matches','4','--pairs','2','--direct-pairs','2') }
        $benchSeed = if ($s -eq 'A') { '42003' } else { '42004' }
        $benchBase = if ($s -eq 'A') { '47000' } else { '48000' }
        if ($Smoke) {
            $benchSeed = if ($s -eq 'A') { '24003' } else { '24004' }
            $benchBase = if ($s -eq 'A') { '34000' } else { '35000' }
        }
        $argsBench += @('--seed',$benchSeed,'--check-seed-base',$benchBase)
        if ($s -eq 'B') { $argsBench += '--auxiliary' }
        Write-Host "=== Benchmark $s ==="
        & python -u @argsBench
        $codes["bench_$s"] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "Benchmark $s failed" }
    }
    $argsAnalysis = @('proyecto/measuring_analysis.py','--directory',$root)
    if ($Smoke) { $argsAnalysis += '--smoke' }
    & python -u @argsAnalysis
    $codes['analysis'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
} finally {
    $codes | ConvertTo-Json | Set-Content -Encoding utf8 "$root/done.json"
    [void][MeasuringAwake]::SetThreadExecutionState([uint32]2147483648)
}





