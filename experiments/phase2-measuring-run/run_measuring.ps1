param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$root = if ($Smoke) { 'training_output/phase2_measuring_smoke_off_20261108' } else { 'training_output/phase2_measuring' }
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
    foreach ($s in @('A')) {
        $seed = '42001'
        $base = '43000'
        $argsTrain = @('proyecto/phase2_trainer.py','--output',"$root/$s",'--champions-dir',"$root/$s/champions",'--panel-path','experiments/phase2-selection-panel/panel-v0-measuring.json','--seed',$seed,'--check-seed-base',$base,'--workers','12','--target','mixed','--temperature','0.1','--matches-per-round','240','--matches-per-generation','240','--generations','1000','--time-cap-seconds','480')
        if ($Smoke) {
            $seed = '24001'
            $base = '30000'
            $argsTrain = @('proyecto/phase2_trainer.py','--output',"$root/$s",'--champions-dir',"$root/$s/champions",'--panel-path','experiments/phase2-selection-panel/panel-v0-measuring.json','--seed',$seed,'--check-seed-base',$base,'--workers','1','--target','1','--temperature','0.1','--matches-per-round','4','--matches-per-generation','4','--generations','3','--panel-pairs','2','--head-to-head-pairs','2','--time-cap-seconds','100')
        }
        Write-Host "=== Segment $s ==="
        & python -u @argsTrain
        $codes[$s] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "Segment $s failed" }
    }
    foreach ($s in @('A')) {
        $argsBench = @('proyecto/bench_phase2.py','bench','--output',"$root/bench_$s.json",'--workers','12','--matches','240')
        if ($Smoke) { $argsBench = @('proyecto/bench_phase2.py','bench','--output',"$root/bench_$s.json",'--workers','2','--matches','4','--pairs','2','--direct-pairs','2') }
        $benchSeed = '42003'
        $benchBase = '47000'
        if ($Smoke) {
            $benchSeed = '24003'
            $benchBase = '34000'
        }
        $argsBench += @('--seed',$benchSeed,'--check-seed-base',$benchBase)
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





