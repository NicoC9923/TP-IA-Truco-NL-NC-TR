# Phase 2 run 1: one pre-registered continuation, after the primary result is saved.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
if ($DryRun) {
    $base = Join-Path ([IO.Path]::GetTempPath()) ('phase2-extension-dry-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path (Join-Path $base 'training_output/phase2') -Force | Out-Null
    $root = Join-Path $base 'training_output/phase2'
    '{"train":0}' | Set-Content -Encoding utf8 (Join-Path $root 'done.json')
    '{"phase_ended":true,"stop_reason":"time_cap","check":null,"promoted":false}' | Set-Content -Encoding utf8 (Join-Path $root 'generations.jsonl')
    New-Item -ItemType File -Path (Join-Path $base 'training_output/EXTENSION_READY') | Out-Null
} else {
    Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
    $base = (Get-Location).Path
    $root = Join-Path $base 'training_output/phase2'
}
$archive = Join-Path $base 'training_output/phase2_8h'
$marker = Join-Path $base 'training_output/EXTENSION_READY'
$done = Join-Path $root 'extension_done.json'
$skipped = Join-Path $root 'extension_skipped.json'
function Write-NewJson($path, $value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes(($value | ConvertTo-Json -Depth 12))
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush() } finally { $stream.Dispose() }
}
function Get-SkipReason {
    if ((Get-Content -Raw (Join-Path $root 'done.json') | ConvertFrom-Json).train -ne 0) { return 'train exit code is not zero' }
    $lines = @(Get-Content (Join-Path $root 'generations.jsonl') | Where-Object { $_.Trim() })
    if ($lines.Count -eq 0) { return 'generation log is empty' }
    $rows = @($lines | ForEach-Object { $_ | ConvertFrom-Json })
    $last = $rows[-1]
    if (-not $last.phase_ended -or $last.stop_reason -ne 'time_cap') { return 'run did not end on time_cap' }
    $stale = 0
    foreach ($row in $rows) {
        if ($null -ne $row.check) { if ($row.promoted) { $stale = 0 } else { $stale++ } }
    }
    if ($stale -ge 40) { return 'stale count is at least 40' }
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { return 'EXTENSION_READY is absent' }
    if (Test-Path -LiteralPath $archive) { return 'primary archive already exists' }
    return $null
}
function Get-TerminalCheckpoint {
    $lines = @(Get-Content (Join-Path $root 'generations.jsonl') | Where-Object { $_.Trim() })
    if ($lines.Count -eq 0) { throw 'generation log is empty' }
    $last = $lines[-1] | ConvertFrom-Json
    $generation = [long]$last.generation
    $matches = [long]$last.matches
    if ($generation -lt 1 -or $matches -lt 1 -or $matches -gt 57120 -or ($matches % 240) -ne 0) {
        throw 'invalid terminal generation or match count'
    }
    $expectedRound = ($generation - 1) * (57120 / 240) + ($matches / 240)
    $candidates = @(Get-ChildItem -LiteralPath $root -Filter 'checkpoint_*.pt' -File | Where-Object { $_.Name -match '^checkpoint_(\d+)\.pt$' } | ForEach-Object {
        [pscustomobject]@{ File = $_; Round = [long]($_.BaseName.Substring(11)) }
    } | Sort-Object Round -Descending)
    if ($candidates.Count -eq 0) { throw 'checkpoint not found' }
    if ($candidates[0].Round -ne $expectedRound) { throw 'latest checkpoint does not match terminal log round' }
    return $candidates[0].File
}
function Get-ByteTotal($path) {
    $sum = (Get-ChildItem -LiteralPath $path -File -Recurse | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return [long]0 }; return [long]$sum
}
if (Test-Path -LiteralPath $done) { Write-Host 'Extension already completed'; exit 0 }
if (Test-Path -LiteralPath $skipped) { Write-Host 'Extension already skipped'; exit 0 }
if (-not $DryRun) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ExtensionAwake {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
    [void][ExtensionAwake]::SetThreadExecutionState([uint32]2147483649)
}
try {
    $minutes = 0
    while (-not (Test-Path -LiteralPath (Join-Path $root 'done.json') -PathType Leaf)) {
        if (($minutes % 10) -eq 0) { Write-Host "Waiting for primary run: $(Get-Date -Format s)" }
        Start-Sleep -Seconds 60
        $minutes++
    }
    try { $reason = Get-SkipReason } catch { $reason = "condition check failed: $($_.Exception.Message)" }
    if ($DryRun) {
        if ($reason) { throw "Dry run unexpectedly skipped: $reason" }
        $cases = @('exit', 'reason', 'stale', 'marker', 'archive')
        foreach ($case in $cases) {
            $caseBase = Join-Path $base $case
            $root = Join-Path $caseBase 'training_output/phase2'
            $marker = Join-Path $caseBase 'training_output/EXTENSION_READY'
            $archive = Join-Path $caseBase 'training_output/phase2_8h'
            New-Item -ItemType Directory -Path $root -Force | Out-Null
            $code = if ($case -eq 'exit') { 1 } else { 0 }
            @{ train = $code } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $root 'done.json')
            $rows = @()
            if ($case -eq 'stale') { for ($i = 0; $i -lt 40; $i++) { $rows += '{"check":{},"promoted":false}' } }
            $stop = if ($case -eq 'reason') { 'patience' } else { 'time_cap' }
            $rows += "{`"phase_ended`":true,`"stop_reason`":`"$stop`",`"check`":null,`"promoted`":false}"
            $rows | Set-Content -Encoding utf8 (Join-Path $root 'generations.jsonl')
            if ($case -ne 'marker') { New-Item -ItemType File -Path $marker | Out-Null }
            if ($case -eq 'archive') { New-Item -ItemType Directory -Path $archive | Out-Null }
            $actual = Get-SkipReason
            if (-not $actual) { throw "Dry run failed $case condition" }
            Write-Host "Dry run: $case skipped: $actual"
        }
        $root = Join-Path $base 'checkpoint-case'
        New-Item -ItemType Directory -Path $root | Out-Null
        '{"generation":2,"matches":480}' | Set-Content -Encoding utf8 (Join-Path $root 'generations.jsonl')
        $older = New-Item -ItemType File -Path (Join-Path $root 'checkpoint_000239.pt')
        $newer = New-Item -ItemType File -Path (Join-Path $root 'checkpoint_000240.pt')
        $older.LastWriteTimeUtc = [datetime]::UtcNow.AddHours(1)
        $chosen = Get-TerminalCheckpoint
        if ($chosen.Name -ne $newer.Name) { throw 'Dry run chose checkpoint by timestamp' }
        $root = Join-Path $base 'checkpoint-mismatch'
        New-Item -ItemType Directory -Path $root | Out-Null
        New-Item -ItemType File -Path (Join-Path $root 'checkpoint_000240.pt') | Out-Null
        '{"generation":2,"matches":720}' | Set-Content -Encoding utf8 (Join-Path $root 'generations.jsonl')
        $rejected = $false
        try { [void](Get-TerminalCheckpoint) } catch { $rejected = $true }
        if (-not $rejected) { throw 'Dry run accepted mismatched checkpoint' }
        Write-Host 'Dry run: numeric checkpoint selected and mismatch rejected.'
        Write-Host 'Dry run: eligible conditions passed.'
        exit 0
    }
    if ($reason) { Write-NewJson $skipped @{ reason = $reason; timestamp = (Get-Date -Format o) }; exit 0 }
    try {
        $champions = Join-Path $base 'champions/phase2'
        if (-not (Test-Path -LiteralPath $champions -PathType Container)) { throw 'champion directory is absent' }
        $primaryBytes = Get-ByteTotal $root
        $championBytes = Get-ByteTotal $champions
        New-Item -ItemType Directory -Path $archive -ErrorAction Stop | Out-Null
        Get-ChildItem -LiteralPath $root -Force | Copy-Item -Destination $archive -Recurse
        $championArchive = Join-Path $archive 'champions_phase2'
        if (Test-Path -LiteralPath $championArchive) { throw 'champion archive destination already exists' }
        Copy-Item -LiteralPath $champions -Destination $championArchive -Recurse
        $copiedChampions = Get-ByteTotal (Join-Path $archive 'champions_phase2')
        if ($copiedChampions -ne $championBytes -or ((Get-ByteTotal $archive) - $copiedChampions) -ne $primaryBytes) { throw 'archive byte totals do not match' }
        $ckpt = Get-TerminalCheckpoint
    } catch {
        Write-NewJson $skipped @{ reason = $_.Exception.Message; timestamp = (Get-Date -Format o) }
        exit 0
    }
    $start = Get-Date -Format o
    $exitCode = -1
    try {
        & python -u proyecto/phase2_trainer.py --output training_output/phase2 --seed 54001 --check-seed-base 55000 --workers 12 --target mixed --temperature 0.05 --profile-prior 5 --matches-per-round 240 --matches-per-generation 57120 --generations 1000 --patience 40 --time-cap-seconds 28800 --resume $ckpt.FullName --extend-time-cap 43200
        $exitCode = $LASTEXITCODE
    } finally {
        Write-NewJson $done @{ exit_code = $exitCode; start = $start; end = (Get-Date -Format o); checkpoint = $ckpt.FullName }
    }
    exit $exitCode
} finally {
    if (-not $DryRun) { [void][ExtensionAwake]::SetThreadExecutionState([uint32]2147483648) }
}
