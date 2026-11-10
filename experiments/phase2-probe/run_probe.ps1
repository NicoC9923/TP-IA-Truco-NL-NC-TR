$ErrorActionPreference = 'Stop'
Set-Location 'E:\Tools\GitHub-Repositories\TP-IA-Truco-NL-NC-TR\.orca\worktrees\TP-IA-Truco-NL-NC-TR\p2-run2'
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProbePower {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][ProbePower]::SetThreadExecutionState([uint32]2147483649)
$runDir = 'experiments/phase2-probe'
$exitCode = 1
try {
    'python -u proyecto/phase2_probe.py' | Set-Content "$runDir/command.txt"
    python -u proyecto/phase2_probe.py 2>&1 | Tee-Object -FilePath "$runDir/output.txt"
    $exitCode = $LASTEXITCODE
} catch {
    $_ | Out-String | Add-Content "$runDir/output.txt"
    $exitCode = 1
} finally {
    [void][ProbePower]::SetThreadExecutionState([uint32]2147483648)
    @{ exit_code = $exitCode; completed_at = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content "$runDir/done.json"
    orca orchestration send --from orca_session_id:0fa2f15e-9563-49f8-8c91-2a652ca297c0 --to dispatch:ctx_a65fdae4a549 --type status --subject 'Probe pilot completed' --body "Pilot process exited with code $exitCode; inspect experiments/phase2-probe/done.json and report before coordinator review." --json | Set-Content "$runDir/notification.json"
}

