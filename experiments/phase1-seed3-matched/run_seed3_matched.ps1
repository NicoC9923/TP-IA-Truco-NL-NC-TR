# Seed 3 matched comparison: v1 then v2 (see training_output/seed3_matched/RULE.txt). Trainer direct, no publish.
$ErrorActionPreference = 'Continue'
Set-Location 'E:\Tools\GitHub-Repositories\TP-IA-Truco-NL-NC-TR\.orca\worktrees\TP-IA-Truco-NL-NC-TR\phase1-seeds'
$keep = "import sys,subprocess; sys.path.insert(0,'proyecto'); import run_phase1; run_phase1.keep_awake(); sys.exit(subprocess.call([sys.executable,'-u']+sys.argv[1:]))"
foreach ($enc in 'v1', 'v2') {
    $out = "training_output/phase1_encoder_${enc}_seed3"
    Write-Host "=== Seed 3 encoder $enc starting $(Get-Date -Format s) ==="
    $a = @('proyecto/phase1_trainer.py','--output',$out,'--seed','3','--workers','12','--population','64','--pairs','16','--generations','100','--patience','50','--validation-pairs','1000','--check-batch','200','--early-stop','reject-only','--promotion-threshold','0.55','--encoder',$enc,'--finalists','8','--finalist-pairs-multiplier','4','--output-init-sigma','0.3','--fitness-mode','classic','--points-weight','0.1','--style-strength','0.5','--styles-after-first-promotion')
    $ckpt = Get-ChildItem $out -Filter 'checkpoint_*.npz' -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($ckpt) { $a += @('--resume', $ckpt.FullName) }
    py -3 -c $keep @a 2>&1 | Tee-Object -Append "$out-train.log"
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        py -3 -c $keep proyecto/reference_evaluation.py --directory $out --promoted-only --pairs 1000 --workers 12 --encoder $enc 2>&1 | Tee-Object -Append "$out-train.log"
        $code = $LASTEXITCODE
    }
    "exit=$code finished=$(Get-Date -Format s)" | Out-File -Encoding ascii "training_output/seed3_${enc}_done.txt"
    Write-Host "=== Seed 3 encoder $enc finished with exit $code ==="
}
git status --short champions
