# Phase 1 run 2 experiment

This learning-fixes experiment trained from scratch with seed 0 at code revision `eafb225`. It is a separate run from the shipped run 1, not a continuation of its training history. `run_config.json` records the seed and target; the settings below come from the pipeline log.

## Execution

The original pipeline log records this 30-generation training command (paths are as recorded):

```text
C:\Users\nicof\AppData\Local\Programs\Python\Python312\python.exe -u E:\Tools\GitHub-Repositories\TP-IA-Truco-NL-NC-TR\.orca\worktrees\TP-IA-Truco-NL-NC-TR\phase1-run2\proyecto\phase1_trainer.py --output training_output/phase1 --generations 30 --population 64 --pairs 16 --workers 12 --seed 0 --validation-pairs 1000 --check-batch 200 --promotion-threshold 0.55 --patience 50 --finalists 8 --finalist-pairs-multiplier 4 --output-init-sigma 0.3 --fitness-mode classic --points-weight 0.1 --style-strength 0.5 --styles-after-first-promotion
```

The surviving `pipeline.log` does not record the resume; shell history records this command, which resumed from the generation 30 checkpoint and stopped at generation 62:

```powershell
py -3 -u proyecto/phase1_trainer.py --output training_output/phase1 --generations 62 --population 64 --pairs 16 --workers 12 --seed 0 --validation-pairs 1000 --check-batch 200 --promotion-threshold 0.55 --patience 50 --finalists 8 --finalist-pairs-multiplier 4 --output-init-sigma 0.3 --fitness-mode classic --points-weight 0.1 --style-strength 0.5 --styles-after-first-promotion --resume training_output/phase1/checkpoint_0030.npz; "exit=$LASTEXITCODE" | Out-File -Encoding ascii training_output/phase1/resume_done.txt
```
 The pre-resume generation log is preserved separately as `generations_trial30.jsonl`; the backup reference evaluation and run configuration are identical to the final copies. Checkpoints are not included.

The promotion gate used the old early-stop rule in both directions: a challenger could be accepted or rejected before the full 1,000 pairs. Generation 3 was accepted early at 200 pairs. The initial champion was generation 1; promotions were generation 3 (60.8% over 200 pairs) and generation 12 (55.3% over 1,000 pairs). Training stopped at generation 62 after 50 generations without promotion, a plateau in measured promotions.

## Evaluation and decision

Reference evaluations are descriptive, not selection criteria. Win rates against Reactive and Adaptive Utility respectively were 34.5% / 15.0% for generation 1, 27.7% / 16.1% for generation 3, and 33.5% / 22.2% for generation 12.

A fresh network-only head-to-head used seed 777 validation-partition deals and 1,000 mirror pairs. Run 1 generation 48 won 48.95% ± 1.85 percentage points against run 2 generation 12. Run 2 therefore did not meet the pre-agreed 55% replacement rule, and run 1 remains shipped. These deals were held out from gate checks, but came from a validation partition repeatedly used for those checks.

SHA-256 champion identifiers:

- Run 1 generation 48, `champions/phase1/reigning_champion.npy`: `0b3e04fad09334f60d2d416b9c6848c7b3aabcfa6bb0d6593ac825feb32e47f7`
- Run 2 generation 12, `reigning_champion.npy`: `1c2b8378a0b7056d2d8c049f36aceb1a64e6a3e01c1ad89eb851e598d2c8db8d`

Do not claim that the runs are equivalent, that training converged, that the learning fixes caused a decline, or that the network beats the baselines. Self-play promotion is relative to the contemporaneous champion and does not establish monotonic general improvement.
