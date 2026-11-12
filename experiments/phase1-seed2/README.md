# Phase 1 seed 2 experiment

This experiment trained from scratch with seed 2, separately from shipped run 1 and the other seed experiments.

## Execution

The pipeline used its defaults, including reject-only early stopping and patience 50, at code revision `2e28156`. It allowed 1,000 generations. The pipeline log records the full trainer and evaluation commands and settings; the top-level command was:

```powershell
py -3 -u proyecto/run_phase1.py --seed 2 --workers 12 --eval-workers 12 --no-commit
```

The output paths in the log refer to `training_output/phase1`, renamed to `training_output/phase1_seed2` after the run. The pipeline published to `champions/phase1` locally; that directory was restored from git with `git checkout -- champions/phase1`, so the shipped champion remained run 1. Checkpoints are not included.

The initial champion was generation 1; generation 2 was promoted at 56.15% over 1,000 pairs. Training stopped at generation 52 after 50 generations without promotion. A PC shutdown interrupted training at generation 13; the pipeline resumed from `checkpoint_0010.npz`, and the trainer trimmed generation log rows after the checkpoint before continuing.

## Evaluation and decision

Reference evaluations are descriptive, not selection criteria. Rates and margins below are percentages from 1,000 mirror pairs; the final row evaluates the final generation, not the reigning champion.

| Generation | Reactive | Adaptive Utility |
|---|---|---|
| 1 | 17.15% ± 1.63 | 7.15% ± 1.10 |
| 2 | 47.05% ± 2.10 | 19.75% ± 1.66 |
| 52 | 40.20% ± 2.09 | 22.20% ± 1.73 |

A network-only head-to-head against run 1 generation 48 used 1,000 mirror pairs with seed 777 validation-partition deals. Seed 2's reigning champion won 46.6% ± 1.9 percentage points; see [the result](seed2_vs_run1.json) and [the head-to-head script](../phase1-seed1/h2h.py). The schedule uses the validation partition also used by gate checks, so a different seed alone does not establish that individual deals were unused. The result did not meet the agreed ≥55% replacement rule; by user decision on 2026-11-07, run 1 remains shipped. The interval measures match sampling, not the selection of runs or opponents.

SHA-256 champion identifiers:

- Run 1 generation 48, `champions/phase1/reigning_champion.npy`: `0b3e04fad09334f60d2d416b9c6848c7b3aabcfa6bb0d6593ac825feb32e47f7`
- Seed 2 generation 2, `reigning_champion.npy`: `5f00bfbaa27d7ee9fcedd7610ca36b092c71b8e935218a8f337a7cc1e25a3e79`

Self-play promotion is relative to the contemporaneous champion and does not establish monotonic general improvement; reference opponents do not measure human strength.
