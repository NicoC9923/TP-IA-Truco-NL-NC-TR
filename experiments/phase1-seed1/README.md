# Phase 1 seed 1 experiment

This experiment trained from scratch with seed 1, separately from shipped run 1 and the other seed experiments.

## Execution

The pipeline used its defaults, including reject-only early stopping and patience 50, at code revision `2e28156`. It allowed 1,000 generations. The pipeline log records the full trainer and evaluation commands and settings; the top-level command was:

```powershell
py -3 -u proyecto/run_phase1.py --seed 1 --workers 12 --eval-workers 12 --no-commit
```

The output paths in the log refer to `training_output/phase1`, renamed to `training_output/phase1_seed1` after the run. The pipeline published to `champions/phase1` locally; that directory was restored from git with `git checkout -- champions/phase1`, so the shipped champion remained run 1. Checkpoints are not included.

The initial champion was generation 1; generations 9 and 23 were promoted at 55.45% and 57.2%, respectively, over 1,000 pairs. Training stopped at generation 73 after 50 generations without promotion. Late checks at generations 65, 66 and 71 reached 53.8%, 54.95% and 54.3%, respectively, over 1,000 pairs.

## Evaluation and decision

Reference evaluations are descriptive, not selection criteria. Rates and margins below are percentages from 1,000 mirror pairs; the final row evaluates the final generation, not the reigning champion.

| Generation | Reactive | Adaptive Utility |
|---|---|---|
| 1 | 28.35% ± 1.93 | 14.55% ± 1.48 |
| 9 | 43.85% ± 2.06 | 15.30% ± 1.54 |
| 23 | 61.80% ± 2.08 | 26.40% ± 1.82 |
| 73 | 59.95% ± 2.11 | 21.95% ± 1.68 |

A network-only head-to-head against run 1 generation 48 used 1,000 mirror pairs with seed 777 validation-partition deals. Seed 1's reigning champion won 53.1% ± 2.0 percentage points (approximately 51.1–55.1%); see [the result](seed1_vs_run1.json) and [the head-to-head script](../phase1-seed1/h2h.py). The schedule uses the validation partition also used by gate checks, so a different seed alone does not establish that individual deals were unused. The result did not meet the agreed ≥55% replacement rule; by user decision on 2026-11-07, run 1 remains shipped. The interval measures match sampling, not the selection of runs or opponents.

SHA-256 champion identifiers:

- Run 1 generation 48, `champions/phase1/reigning_champion.npy`: `0b3e04fad09334f60d2d416b9c6848c7b3aabcfa6bb0d6593ac825feb32e47f7`
- Seed 1 generation 23, `reigning_champion.npy`: `413542520b893418371fa0ff29b4b693d0675ef60cd4be55f8e782dd544db31c`

Self-play promotion is relative to the contemporaneous champion and does not establish monotonic general improvement; reference opponents do not measure human strength.

A bounded extension from checkpoint 70 with patience 100 is planned as a separate experiment.
