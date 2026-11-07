# Phase 1 seed 1 extension experiment

This experiment resumed [seed 1](../phase1-seed1/README.md) as the bounded extension described in [ADR 0030](../../docs/adr/0030-patience-extension.md).

## Execution

On a copy of the seed 1 output folder, the pipeline resumed `checkpoint_0070` at code revision `2fbbcef` and extended patience from 50 to 100. The top-level command was:

```powershell
py -3 -u proyecto/run_phase1.py --seed 1 --workers 12 --eval-workers 12 --extend-patience 100 --no-commit
```

The pipeline log records the trainer and evaluation commands and settings. Its output paths refer to `training_output/phase1`; the extension output was retained as `training_output/phase1_seed1_ext`. The publish step overwrote `champions/phase1` locally; it was restored with `git checkout -- champions/phase1`. Checkpoints are not included.

The generation log contains the original rows 1–70 followed by the resumed rows 71–123. Generation 71 records `patience_extended_from: 50` and `patience: 100`. Replayed generations 71–73 matched the original promotion-check win rates (0.543, 0.485 and 0.4975) and maximum fitness exactly. The patience metadata and generation 73 stopping flag differ as expected for the extension.

There were no promotions in generations 74–123. Training stopped at generation 123, 100 generations after the generation 23 promotion. Near-misses at or above 53.5% over 1,000 pairs were generation 121 (54.25%), 87 (54.15%), 118 (54.1%), 82 (54.0%) and 85 (53.8%).

## Evaluation and decision

Reference evaluations are descriptive, not selection criteria (spec #15 and [ADR 0021](../../docs/adr/0021-existing-agents-take-no-part-in-training.md)). Rates and margins below are percentages from 1,000 mirror pairs. Generation 123 is the final population's best, not a promoted champion.

| Generation | Reactive | Adaptive Utility |
|---|---|---|
| 23 (reigning champion) | 61.8% ± 2.1 | 26.4% ± 1.8 |
| 123 (final population's best) | 65.6% ± 2.0 | 24.7% ± 1.7 |

These reference results do not change any champion. Seed 1's champion remains generation 23 and run 1 generation 48 remains shipped, following the decision in [the seed 1 record](../phase1-seed1/README.md).

The extension's reigning champion is byte-identical to [seed 1's generation 23 champion](../phase1-seed1/reigning_champion.npy), so it is not duplicated here. Its SHA-256 is `413542520b893418371fa0ff29b4b693d0675ef60cd4be55f8e782dd544db31c`.
