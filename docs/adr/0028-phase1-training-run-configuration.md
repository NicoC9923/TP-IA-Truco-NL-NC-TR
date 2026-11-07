# Phase 1 training run configuration

The first real phase 1 run uses these settings, chosen after measuring the engine on the 16-thread development machine (5- and 4-generation tuning runs, 2026-11-06):

| Setting | Value | Reason |
|---|---|---|
| Population | 64 | The spec's value and the usual range for a genome of 1,860 weights. A larger population costs league time linearly and, with noisy fitness, makes the top-fitness champion more often a lucky one that then fails the promotion check. |
| Pairs per individual | 16 | Every individual meets each of the 8 league slots (4 unbiased, 4 styles) twice. 32 matches give a fitness standard error about 30% lower than 8 pairs. Measured cost: league ~62 s instead of ~29 s per generation, about +33% per generation, because the promotion check was the larger cost before early stopping. |
| Workers | 12 | Leaves CPU for the presenter to use the machine during the run, at about 25% less throughput than 16. |
| Target | mixed 15 / 30, 30% random starting scores | As in the spec (ADR 0016). |
| Seed | 0 | Recorded so the run is reproducible. |
| Promotion check | up to 1,000 pairs, batches of 200, threshold 55% | ADR 0017. Measured: a clear result is decided in ~14 s (200 pairs) instead of ~70 s. |
| End of phase (patience) | 30 generations without a promotion | Raised from the starting value of 20. In tuning, every challenger lost clearly to the first reigning champion (40-48%), so promotions are expected to be rare early. With early stopping a stale generation costs only ~75 s, so 10 extra generations (~12 min) are cheap insurance against ending before evolution gets going. |
| Mutation | 10% of weights, gaussian sigma 0.05 | Unchanged (ADR 0026). Many small nudges is the usual scheme for evolving network weights; the 1/L rates of bit-string GAs don't apply. To be revisited only if diversity collapses or the run ends with no promotions. |
| Reference evaluation | after training, promoted champions only, 1,000 pairs, 12 workers | ADR 0022. Scoring every generation's champion during training would compete with the trainer for CPU and take hours. Generation 1, every promoted champion and the final one give the curve the evolution page needs. |

Expected duration: ~75 s per generation (up to ~130 s when a check is close), 40-80 generations, so about 1-2 hours of training plus 10-20 minutes of evaluation. If no promotion ever happens, the run ends at generation 31 (~40 min).

## How the run is executed

- In a dedicated worktree on branch `train/phase1-run1`, based on `feat/ia-entrenada`, so the integration branch and `main` aren't touched while it runs.
- With one command, `proyecto/run_phase1.py`, which chains training, reference evaluation, publishing to `champions/phase1/` and a local commit (ADR 0018). Re-running it resumes from the latest checkpoint. It keeps Windows awake and writes `training_output/phase1/pipeline.log`.
- The run branch is merged into `feat/ia-entrenada` afterwards, with the champion and its logs.

## Considered Options

- **Pairs 8:** half the league time, but 16 matches per individual leave fitness close to a coin flip.
- **Patience 20 (the starting value):** risks ending the phase at generation 21 before any promotion, given the tuning results.
- **Lowering the promotion threshold to get more promotions:** with a ±2.5% margin, a threshold near 52% would promote card luck and fill the hall of fame with noise.
- **Running the reference evaluation live with `--watch`:** useful for a live demo, but competes with the trainer for CPU. Left as an option, not the default.
- **Training in the main checkout:** the main checkout is on `main`, which doesn't have the trainer, and switching it would disturb the current demo.

## Changes after the first run

The table above records the historical first run, not the current defaults. The default patience is now 50 failed promotion checks; the Sonnet learning review F6 observed that 30 could cut off slow progress, while the empirical review subsequently found promotions at generations 22 and 48 and stopping at 78. Selection now uses the shared league and finalist re-test in ADR 0027, output initialization in ADR 0026, and the configurable fitness in ADR 0007. Phase 1 styles default to ±1.5 after the first promotion, a user-approved deviation from story 17's 50/50-from-the-start requirement; see ADR 0013 for the reasoning and evidence limits. Every new setting is forwarded by the pipeline and recorded in checkpoint configuration so resume detects mismatches.

Future runs default to reject-only early stopping (ADR 0017). Run 2 retains `both` when resuming its existing checkpoint.
