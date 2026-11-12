# Phase 1 parent selection diagnostic

These frozen seed 1 populations were evaluated with the trainer's league: 16 first-pass pairs and an independent shared 64-pair schedule, 200 tournaments, 2,000 bootstrap replicates and 12 workers. The checkpoints saved after generations 10 and 70 contain the populations for generations 11 and 71. Each tournament samples three candidate slots with replacement, selects the first-pass maximum, and compares its independent fitness with the mean of the other two slots. The random baseline selects the first slot. Best fractions split ties among candidate slots; chance is one third. JSON files contain scores, schedules, configuration counts, and exact results.

[ADR 0037](../../docs/adr/0037-parent-selection-diagnostic.md) preregistered reliability as a strictly positive 95% lower bound for winner minus losers in **both** populations. The verdict is **no demonstrated selection advantage**: the late population's interval includes zero. This does not imply the genetic algorithm is broken.

| Checkpoint | Winner advantage (95% CI) | Advantage / population SD | Winner best | Random advantage (95% CI) | Random best | Independent SD |
|---|---:|---:|---:|---:|---:|---:|
| 0010 | 0.02466 [0.00646, 0.04179] | 0.439 | 0.4600 | 0.00164 [-0.00927, 0.00915] | 0.360 | 0.05617 |
| 0070 | -0.00197 [-0.01599, 0.01536] | -0.047 | 0.3575 | 0.00307 [-0.00706, 0.00696] | 0.355 | 0.04155 |

The percentile interval resamples individuals with their paired scores and redraws tournaments; it accounts for repeated individuals but conditions on the measured league schedules. The results represent two populations from one training seed and one independent measurement schedule each, not uncertainty over schedules, training runs, or downstream evolutionary improvement. Fitness includes the run's configured point component, so differences are in fitness units rather than pure win rates.

The diagnostic took 222.64 seconds for checkpoint 0010 and 159.17 seconds for checkpoint 0070, 381.81 seconds total (6.36 minutes), excluding the test suite. Both used the default 12 workers and no training was performed.
