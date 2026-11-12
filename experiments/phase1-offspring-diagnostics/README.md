# Phase 1 offspring diagnostics

The seed 1 generation 70 checkpoint was evaluated without training. The 32 uniformly sampled parent pairs use replacement because the checkpoint contains no fitness scores; four other population members were reserved as opponents alongside the reigning champion and two hall entries. Each network was measured on the same 32 mirrored pairs. Confidence intervals are normal paired intervals across parent pairs; ties do not count as beating the parent.

ADR 0033 fixed the four decision rules and their thresholds before either run. Each offspring behavior was measured over the same 2,000 legal observations, using a fresh default opponent profile for a common context.

| Variant | Mean change | 95% CI | Fraction beating parent | Spread | Child-parent TV | Argmax agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crossover_mutation | -0.0132 | [-0.0341, 0.0078] | 0.3125 | 0.0605 | 0.1512 | 0.6623 |
| crossover_only | -0.0166 | [-0.0382, 0.0050] | 0.3750 | 0.0622 | 0.1496 | 0.6661 |
| mutation_0.025 | 0.0020 | [-0.0089, 0.0128] | 0.4688 | 0.0312 | 0.0117 | 0.9733 |
| mutation_0.050 | -0.0063 | [-0.0217, 0.0090] | 0.4062 | 0.0443 | 0.0226 | 0.9489 |
| mutation_0.100 | -0.0068 | [-0.0259, 0.0122] | 0.3438 | 0.0550 | 0.0447 | 0.8983 |

Crossover plus mutation minus sigma-0.05 mutation has mean -0.0068, 95% CI [-0.0333, 0.0196]. Population pairwise TV is 0.1981 (argmax agreement 0.5864); panel TV is 0.2505 (agreement 0.4636).

Rules fired: none. Recommendation: no operator diagnosis; consider inputs (proposal 7).

Offspring run: 210.4 seconds; behavior run: 20.0 seconds; both used 12 workers. These findings concern one frozen population and one fixed opponent set; repeated parent selection and shared schedules can correlate samples, and the simple confidence intervals do not model that dependence. The bank uses a default profile rather than each policy's evolving opponent history; panel members are a reference scale, not a statistical control.
