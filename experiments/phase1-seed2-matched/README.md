# Phase 1 matched encoder comparison, seed 2

The [pre-registered rule](RULE.txt), written before training, specifies a matched seed 2 comparison. V2 and v3 trained from scratch at revision `f8b4441`, with the corresponding seed 1 settings and seed 2. The [v1 baseline](../phase1-seed2/reigning_champion.npy) is seed 2 generation 2; its run used 1,000 generations and patience 50, stopping at generation 52, so budgets are comparable but not identical. The trainer ran directly without publishing; `champions/` was untouched. Checkpoints and individual generation weights are excluded.

## Execution and reference evaluation

Both runs used 12 workers, population 64, 16 pairs, a 100-generation cap, patience 50, 1,000 validation pairs, check batches of 200, reject-only early stopping, a 55% promotion threshold, eight finalists, finalist-pairs multiplier 4, output-init sigma 0.3, classic fitness, points weight 0.1, style strength 0.5 and styles after the first promotion. Only the encoder differed. The archived [v2](v2/train.log) and [v3](v3/train.log) logs and run data document execution.

V2 promoted generation 14 (55.8%) and generation 25 (56.4%), stopped at generation 75 and retained generation 25. V3 promoted generation 5 (56.9%) and generation 50 (55.3%), reached the generation 100 cap and retained generation 50. See their respective `generations.jsonl` files.

Reference evaluations use 1,000 mirror pairs and are descriptive only; they never select champions. V2 generation 25 scored 46.4% against Reactive and 32.35% against Adaptive Utility. V3 generation 50 scored 56.3% and 26.05%; its final generation 100, which was not a champion, scored 64.6% and 37.35%. See [v2](v2/reference_evaluation.jsonl) and [v3](v3/reference_evaluation.jsonl) reference results.

## Panel comparisons

Under [ADR 0031](../../docs/adr/0031-frozen-evaluation-panel.md), each direct comparison uses 4,000 mirror pairs and each panel row uses 1,000 pairs on the same panel schedule for candidate and baseline. Direct margins are pair-level 95% margins. A baseline identical to a panel member is skipped for both means.

| Panel member | C1 v2 / v1 | C2 v3 / v2 |
| --- | ---: | ---: |
| run1-gen22 | 55.65% / 53.25% | 56.30% / 56.00% |
| run1-gen48 | 50.45% / 48.90% | 50.75% / 51.65% |
| run2-gen12 | 52.15% / 47.00% | 50.60% / 52.40% |
| seed1-gen9 | 53.70% / 49.40% | 55.00% / 54.50% |
| seed1-gen23 | 52.35% / 44.70% | 52.60% / 52.90% |
| seed2-gen2 | skipped (baseline) | 54.20% / 53.95% |
| Mean | 52.86% / 48.65% | 53.24% / 53.57% |

[C1](C1_v2_vs_v1.json), v2 versus v1, used schedules 9211/9212: direct 54.69% ± 0.92 percentage points and panel difference +4.21 points, yielding **promising**. [C2](C2_v3_vs_v2.json), v3 versus v2, used 9213/9214: direct 50.15% ± 0.85 and panel difference −0.32 points, yielding **no demonstrated benefit** (displayed means are rounded independently). [C3](C3_v3_vs_v1.json), v3 versus v1, was descriptive and used 9215/9216: direct 53.08% ± 0.97, means 52.81% versus 47.53% (+5.28 points), yielding **promising**.

Under the pre-registered interpretation, v2 replicates because C1 is promising. The seed 1 v3 failure is not confirmed as worse: C2 is not promising, but its direct rate of 50.15% is not below 50%. V3 shows no panel benefit over v2 on either seed: seed 1 was worse, seed 2 a tie, while v3 has the best scores against the descriptive reference opponents. These results do not establish general superiority, convergence or human strength.

Shipping was not tested here. No candidate is near the separate ≥55% rule against run 1 generation 48; v2 seed 2 generation 25 scored 50.45% against it on the C1 panel schedule. Run 1 generation 48 stays shipped.

SHA-256 of [v2 reigning champion](v2/reigning_champion.npy): `3f25fb0b59b5c7a10c4666bb2450717f822ad6d13c8283a217069645a7f68a06`.

SHA-256 of [v3 reigning champion](v3/reigning_champion.npy): `6664e68379ef90db275ab6432b2a2b3df3562104bfe24a239e671452d77bae38`.
