# Phase 1 matched encoder comparison, seed 3

The [pre-registered rule](RULE.txt), written before training, specifies a matched seed 3 comparison of v2 against v1. Both trained from scratch with identical settings except the encoder: 12 workers, population 64, 16 pairs, a 100-generation cap, patience 50, 1,000 validation pairs, check batches of 200, reject-only early stopping, a 55% promotion threshold, eight finalists, finalist-pairs multiplier 4, output-init sigma 0.3, classic fitness, points weight 0.1, style strength 0.5 and styles after the first promotion. The [runner](run_seed3_matched.ps1), [v1 log](v1/train.log) and [v2 log](v2/train.log) document execution. Training ran directly without publishing; checkpoints and individual generation weights are excluded.

V1 promoted generation 10 (58.8%), stopped at generation 60 and retained generation 10. V2 promoted generation 23 (56.4%), stopped at generation 73 and retained generation 23. Their respective `generations.jsonl` files record the runs. Reference evaluations use 1,000 mirror pairs and are descriptive only; they do not select champions. V1 generation 10 scored 49.7% against Reactive and 29.1% against Adaptive Utility; v2 generation 23 scored 38.75% and 32.05%, respectively. See the [v1](v1/reference_evaluation.jsonl) and [v2](v2/reference_evaluation.jsonl) reference results.

## Panel comparison

[C1s3](C1s3_v2_vs_v1.json) compares the frozen v2 candidate with the v1 baseline under ADR 0031, using 4,000 direct mirror pairs on schedule 9221 and 1,000 pairs per panel member on schedule 9222. The direct result was 50.29% ± 0.96 percentage points (paired 95% margin).

| Panel member | V2 candidate | V1 baseline |
| --- | ---: | ---: |
| run1-gen22 | 54.40% | 55.80% |
| run1-gen48 | 50.65% | 50.20% |
| run2-gen12 | 50.40% | 48.15% |
| seed1-gen9 | 53.45% | 52.35% |
| seed1-gen23 | 49.85% | 51.35% |
| seed2-gen2 | 53.10% | 54.85% |
| Mean | 51.98% | 52.12% |

The panel difference is −0.14 percentage points: **no demonstrated benefit** on seed 3. The distinct schedule seeds do not guarantee disjoint individual deals.

## Across seeds

These summaries give equal weight to each training seed, rather than pooling matches. Seed 1's panel used reused schedule seeds and was re-confirmed later on fresh seeds in the v3 comparison; seed 2 and seed 3 used fresh schedules. See the [seed 1](../phase1-encoder-v2-seed1/README.md), [v3 seed 1](../phase1-encoder-v3-seed1/README.md) and [matched seed 2](../phase1-seed2-matched/README.md) records.

| Seed | Direct v2 vs v1 | Panel difference | Pre-registered interpretation |
| --- | ---: | ---: | --- |
| 1 | 53.79% | +2.07 points | Promising |
| 2 | 54.69% | +4.21 points | Promising |
| 3 | 50.29% | −0.14 points | No demonstrated benefit |
| Equal-seed mean | ≈52.9% | ≈+2.0 points | — |

V2 helps on average in these comparisons but not on every seed; this is moderate, not conclusive evidence. The panel is development evidence, not a shipping test. No champion was selected or replaced.

SHA-256 of [v1 reigning champion](v1/reigning_champion.npy): `9986cf1815f806edfb33e9d1095aab972102c3f066c2fe78a9b6773a27d7364e`.

SHA-256 of [v2 reigning champion](v2/reigning_champion.npy): `8777c3302d504dd8ac4375013098b78a58a4ac0dfd38b57ecb87431fd7c0d7de`.
