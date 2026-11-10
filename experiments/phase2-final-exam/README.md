# Phase 2 final exam, first use (run 2 candidates)

The exam followed the frozen [rule](RULE.txt) (ADR 0061). Each part ran once with 12 workers and exited 0. Raw records, configuration and results are in [part-t/](part-t/) and [part-e/](part-e/). The coordinator independently recomputed every Part E gain and lower bound from `part-e/partial.jsonl`: 5,000 unique records, 1,000 unique deal seeds per opponent, and 500 pairs at each target per opponent.

## Part T: temperature (exploratory)

Run 2 generation 38 played 300 mirror pairs against each opponent at each setting. Win rates have 95% intervals.

| Selection | vs Adaptive Utility | vs phase 1 g48 | Mean |
|---|---:|---:|---:|
| T = 0.05 | 52.0% [48.7, 55.3] | 82.8% [80.0, 85.6] | 67.4% |
| T = 0.02 | 65.7% [62.4, 68.9] | 90.3% [88.0, 92.6] | 78.0% |
| Greedy | 73.2% [70.0, 76.3] | 92.8% [90.8, 94.9] | 83.0% |

Greedy beat T = 0.05 by +15.6 points on the mean (paired 95% interval [+13.0, +18.2]). That exceeds the 2-point threshold, so the candidates played greedy in Part E.

## Part E: exam

The candidates played greedy. The baseline, shipped run 1 generation 28, played at T = 0.05. Each cell is 1,000 mirror pairs on shared deals and action seeds. Gains are paired differences against the baseline, with 95% intervals.

| Opponent | g28 (T 0.05) | g38 greedy | Gain g38 | g49 greedy | Gain g49 |
|---|---:|---:|---:|---:|---:|
| Adaptive Utility (fixed) | 34.6% | 71.5% | +36.9 [34.3, 39.5] | 64.5% | +29.9 [27.3, 32.5] |
| Reactive | 68.8% | 99.3% | +30.4 [28.6, 32.3] | 99.0% | +30.1 [28.3, 32.0] |
| Phase 1 g48 | 77.8% | 92.5% | +14.7 [12.7, 16.7] | 92.5% | +14.7 [12.7, 16.7] |
| Held-out v1s1-g23 | 76.4% | 91.9% | +15.5 [13.5, 17.5] | 91.8% | +15.4 [13.4, 17.5] |
| Held-out v1s2-g2 | 79.2% | 93.2% | +14.1 [12.2, 15.9] | 92.8% | +13.6 [11.6, 15.5] |

Both candidates pass the primary test (Holm one-sided p < 1e-100), all guardrails and the balance test; the balance means are +22.3 points for g38 and +20.8 points for g49. By the fixed ranking, **generation 38 ships, played greedy**.

## What this does and does not show

- **Most of the gain comes from greedy play, not from run 2's training.** Part T alone moved generation 38 from 52% to 73% against Adaptive Utility. The exam compares two shipped packages (network plus selection rule), and g28 was not tested greedy. So these results do not separate the effect of run 2 from the effect of greedy play.
- **Adaptive Utility and Reactive were run 2 training opponents** (ADR 0056), so results against them are against familiar opponents. Phase 1 g48 and the two held-out networks were never in run 2's training or selection, and the gains against them are of similar size.
- **The candidates were named after their development checks had been seen.** Fresh seeds make the exam unbiased for these two named candidates; the ranking was fixed before play.
- **Greedy play is deterministic for a given situation,** so a human who plays it many times may learn to read it. ADR 0061 ships the candidate with the selection rule it was examined with.
- The intervals describe this protocol and these opponents. They do not establish universal strength.

## Files

The rule names `champions/phase2-run2/reigning_champion.pt` for candidate C1. That file is byte-identical (SHA-256 `95eee708…60bf`) to the shipped `champions/phase2/reigning_champion.pt`, and an identical copy with its sidecar is kept in the training archive at `TP-IA-Truco-training-archive/phase2-run2/champions/phase2-run2/`. To reproduce the exam, restore that folder to the path the rule names.

The rule's other inputs, `training_output/phase2-run2/generation_000038.pt` and `generation_000049.pt`, were removed from the working tree after verifying all 102 files of `training_output/phase2-run2` by SHA-256 against `TP-IA-Truco-training-archive/phase2-run2/manifest.json`. Restore that folder to reproduce the exam.
