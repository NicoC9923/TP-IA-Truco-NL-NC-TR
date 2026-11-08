# Phase 1 encoder v2, seed 1

This isolated experiment trained from scratch with seed 1. The only change from the [seed 1 experiment](../phase1-seed1/README.md) was `--encoder v2`, encoding held-card Truco rank as described in [ADR 0035](../../docs/adr/0035-phase1-encoder-v2.md). It used code revision `9a16b0c`, a 55% promotion gate, at most 100 generations and patience 50. The trainer ran directly under a keep-awake wrapper, followed by descriptive reference evaluation; there was no publish step and `champions/` was untouched. Checkpoints and individual generation weights are not included.

## Execution

The trainer flags from the run script were:

```powershell
py -3 -u proyecto/phase1_trainer.py --output training_output/phase1_encoder_v2_seed1 --seed 1 --workers 12 --population 64 --pairs 16 --generations 100 --patience 50 --validation-pairs 1000 --check-batch 200 --early-stop reject-only --promotion-threshold 0.55 --encoder v2 --finalists 8 --finalist-pairs-multiplier 4 --output-init-sigma 0.3 --fitness-mode classic --points-weight 0.1 --style-strength 0.5 --styles-after-first-promotion
```

Generation 1 was the initial champion. The only promotion was generation 46 (55.20%); training stopped at generation 96, with generation 46 reigning and a [v2 encoder sidecar](reigning_champion.encoder.json). The first promotion came later than v1 seed 1's generation 9. Rejected near-misses at or above 54% were generations 29 (54.00%), 31 (54.30%), 35 (54.10%), 40 (54.05%), 62 (54.65%), 78 (54.75%) and 92 (54.40%). The console rounds to whole percentages; precise rates are in [generations.jsonl](generations.jsonl).

## Reference evaluation

Reference results are descriptive only ([ADR 0021](../../docs/adr/0021-existing-agents-take-no-part-in-training.md)); the v2 run was evaluated with `--encoder v2`. Rates and margins are percentages from 1,000 mirror pairs. The final generation is the final population's best, not the reigning champion.

| Generation | Reactive | Adaptive Utility |
|---|---|---|
| 46 (champion) | 56.2% ± 2.0 | 17.15% ± 1.5 |
| 96 (final) | 56.1% | 21.35% |
| v1 seed 1 generation 23 | 61.8% | 26.4% |

## Panel and decision

The pre-registered [ADR 0031](../../docs/adr/0031-frozen-evaluation-panel.md) evaluation used candidate encoder v2, baseline and panel encoder v1, and schedule seeds 9001 (direct) and 9002 (panel). Against the seed 1 generation 23 baseline, the candidate scored 53.79% ± 1.01 over 4,000 mirror pairs, with a 52.78% lower bound. Panel rates below use 1,000 mirror pairs per opponent on the same panel schedule; see [panel-report.json](panel-report.json).

| Panel member | Candidate | Baseline |
|---|---|---|
| run1-gen22 | 60.05% | 59.60% |
| run1-gen48 | 54.80% | 51.15% |
| run2-gen12 | 52.90% | 50.20% |
| seed1-gen9 | 55.95% | 54.50% |
| seed2-gen2 | 56.15% | 54.05% |
| Mean | 55.97% | 53.90% |

Seed1-gen23 was skipped because it is the baseline. The pre-registered decision is `promising`: the mean gain was 2.07 percentage points and every compared member improved, with no regression. This is one run, and seeds 9001 and 9002 were already used for the 53% gate evaluation, so this is exploratory rather than fresh confirmation. Reference opponents moved in the opposite direction from the panel comparison. Networks using v2 and v1 cannot be compared input-for-input because their encodings differ; they can be compared by play.

Against shipped run 1 generation 48, the candidate scored 54.80% on the 1,000-pair panel schedule, below the separate ≥55% replacement rule. Run 1 generation 48 stays shipped.

SHA-256 of [reigning_champion.npy](reigning_champion.npy): `9659d27a3c70f369751d48a9ba4d97f555f4d8cf20931cbbd23f06c90ecd7fe6`.
