# Phase 1 53% gate, seed 1

This isolated experiment trained from scratch with seed 1. The only change from the [seed 1 experiment](../phase1-seed1/README.md) was the in-run promotion threshold of 53%. It used the 2e28156-era trainer, with a maximum of 100 generations and patience 50. The trainer ran directly under a keep-awake wrapper, followed by descriptive reference evaluation; there was no publish step and `champions/` was never touched. Checkpoints and individual generation weights are not included.

## Execution

The trainer flags from the run script were:

```powershell
py -3 -u proyecto/phase1_trainer.py --output training_output/phase1_gate53_seed1 --seed 1 --workers 12 --population 64 --pairs 16 --generations 100 --patience 50 --validation-pairs 1000 --check-batch 200 --early-stop reject-only --promotion-threshold 0.53 --finalists 8 --finalist-pairs-multiplier 4 --output-init-sigma 0.3 --fitness-mode classic --points-weight 0.1 --style-strength 0.5 --styles-after-first-promotion
```

The actual invocation used a Python keep-awake wrapper around the trainer. Generation 1 was the initial champion; generations 7 (54.5%), 18 (53.0%) and 26 (54.8%) were promoted. Training stopped at generation 76, with generation 26 reigning. `champion_0027.npy` is byte-identical to the reigning champion: it is an elite copy. Generations 4 (52.65%) and 23 (52.70%) were rejected even though the console rounds each to 53%; precise checks are in [generations.jsonl](generations.jsonl).

## Reference evaluation

Reference results are descriptive only ([ADR 0021](../../docs/adr/0021-existing-agents-take-no-part-in-training.md)); rates and margins are percentages from 1,000 mirror pairs. The final generation is the final population's best, not the reigning champion.

| Generation | Reactive | Adaptive Utility |
|---|---|---|
| 26 (champion) | 58.75% ± 2.0 | 15.4% ± 1.5 |
| 76 (final) | 59.65% | 22.0% |
| Seed 1 generation 23 | 61.8% | 26.4% |

## Panel and decision

The pre-registered [ADR 0031](../../docs/adr/0031-frozen-evaluation-panel.md) evaluation used `proyecto/panel_evaluation.py`, panel-v1, seed 1 generation 23 as baseline, and schedule seeds 9001 (direct) and 9002 (panel). The candidate scored 53.09% ± 0.98 against the baseline over 4,000 mirror pairs, with a 52.11% lower bound. Panel rates below use 1,000 mirror pairs per opponent on the same panel schedule; see [panel-report.json](panel-report.json).

| Panel member | Candidate | Baseline |
|---|---|---|
| run1-gen22 | 57.60% | 59.60% |
| run1-gen48 | 53.90% | 51.15% |
| run2-gen12 | 50.65% | 50.20% |
| seed1-gen9 | 54.65% | 54.50% |
| seed2-gen2 | 53.75% | 54.05% |
| Mean | 54.11% | 53.90% |

Seed1-gen23 was skipped because it is the baseline. The run1-gen22 difference is −2.00 points, exactly at the permitted regression limit. The pre-registered decision is `promising`: the head-to-head gain over the baseline is solid, but the broad panel gain of about 0.2 points is within noise and one member sits exactly at the regression limit, so no general improvement is demonstrated. This is a single run; replication on more seeds was recommended, and the user chose not to replicate on 2026-11-07.

Against shipped run 1 generation 48, the candidate scored 53.90% on the 1,000-pair panel schedule, below the separate ≥55% replacement rule. Run 1 generation 48 stays shipped.

SHA-256 of [reigning_champion.npy](reigning_champion.npy): `b00e64118ce54e8063b7eb61c8085ec6093f732c00112d61a26a608ee666f5fd`.
