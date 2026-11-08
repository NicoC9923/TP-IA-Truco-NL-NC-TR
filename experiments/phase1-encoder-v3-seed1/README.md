# Phase 1 encoder v3-core, seed 1

This experiment trained from scratch with seed 1. The only trainer flag change from the [v2 seed 1 run](../phase1-encoder-v2-seed1/README.md) was `--encoder v3`: the [ADR 0036](../../docs/adr/0036-phase1-encoder-v3-core.md) v3-core package bundles table cards by Truco rank, per-card current-trick values, pending envido stakes, corrected truco-resolution profile counting and an explicit trick index. Its component effects cannot be attributed separately. It used revision `f8b4441`, including the [bit-identical speed optimization](../../docs/perf/phase1-trainer-optimization.md), a 55% promotion gate, a maximum of 100 generations and patience 50. The trainer ran directly under a keep-awake wrapper; there was no publish step and `champions/` was untouched. Training took about 34 minutes. Checkpoints and individual generation weights are not included.

## Execution

The trainer flags from the run script were:

```powershell
py -3 -u proyecto/phase1_trainer.py --output training_output/phase1_encoder_v3_seed1 --seed 1 --workers 12 --population 64 --pairs 16 --generations 100 --patience 50 --validation-pairs 1000 --check-batch 200 --early-stop reject-only --promotion-threshold 0.55 --encoder v3 --finalists 8 --finalist-pairs-multiplier 4 --output-init-sigma 0.3 --fitness-mode classic --points-weight 0.1 --style-strength 0.5 --styles-after-first-promotion
```

Generation 1 was the initial champion. Promotions occurred at generation 5 (55.6%) and generation 55 (56.6%). The run reached the generation-100 cap, with generation 55 reigning and a [v3 encoder sidecar](reigning_champion.encoder.json); see [generations.jsonl](generations.jsonl).

## Reference evaluation

Reference results are descriptive only ([ADR 0021](../../docs/adr/0021-existing-agents-take-no-part-in-training.md)); they use 1,000 mirror pairs. Generation 55 scored 62.25% ± 2.0 against Reactive and 27.9% ± 1.8 against Adaptive Utility, the best results against both reference opponents among networks evaluated so far. For comparison, v2 seed 1 generation 46 scored 56.2% and 17.15%, and v1 seed 1 generation 23 scored 61.8% and 26.4%. See [reference_evaluation.jsonl](reference_evaluation.jsonl).

## Panel and decision

The [pre-registered rule](PANEL_RULE.txt), written before evaluation, compared the v3 candidate with v2 seed 1 generation 46 and panel-v1 under [ADR 0031](../../docs/adr/0031-frozen-evaluation-panel.md). Fresh schedule seeds 9201 and 9202 were used for the direct and panel comparisons. The candidate scored 48.59% ± 0.99 against v2 over 4,000 mirror pairs; the 95% upper bound was 49.58%, so v2 was better head-to-head on this evaluation. Panel rates use 1,000 mirror pairs per member on the same panel schedule; see [panel-report.json](panel-report.json).

| Panel member | Candidate v3 | Baseline v2 |
|---|---:|---:|
| run1-gen22 | 58.20% | 59.85% |
| run1-gen48 | 52.55% | 53.35% |
| run2-gen12 | 50.95% | 52.65% |
| seed1-gen9 | 53.10% | 58.55% |
| seed1-gen23 | 49.60% | 52.30% |
| seed2-gen2 | 55.75% | 56.75% |
| Mean | 53.36% | 55.58% |

The candidate mean was 2.22 percentage points lower; the pre-registered decision is `no demonstrated benefit`. On these fresh schedules v2 again beat each of the five panel members it had beaten previously (mean about 56%), and beat v1 seed 1 generation 23 at 52.30%. Thus its earlier panel result was not an artefact of reused deals, though distinct seeds do not guarantee entirely disjoint individual deals.

The panel and reference opponents rank v2 and v3 in opposite directions. Non-transitive strength is plausible but not proven. This is one run, and which v3 component caused the panel drop is unknown. A matched seed 2 comparison of v1, v2 and v3 is under way.

Shipping was not tested; run 1 generation 48 stays shipped. The panel result is not the separate pre-registered shipping test.

SHA-256 of [reigning_champion.npy](reigning_champion.npy): `24773046ba6753f1f13614fda74d15a31b1756c2976a76cc6eb384504a311763`.
