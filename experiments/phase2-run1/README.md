# Phase 2 run 1 (#29)

The first real phase 2 training run. It followed the [pre-registered rule](RULE.txt), including its two amendments, which were written before they took effect. The published champion is **generation 28**, in [`champions/phase2/`](../../champions/phase2/). It becomes the baseline that run 2 must beat.

## The aborted first attempt

The first launch used seed 52001 and check base 53000, starting 2026-11-08 at 23:50. It was stopped after about 80 minutes and 6 generations because of a rules bug in `proyecto/symmetric_engine.py`. Answering truco with retruco, or retruco with vale cuatro, did not accept the lower bet. So envido stayed legal after a retruco, and a later fold scored 1 point short. This happened in about 0.26% of decisions. ADR 0055 records the fix.

A full review before the re-launch found and fixed six more problems:

1. Training and validation deals could overlap. They now come from disjoint partitions: training uses partition 1, checks use partition 0.
2. The promotion interval ignored that panel members share deals. It is now clustered by pair, and the thresholds are unchanged.
3. Champion export was not atomic. It is atomic now, and a resume reconciles it.
4. Empty learner rounds are now handled.
5. Every setting must now be a finite number.
6. Checkpoints now record the engine and encoder versions.

Two inference shortcuts with bit-identical results made training about 11% faster. The aborted outputs are archived in `TP-IA-Truco-training-archive/phase2-run1-aborted-engine-bug` and were not used for anything. That attempt does not count as a retry.

## The run

The re-launch used seed 54001 and check base 55000. Its settings are in the rule: temperature 0.05, profile prior 5, 57,120 matches per generation, 12 workers, mixed targets, auxiliary head off, panel v1 and patience 40.

- **8 hours** were the planned budget, which ended at generation 34 by the time cap. The 8 h result is copied to `training_output/phase2_8h`.
- **The 4 h extension** was pre-registered and user-approved. It resumed the 8 h checkpoint to a 12 h total, after its code passed tests and review ([extension record](../../champions/phase2/time_cap_extension.json), [runner](run_extension.ps1)). It ended at the cap after generation 48, at 11.65 h of active time.
- **Totals:** 2,735,040 training matches, 91.5 million learner samples and 184,354 updates.

### Promotions and the plateau

There were six promotions: generations 3, 4, 7, 8, 10 and 28.

- Each promotion needed a panel gain of at least 0.01 with a 95% lower bound above 0, plus at least 47.5% in the direct match.
- Gains at promotion ranged from +4.3 to +7.0 points, with lower bounds from +0.1 to +3.0.
- At generation 14 the panel rotated automatically to **panel v2** (`panel-v2.json` next to the champion). v2 replaced run1-g48 and the measuring-run network with run 1's own generations 2 and 12.
- After generation 10 the challenger hovered around the champion: gains of −4 to +3.5 points, with lower bounds below zero. The one exception was generation 28, at +5.2 points with a lower bound of +1.5.
- Generations 29 to 48 brought no promotion: 19 checks failed in a row (generation 48 was cut by the cap before its check). Patience (40) was never reached, so the time cap ended the run.
- The 8 h champion and the final champion are the same model, generation 28. Its tensors match `generation_000028.pt`.

## Evaluation against the reference opponents

[Pre-registered](reference-eval/RULE.txt) and descriptive only (ADR 0021). It used 400 identical mirror pairs per opponent, validation deals and seed 56001. Win rates with adaptation on, with 95% margins:

| Network | vs Reactive | vs Adaptive Utility | Equal-weight mean |
|---|---|---|---|
| gen4 | 46.9% ± 3.1 | 23.5% ± 2.6 | 35.2% ± 2.2 |
| gen10 | 68.4% ± 2.8 | 37.2% ± 2.8 | 52.8% ± 2.0 |
| gen28 | 68.8% ± 2.9 | 34.6% ± 3.0 | 51.7% ± 2.0 |

Paired differences on the same deals:

- **gen10 minus gen4:** +17.6 ± 2.7 points. Early training clearly helped.
- **gen28 minus gen10:** −1.1 ± 2.5 points. That is no measurable change, so the late panel progress (generation 28's promotion) did not carry over to these opponents.
- **Adaptation on minus off:** +2.8 ± 1.9 for gen4, −1.4 ± 2.0 for gen10 and +0.2 ± 2.0 for gen28. Reading the opponent's profile makes no measurable difference for the trained networks.
- **Overfitting flag:** not applicable, because the 8 h and final champions are the same model.

([results](reference-eval/results.json), [output](reference-eval/output.txt))

## What we learned

These are findings from a Codex diagnosis (`specs/opinion-adaptive-utility*.md.out` in the coordinator's notes), not pre-registered tests:

- **A narrow league.** About 90% of the training opponents were closely related networks: itself, its own snapshots and phase 1 networks. The selection panel was made only of networks too. So the run learned to beat networks like itself.
- **Points are lost on bets.** Against Adaptive Utility, the losses concentrate in accepted envidos, and in folds and refusals of truco.
- **The Adaptive Utility reference does not adapt.** The wrapper in `proyecto/engine_policies.py` never calls its adaptation step, so it plays a fixed strategy. The 34.6% is a score against a fixed, strong heuristic, not against an opponent that learns during the match.

Run 2 changes the league and the panel in response; see its own rule in `experiments/phase2-run2/`.

## Files

- `champions/phase2/`: weights `reigning_champion.pt` (SHA-256 `05f18949…cdcd`) and sidecar `reigning_champion.json` (generation 28, temperature 0.05, profile prior 5). Also copies of `generations.jsonl`, `progress.json`, `panel-v2.json` and `time_cap_extension.json`.
- Full outputs (`training_output/phase2`, `phase2_8h`, champion) are archived in `TP-IA-Truco-training-archive/phase2-run1/`.
