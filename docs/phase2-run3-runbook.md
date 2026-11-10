# Phase 2 run 3: tools, measurements and commands

Implements issues #39 (encoder) and #40 (target) and the greedy checks of ADR 0062 (with its amendment on
generation and check sizes). Nothing here was run for real; only synthetic and tiny seeded runs.

## What was built

| Piece | File | Interface |
|---|---|---|
| Truco strength percentiles | `proyecto/truco_strength.py`, `truco_strength_percentiles.json` | `percentile(cards)`; `py truco_strength.py --write` regenerates the table, a test checks it is identical |
| Hand facts / encoder versions | `phase2_hand_facts.py`, `phase2_network.py` | `encode_observation(..., encoder_version=1/2/3)`; sizes 1628 / 1634 / 1635 |
| Warm start | `phase2_torch.py` | `widen_state_dict(weights, version)`, `Phase2Network.from_state(weights)`; the encoder follows the network's input width |
| Equity table | `phase2_equity.py` | `EquityTable.equity(own, opp, target, own_is_mano)`, `build_equity_table`, `brier_comparison`, CLI `build` / `check` |
| Blended target | `phase2_target.py` | `decision_targets(result, seat, table, blend)` |
| Generation schedule | `phase2_schedule.py` | one logged size change on resume |
| Pilot | `phase2_run3_pilot.py` | `run_pilot(...)`, seed audit, generation sizing |

Encoder columns appended after the original 1,628 inputs: own envido total / 33; flush-pair bit; truco percentile
of the three-card hand; percentile of the unplayed own cards (0 when none); envido stake and real envido stake / 30
(0 when that call is not legal, pending chain included); version 3 only: P(own tanto beats the opponent's) from
the exact-filter beliefs, ties to mano. Percentiles are the mid-rank of `poder_mano_truco` among all C(40, k)
card sets; the table is computed from the rules module only.

New trainer flags (all optional; absent means run 1 / run 2 behaviour and the golden stays identical):
`--encoder-version {1,2,3}`, `--warm-start PATH`, `--target-blend B` with `--equity-table PATH`
(`target = (1 - B) * result + B * (2 * equity - 1)`), `--check-selection {sampled,greedy}`,
`--change-matches-per-generation` (with `--resume`). `--panel-pairs` and `--head-to-head-pairs` already existed.
When any run 3 option is set the generation log gains `decisions` and `greedy_disagreement` (fraction of learner
decisions where the sampled action differs from the greedy one). Options are recorded in the checkpoint config
only when set (`encoder_version`, `warm_start_sha256`, `target_blend`, `equity_table_sha256`, `check_selection`).
With greedy checks the exported champion's sidecar carries `"selection": "greedy"`.

## Equity table method

Direct empirical cells with smoothing, not a DP: each hand start of greedy self-play (random start scores, mixed
targets, training deal partition) gives one observation per player view. A cell is
`(wins + 5 * prior) / (visits + 5)` with a symmetric logistic prior on the remaining-points difference. The table is
then forced monotone (average of the upper and lower monotone envelopes) and symmetrised
(`eq(own, opp, m) + eq(opp, own, not m) = 1`). Terminal scores are exact. A DP over per-hand point distributions was
rejected because falta envido makes a hand's points depend on the score. Tests use an independent exact DP of a toy
chain as truth. Brier comparison: equity versus `p = 1/2 + (own - opp) / (2 * target)`.

## Throughput (same machine, 12 workers, `bench_phase2_speed.py round --rounds 5`, run 2 champion widened with zero columns)

| Encoder | matches/s, two runs | mean | vs v1 |
|---|---|---:|---:|
| v1 (1628 inputs) | 438.0, 426.8 | 432.4 | |
| v2 (no input 6) | 422.8, 415.0 | 418.9 | -3.1% |
| v3 (with input 6) | 409.9, 401.4 | 405.7 | -6.2% |

Input 6 costs about 3 points on top of inputs 1-5, and the whole change is 6.2% (< 10%), so **version 3 is the run 3
encoder**. The posterior sums are memoised per posterior state, which keeps input 6 cheap. The blended target and the
greedy-disagreement counting were within noise (v3, 369 / 375 matches/s without and 370 / 398 with `--target-blend .5`;
background load differs between runs). Raw files: `docs/perf/run3-encoder/`.

Widened networks are bit-identical: 3,214 decisions of the real run 2 champion (30 matches, versions 2 and 3,
greedy) produce identical Q bytes and actions; unit tests check batches and single decisions too.

## Seeds (audited against `experiments/*/RULE.txt`, the final exam plans and the earlier-namespace tables)

| Use | Seed |
|---|---|
| Equity table build / held-out check | 67001 / 67002 |
| Pilot training | 63001 |
| Pilot evaluation | deal base 92001, action exponent 132, 300 pairs vs Adaptive Utility |
| Run 3 training | 65001 |
| Run 3 checks | 66000 to 66999 |

None appears in any RULE.txt, README, log or earlier namespace; `phase2_run3_pilot.audit_seeds()` checks this in
code and a test fails if the pilot plan overlaps the exam or earlier ranges. The final exam for run 3 needs its own
fresh seeds (for example 93001 upward, exponent 133).

## Commands (proposed, not run)

```powershell
# 1. Equity table from the shipped network, greedy self-play (about 200k matches), then a held-out check
py proyecto/phase2_equity.py build --weights champions/phase2/reigning_champion.pt --matches 200000 --seed 67001 --workers 12 --output experiments/phase2-run3/equity-table.json
py proyecto/phase2_equity.py check --table experiments/phase2-run3/equity-table.json --weights champions/phase2/reigning_champion.pt --matches 20000 --seed 67002 --workers 12

# 2. Pilot: about 1,000 updates, then 300 greedy mirror pairs vs Adaptive Utility; exit code 3 below 68%
py proyecto/phase2_run3_pilot.py --warm-start champions/phase2/reigning_champion.pt --equity-table experiments/phase2-run3/equity-table.json --output training_output/phase2-run3-pilot --workers 12

# 3. Run 3 (M = generation_sizing.matches_per_generation from pilot.json, about 7.5 minute generations)
py proyecto/phase2_trainer.py --output training_output/phase2-run3 --champions-dir champions/phase2-run3 --generations 1000 --matches-per-generation M --matches-per-round 240 --workers 12 --seed 65001 --check-seed-base 66000 --target mixed --lr 0.0003 --minibatch 512 --temperature 0.05 --profile-prior 5 --patience 40 --time-cap-seconds 43200 --panel-path experiments/phase2-run2/panel.json --no-panel-rotation --held-out-manifest experiments/phase2-run2/held-out.json --league-shares experiments/phase2-run2/league-shares.json --timing-log --panel-pairs 300 --head-to-head-pairs 300 --pool-recent 8 --pool-diverse 4 --encoder-version 3 --warm-start champions/phase2/reigning_champion.pt --target-blend 0.5 --equity-table experiments/phase2-run3/equity-table.json --check-selection greedy

# If the first 2-3 generations take over 10 minutes: resume once from a generation-boundary checkpoint
py proyecto/phase2_trainer.py <the same flags> --matches-per-generation M2 --resume training_output/phase2-run3/checkpoint_NNNNNN.pt --change-matches-per-generation
```

Before step 3, the table must pass: `check` prints `monotone: true` and `brier.equity_better: true`.
The change is recorded in the checkpoint (`generation_schedule`, config `generation_size_change`), in
`generation_size_change.json` and on the first generation entry of the new size; a second change is refused.

## Open doubts

- Greedy checks make phase 2 panel members (run 1 / run 2 networks) greedy too, not only challenger and champion;
  phase 1 members and the references play as before. Say so if only the two candidates should be greedy.
- The pilot picks the number of rounds from an assumed 35 learner samples per match (run 2 late value); the report
  gives the actual update count. Matches per second is end to end (actors and learner), without a check; the
  generation size subtracts a 5% check share (ADR amendment), which should be confirmed on the first real generation.
- `--target-blend` is the weight of the equity part (0 = old target); ADR 0062's 0.5 / 0.5 is `--target-blend 0.5`.
- Equity is built only from self-play of the warm-start network, as in the ADR; it is not conditioned on opponent type.
- Snapshot records in the checkpoint still label phase 2 networks `encoder: v1` (the panel validator accepts only
  `phase2` or `v1` there); the real encoder version is in the weights' shape and the config.
- Resume of an old (pre-run 3) checkpoint with new code works (config keys are only added when set); a run 3
  checkpoint cannot be resumed without the same run 3 flags.

## Coordinator additions before running (advisor review, 2026-11-10)

Fixed before any real run:
- Pilot pass = the tool's `passes` (pilot greedy win rate vs fixed Adaptive Utility >= 68%) AND the paired difference pilot minus warm start (same 300 pairs, both in `evaluation`) >= -0.03.
- After the table check: report visit counts and spot cells (0-0 mano near 0.5-0.55; 14-0 at target 15 near 1).
- In the pilot entry, `greedy_disagreement` must be non-null; report it and the mean and SD of the blended targets against the plain match results.
