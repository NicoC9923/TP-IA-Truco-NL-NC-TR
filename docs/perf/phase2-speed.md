# Phase 2 speed work for run 3

Goal: more training matches per hour in run 3 (12 CPU workers, ADR 0044) without
changing anything the trainer computes. Baseline revision `d406e3e` (= `0c017e8`
plus the golden harness). Branch `hllous/p2-speedup`.

## Fidelity rule and golden check

An optimization is kept only if a fixed-seed run produces exactly the same
results: matches and winners, learner samples (input bytes, actions, targets,
seats), model and Adam tensors, every checkpoint, the generation log, the
generation snapshots and the exported champion. Only timing fields
(`elapsed`, `elapsed_seconds`, `training_seconds`, `check_seconds`,
`generation_training_seconds`) and resolved paths are excluded. Floats are compared
by `float.hex`, tensors and arrays by raw bytes.

`proyecto/golden_phase2_speed.py` runs three generations of 96 matches (rounds of
48, blocks of 4, minibatch 256) with the **run 2 settings**: run 2 league shares,
the six-member run 2 panel (two checks, 2 pairs per member and 2 direct pairs),
held-out manifest, temperature 0.05, profile prior 5, no panel rotation, timing
log, auxiliary head off. 5,579 learner samples and 25 updates. The reference
`docs/perf/phase2-speed/golden.json` was recorded with 1 worker on the baseline
code. Workers 12 matched it before any change. Every commit below re-ran:

```powershell
py proyecto/golden_phase2_speed.py compare --workers 1  --reference docs/perf/phase2-speed/golden.json
py proyecto/golden_phase2_speed.py compare --workers 12 --reference docs/perf/phase2-speed/golden.json
```

Both print `GOLDEN OK` on the final code (workers 3 also checked).
`proyecto/test_phase2_speed_golden.py` runs it with 2 workers in the suite. To show
the check can go red, a 1e-6 change to the profile confidence inputs fails it on
model, optimizer, samples, checkpoints, snapshots and champion. The existing
pre-optimization guard `test_phase2_optimization_golden.py` (both heads, resume)
also still passes.

## Profile: where the time went

Hardware: Windows 11, AMD Family 26 (8 cores / 16 threads), Python 3.12.10, torch
2.15.0+cpu, one torch thread per process. Background CPU load 5–9% before each
measurement; measurements were run one at a time.

Run 2 itself: training was ≈95% of the time (≈820 s per generation versus ≈39 s per
check). Late in the run there were ≈35 learner samples per match.

Actor profile, 40 league matches with the run 2 champion as learner (one process,
`bench_phase2_speed.py profile`):

| Component | Baseline | Final |
|---|---:|---:|
| Whole chunk | 6.23 s | 0.93 s |
| Phase 2 encoding | 5.63 s | 0.51 s |
| of which posterior work (filters, arrays, summaries) | ≈5.0 s | ≈0.39 s |
| Network forward (policy) | 0.17 s | 0.13 s |
| Engine and the rest | ≈0.4 s | ≈0.3 s |

At 12 workers the round wall time was the slowest job plus ≈15 ms. Weight
pickling (1.8 MB, ≈1 ms) and sample transfer (≈64 MB per round, unpickled while
the other workers still run) were small. The idle time came from load imbalance:
the slowest job took 0.72 s against a 0.60 s mean. In a real `train()` run the
learner is ≈13% of wall time (forward 3.8 ms, backward 4.3 ms, Adam 1.3 ms,
batching 0.9 ms per 512-sample update) and checkpoint saving plus fsync ≈2%.
Raw profiles: `phase2-speed/actor-profile-*.txt`.

## Kept optimizations (one commit each, golden equal at workers 1 and 12)

| Commit | Change | Actor profile (40 matches) |
|---|---|---:|
| `0ce5d2b` | Opponent posterior as ordered numpy arrays (ids + float64 weights) instead of a dict of hands | 6.23 → 1.96 s |
| `24fa02f` | Reuse the exact v3 summaries when the weighted posterior is identical (no opponent truco call yet) | 1.96 → 1.46 s |
| `b017d47` | Per-hand lookup tables for ranks, brava, strength and the int64 slot sign sums | 1.46 → 1.06 s |
| `cd6a502` | One task per 4-match block (dynamic scheduling, network blocks first) | round: 306 → 354 matches/s |
| `bbdf2fc` | Unit-weight total = float(n); memoized posterior sums; direct linear/ReLU calls for a plain network | 1.06 → 0.93 s |
| `6fea3a0` | One check network per weights object per chunk instead of one per pair | check only |

Why each one is exact:

- **Arrays.** The dict kept hands in ascending hand id, and every filter kept that
  order. The arrays use the same order. Filters are membership masks. The truco update is
  the same elementwise IEEE expression, so it rounds the same per hand. Totals still use Python
  `sum` over the same floats in the same order. Python 3.12 uses compensated summation for
  floats, while `np.float64` objects (from a NumPy bluff rate) were added naively;
  `_scalars` keeps that case too. Other NumPy scalar rates now raise instead of
  promoting differently. No production caller passes one; all rates come from
  `OpponentProfile.rates()` as Python floats. `test_inference_arrays.py` compares a frozen
  copy of the dict implementation over 250 random evidence sequences: weights,
  distribution, summary arrays (bytes and dtype), `summaries` and `v3_summaries`.
- **Summary reuse.** Values 50:55 and 1623:1628 depend only on the posterior and
  the observation. Identity is checked from the arrays (`same_posterior`), and the
  weighted posterior gets the same update `encode_v3` made.
  `test_phase2_encoder_reuse.py` compares a frozen copy of the old encoder byte
  for byte over 30 random matches (with and without incremental inference or
  weighting).
- **Lookup tables.** Same int64 values and dtypes. The float expressions and the
  sequential sums are unchanged.
- **Scheduling.** Each block seeds all its policies and matches from its match ids
  and builds fresh policies, and results are merged sorted by match id. Splitting
  and ordering therefore cannot change anything, the same argument that already
  makes 1 and 12 workers equal.
- **Unit total.** A compensated sum of n copies of 1.0 is exactly n.
  **Memoized sums** use the same expression once per posterior state.
  **Direct forward**: `nn.Linear`/`nn.ReLU` call these same kernels. Subclasses and
  hooked models still go through `model()` (`test_phase2_fast_paths.py`).
- **Check networks.** Policies only read their network.

## Measured result

The same benchmark was run on the baseline tree and on the final code, one after
the other. Rounds of 240 matches use the run 2 league shares, style offsets fully
ramped, the run 2 champion as learner and snapshot, and 12 workers; after one
warm-up round, 5 timed rounds follow. The check uses the run 2 panel with 100 pairs per
member and 100 direct pairs (`bench_phase2_speed.py round --rounds 5 --check-pairs 100`).

| Measurement | Baseline | Final | Speedup |
|---|---:|---:|---:|
| Training matches/s (actor rounds) | 53.6 | 392.0 | 7.3× |
| Learner samples/s (actor rounds) | 2,280 | 16,683 | 7.3× |
| Mean round, 240 matches | 4.48 s | 0.61 s | 7.3× |
| Check, 6 members × 2 × 100 + 100 direct pairs | 42.7 s | 7.7 s | 5.6× |
| Learner update, 512 samples | 10.6 ms | 11.0 ms | unchanged |
| Pool warm-up round | 6.5 s | 2.7 s | |

Sample digests per round and the check result are identical. End to end, a real
`train()` from a fresh network with run 2 settings, 12 workers, 2 × 2,400 matches
and one 20-pair check ran at **110.0 → 450.9 matches/s (4.1×)**, with 73,729 samples
and 153 updates in both. Fresh networks play shorter matches, so the fixed learner
cost is a larger share there. Run 2's late matches look more like the champion
benchmark. With the learner now a larger share of each round, expect between 4× and 7× more
matches per hour in run 3. This is a local measurement, not a promise for a long overnight run.

Worker count, final code: 8 → 329.5, 12 → 422.1, 14 → 415.1, 16 → 423.6 matches/s.
More than 12 workers gives nothing on this CPU, so 12 stays. Results do not depend
on the worker count.

Run 2's generation size (57,120 matches, ADR 0044) was chosen for 5–10 minute
generations at the old speed. At the new speed a generation takes roughly a fifth of
the time, and the check share of each generation stays similar because checks sped up too. Whether
to change the generation size for run 3 is a separate decision; nothing here changes it.

## Rejected ideas

| Idea | Why rejected |
|---|---|
| More torch intra-op threads for the learner (2–8) | Not bit-identical: with batch sizes such as 7 and the auxiliary head, 2 and 8 threads change gradients (4 matched only by chance). It would also contradict ADR 0042's fixed thread counts. |
| Caching the weights inside workers (no 1.8 MB per task) | Measured, no gain (within noise). |
| Sparse or stacked sample transfer | Unpickling overlaps with the other workers still playing; ≈15 ms of each round is outside the slowest job. Not worth the complexity. |
| Batched or vectorized inference across decisions | Decisions are sequential within a match, and batching changes GEMM shapes and therefore rounding. |
| `np.add.reduce` / `math.fsum` / pairwise sums for posterior summaries | Changes summation order; the left-to-right `cumsum` is kept. |
| Pipelining actors with the learner (stale weights) | Changes what is computed (ADR 0042 synchronous rounds). |
| Adam `foreach` or fused implementations | Different kernels with no guarantee of equal rounding; Adam is only ≈1.3 ms of an 11 ms update. |
| More than 12 workers | Measured: no throughput gain (above). |

## Tests

Full suite `py -m unittest discover -s proyecto -p "test*.py"`: 499 tests OK (237 s). Phase 2 files `test_phase2_*.py`: 214 OK; `test_evolution_phase2.py`: 19 OK.

## Reproduction

```powershell
py proyecto/bench_phase2_speed.py profile --weights <run2 reigning_champion.pt> --matches 40 --output <dir>/profile
py proyecto/bench_phase2_speed.py round --weights <run2 reigning_champion.pt> --rounds 5 --check-pairs 100 --output <dir>/round.json
```

For the baseline, run the same commands from an export of `d406e3e`
(`git archive d406e3e proyecto experiments`) with the current
`bench_phase2_speed.py` copied in. The weights file is only read.
