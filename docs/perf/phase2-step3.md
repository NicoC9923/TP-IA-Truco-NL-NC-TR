# Phase 2 optimization step 3: boolean masks and exact combination scoring

Ticket #35, chunk 4. Kept: precomputed boolean membership and faster combination
scoring with the original gradient accumulation order. This step reduces full-run
head-on learner time **59.3%**, with identical original semantic hashes. The fixed
benchmark protocol remains in [phase2-baseline.md](phase2-baseline.md); raw timing,
operation profiles and equality evidence are in `phase2-step3/`.

## Boolean membership

The fixed 40-card by 9,880-hand boolean membership table replaces each repeated
`(_IDS == card).any(axis=1)` scan in `compatible_mask`. It occupies 395,200 bytes
per process. Evidence application order, exact tantos constraints, hand order,
boolean output and all floating-point operations are unchanged. The independent
reference checks 512 varied evidence rows and empty input against the original
implementation. Actual true-label compatibility is also exercised by training.

## Remaining auxiliary loss: operation profile

Reproduce with the same 175-second supervised runner:

```powershell
python proyecto/bench_phase2.py auxiliary --output training_output/perf35/auxiliary.json
```

This collects four real seed-35 mixed-target self-play matches, repeats their 206
decisions to form a fixed 512-sample minibatch, and performs eight Adam updates per
variant from identical initial weights and optimizer state. Each variant compares
every loss, every parameter-gradient byte hash, final model and optimizer hashes
with the original scoring path. Hashing is outside timed operations. Input/label
preparation is outside this microprofile; the end-to-end benchmark includes it.

Mean operation times, milliseconds/update:

| Operation | Old masks + old scores | New masks + old scores | New masks + exact fast scores |
|---|---:|---:|---:|
| Q forward + MSE | 3.94 | 3.99 | 4.10 |
| Compatible masks | 135.68 | 3.92 | 3.88 |
| Auxiliary forward | 4.15 | 4.06 | 4.15 |
| 512 x 9,880 combination scores | 95.82 | 96.47 | 16.93 |
| Label check + masked fill | 5.38 | 5.28 | 5.86 |
| Cross entropy | 3.54 | 3.30 | 2.62 |
| Zero gradients + backward | 49.43 | 50.18 | 73.89 |
| Adam | 1.50 | 1.67 | 1.53 |
| Total measured operations | **299.44** | **168.86** | **112.95** |

The mask lookup removes about 132 ms/update. After that, the expensive operation
is materializing `logits[:, hand_ids]` as a batch x 9,880 x 3 tensor and reducing
its last dimension. Cross entropy itself is only about 3 ms. Reusing trunk outputs
would save little relative to scoring and cannot change gradient rounding.

## Exact faster scoring

`combination_scores` gathers each card column separately and adds in the same order
as the original three-element CPU reduction, starting at positive zero. The initial
zero matters: three negative zeros otherwise produce different result bits.

Ordinary autograd for the three gathers accumulates duplicated-card gradients in
a different order. A small custom autograd function therefore uses the original
gather backward's row/hand/card order: broadcast the incoming score gradient over
the three cards and accumulate with `index_put_(..., accumulate=True)` into a zeroed
batch x 40 gradient. It retains the two original independent trunk forwards and
the original loss, mask, cross-entropy, backward and optimizer call order.

Backward is slower by about 24 ms in the microprofile, but scoring saves about
80 ms, so net update work still falls another 33.1% after the mask change. No
precision, model topology, targets, summation order, training RNG or sampling changes.
Exactness is checked for the agreed CPU/machine/library configuration (ADR 0042).

Byte-level oracle tests cover float32/float64, batches 1/7/32/512, all hands and a
noncontiguous hand subset, negative-zero inputs and arbitrary upstream gradients.
They compare raw forward and gradient bytes with the original expression. The
eight-update profile additionally compares the complete model/optimizer trajectory.

## Measured and rejected candidates

| Candidate, using new masks | Total ms/update | Exact trajectory? | Decision |
|---|---:|---|---|
| Original advanced indexing | 168.86 | Reference | Reference only |
| Flattened `index_select` + reshape | 165.96 | Yes in this probe | Drop: only 1.7% net microprofile gain; inferior to kept path |
| Separate gathers, ordinary autograd | 76.12 | **No** | Reject: first scalar loss matches but first gradients differ |
| Share the Q/auxiliary trunk forward | 160.67 | **No** | Reject: first scalar loss matches but first gradients differ |
| Separate gathers, original accumulation order | 112.95 | Yes | Keep; all original golden gates pass |

The two float-changing candidates exist only as benchmark counterexamples; neither
is wired into production. They require explicit approval before any deployment and
are not proposed as part of this exactness-preserving change.

## Fixed full benchmark

Same sequential 12-worker, two 96-match rounds, mixed targets, league/progress on,
measuring panel v0, full 100/200-pair check and 400-situation validation as step 2.

| Measurement | Off step 2 | Off step 3 | On step 2 | On step 3 |
|---|---:|---:|---:|---:|
| Total wall, seconds | 21.494 | 21.032 | 43.609 | 42.333 |
| Training matches/s | 8.933 | 9.129 | 4.403 | 4.535 |
| Learner samples/s | 179.628 | 183.578 | 88.400 | 91.064 |
| Learner seconds/update | 0.01144 | 0.01062 | 0.28459 | **0.11572** |
| Full check, seconds | 14.312 | 14.214 | 14.363 | 14.238 |
| Cold validation, seconds | disabled | disabled | 18.338 | 18.446 |
| Warm validation, seconds | disabled | disabled | 1.826 | 1.831 |
| Samples / updates | 3,861 / 8 | 3,861 / 8 | 3,855 / 8 | 3,855 / 8 |

Head-on whole-run time falls 2.9%; learner work falls 59.3%, saving about 1.35 seconds
over eight updates, consistent with the 1.28-second whole-run reduction. This
deliberately check-heavy fixture spends little time learning; a generation with more
training matches would give learner savings a larger share. Head-off differences
are run variation: this chunk changes no head-off learning work.

Against the original baseline, wall time is now **29.212 -> 21.032 s** head off
and **67.342 -> 42.333 s** head on (37.1% less time head on). Both full benchmark
hash sets match the original baseline in every field. Timing probes and their
interpretation are unchanged, including serialization overhead and wait residuals.

## Equality gates

All 12 comparisons against the original chunk-1 goldens passed: head off/on, workers
1/12, uninterrupted runs, fresh-process partial-generation resumes, and fresh-process
generation-boundary resumes with a cold rebuilt validation cache. Original fixtures
are unchanged. Both full benchmark hash sets also match the original full benchmark
records, including logs, checkpoints, optimizer and snapshot bytes.

Checks passed:

- `test_phase2_combination_scores.py`: 2 tests, 0.977 s.
- `test_phase2_auxiliary.py`: 6 tests, 1.437 s.
- All `*golden*.py` suite tests: 4 tests, 38.893 s, including all six phase 1
  encoder/worker training comparisons and their checkpoint contents.
- Python compilation and staged whitespace checks pass. The longest timing run is
  42.333 s; all CPU runs are under three minutes and none remain running.

## Recommendation on plan items 4–7

- **Item 4, summary-array reuse:** worth one bounded experiment. The last actor
  profile spends 0.406 of 1.150 seconds constructing summary arrays, while checks
  still take about 14.2 seconds. Preserve posterior order and invalidate on every
  evidence/weight/played-card change; actual savings need measurement.
- **Item 5, validation IDs/marginals:** fixed hand IDs are already cached in step 2.
  Only the learned metric scan remains each call, and all warm validation now costs
  1.83 seconds. Defer further marginal work until after item 4; any gain is bounded
  by that small total and float/tie order must remain exact.
- **Item 6, IPC:** defer. Warm map residual is only 3.24/3.38 ms off/on, with the
  original serialization caveats. Avoid added transfer machinery without evidence.
- **Item 7, events/batched forwards:** defer. Forward was about 1.3% of the actor
  profile and event work was small; changing batching/RNG order is not justified.

After an item-4 experiment, prefer final integration and the full-suite gate unless
a fresh profile reveals a larger exact-preserving opportunity.
