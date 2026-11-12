# Phase 2 optimization step 4: reuse unchanged summary arrays

Ticket #35, chunk 5. Kept: reuse derived posterior arrays until evidence or weights
change. The original ordered array builder and every arithmetic expression remain
unchanged. All original golden fixtures remain frozen.

## Experiment and invalidation

The previous actor profile spent 0.406 of 1.150 seconds in `_summary_arrays`.
`Inference` now retains that derived tuple. `declared_tantos`, `son_buenas`, a new
played/excluded-card filter, and every valid `truco_call` invalidate it. The existing
idempotent evidence fast path leaves it intact. Every newly constructed inference
starts empty; incremental resets already construct a new inference. Available own
card slots and trick tie rules still enter the unchanged per-decision calculations
after this cache, so they are never cached accidentally.

The cache is per inference object, not global. No new persistent/checkpoint state,
posterior order, total-weight summation, precision, model, RNG or sampling changes.
Tests compare raw arrays and scalar summaries with forced rebuilding after every
mutator, including repeated/new evidence, changing weights, all three cards played,
and a contradictory exclusion yielding an empty posterior.

Same four-match, 206-decision cProfile pass: **1.150 -> 1.038 s**, with summary-array
time **0.406 -> 0.296 s**. `numpy.fromiter` calls fall from 824 to 604; profile call
count falls from 3,704,604 to 3,199,108. The kept speedup is bounded, as planned.

## Fixed full benchmark

Same 12-worker, seed-35, two 96-match rounds, mixed targets, league/progress on,
v0 measuring panel, full 100/200-pair check and 400-situation validation protocol.
Runs are sequential and use the unchanged 175-second supervised harness.

| Measurement | Off step 3 | Off step 4 | On step 3 | On step 4 |
|---|---:|---:|---:|---:|
| Total wall, seconds | 21.032 | 19.738 | 42.333 | 39.386 |
| Training matches/s | 9.129 | 9.727 | 4.535 | 4.875 |
| Learner samples/s | 183.578 | 195.609 | 91.064 | 97.877 |
| Full check, seconds | 14.214 | 13.212 | 14.238 | 12.733 |
| Cold validation, seconds | disabled | disabled | 18.446 | 17.353 |
| Warm validation, seconds | disabled | disabled | 1.831 | 1.859 |
| Learner seconds/update | 0.01062 | 0.01078 | 0.11572 | 0.11582 |

Wall time falls **6.2% head off / 7.0% head on** in this step. Warm validation and
learning are unchanged within variation; actors, checks and cold fixed replay
benefit. Samples/updates stay 3,861/8 off and 3,855/8 on. All model, optimizer,
semantic-log, checkpoint and snapshot hashes in both full benchmarks equal the
original pre-optimization baseline.

The original 12-case golden matrix is recorded under `phase2-step4/equal-*.json`.
Phase 1 encodings/training checkpoints and permanent suite guards are checked again
before integration. Items 5–7 are deferred: IDs are already cached, warm validation
is under two seconds, warm transfer residual is milliseconds, and actor forward/event
costs are small; the detailed reasoning remains in [phase2-step3.md](phase2-step3.md).
The final report records integration and full-suite results.

Pre-integration gates passed: 12/12 original golden comparisons, both original full
benchmark hash sets, the summary-cache invalidation test (0.049 s), and all five
`*golden*.py` suite tests (62.939 s). The latter includes the new permanent head-off/on
fresh-process guard plus every existing phase 1 encoding/training golden.
