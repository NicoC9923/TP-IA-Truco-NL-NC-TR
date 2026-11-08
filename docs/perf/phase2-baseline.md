# Phase 2 throughput baseline and optimization plan

Ticket #35, chunk 1. Baseline production revision: `7ed6b76078e7cb8c1527200ed0c9a7f3b732cda5`.
No production code, defaults, sampling, checks or arithmetic changed in this chunk.
The companion `phase2-baseline/` directory contains the machine-readable measurements,
pre-optimization goldens and cumulative/internal cProfile reports.

## Reproduce

Run these commands from the repository root, on the same machine and library versions:

```powershell
python proyecto/bench_phase2.py bench --output training_output/perf35/off.json
python proyecto/bench_phase2.py bench --auxiliary --output training_output/perf35/on.json
python proyecto/bench_phase2.py actor --output training_output/perf35/actor.json
python proyecto/bench_phase2.py validation --output training_output/perf35/validation.json
python proyecto/bench_phase2.py bench --auxiliary --profile --pairs 12 --direct-pairs 24 --output training_output/perf35/learner.json
```

Each invocation supervises a fresh process, allows at most 175 seconds, and on Windows
terminates that process and its spawned process tree on timeout. Timing runs are sequential;
no training/profiling jobs overlap. Temporary training output stays on the repository's
drive (Windows cannot relativize panel paths between C: and E:) and is removed on success.
Profiles produce `.pstats` for local inspection and `.profile.txt` for review; the text
reports are committed. This is not a statistically replicated machine benchmark.

Machine: Windows 11, AMD64 Family 26 Model 68 Stepping 0, 16 logical CPUs, 12 actor
processes, one torch/BLAS thread per process, Python 3.12.10, torch 2.15.0+cpu.
Fixed benchmark: seed 35, two generations with one 96-match round each, mixed targets
15/30, 30% random starts, default league shares, four matches per league block,
minibatch 512, measuring panel v0, 100 mirrored pairs per member per candidate and
200 direct pairs. The first generation establishes the champion; the second checks
it (800 mirror pairs / 1,600 check matches in total). Progress is enabled; champion
export is disabled to avoid touching the repository's champion. Head-on validation
uses all 400 fixed situations each generation. Both configurations make eight updates.

The reduced check in the parent cProfile command keeps profiling short and does not
change the learner workload. It is not the throughput baseline. Initial exploratory
12/24-pair checks measured 16.77 matches/s off and 3.85 on; do not compare these
with the retained full-check baseline.

## Retained throughput baseline

| Measurement | Head off | Head on |
|---|---:|---:|
| End-to-end wall time (192 training matches) | 29.212 s | 67.342 s |
| Training matches / total second | 6.573 | 2.851 |
| Learner samples / total second | 132.170 | 57.245 |
| Samples / updates | 3,861 / 8 | 3,855 / 8 |
| Actor-only matches/s (including cold map wait) | 30.355 | 29.072 |
| Learner seconds/update (mean) | 0.01064 | 0.29744 |
| Full selection check | 21.064 s | 20.667 s |
| Validation, cold / warm | disabled | 24.652 / 11.236 s |
| Entire round 1 / round 2 | 4.481 / 23.123 s | 30.497 / 35.193 s |

| Round component, seconds | Off round 1 | Off round 2 | On round 1 | On round 2 |
|---|---:|---:|---:|---:|
| Actors + serialization probe | 4.41289 | 1.96957 | 4.57630 | 2.10356 |
| Merge | 0.00051 | 0.00197 | 0.00054 | 0.00198 |
| Learner | 0.04438 | 0.04071 | 1.24235 | 1.13714 |
| Boundary (check / metrics / persistence) | 0.02049 | 21.08616 | 24.67555 | 31.92604 |

Warm actor compute critical path: 1.93858 s off / 2.06459 s on; warm map wait:
1.94169 / 2.06807 s; non-overlapped IPC/scheduling residual: **3.11 / 3.48 ms**.
Cold map residual is 2.13 / 2.23 s, dominated by spawn/import startup. Ordinary
pickle probes per training round total 27–29 ms off / 35–39 ms on for dumps+loads.
Independent job payload proxies total 21.7 / 22.3 MB, results about 12.6–12.8 MB.
The full check's serialization probe adds 0.163 s off / 0.201 s on; actual check
map waits are 20.892 / 20.458 s. Check scheduling residual is not pure IPC.

The head adds about 38.13 s to this fixed run: 35.89 s is validation and about
2.29 s is extra learner work. Actor throughput is essentially unchanged. A full
check dominates the head-off run; optimizing decisions helps both training and
checks. With 96 matches/generation this is deliberately a check-heavy throughput
fixture, not a recommendation for the later measuring run's generation length.

## Timing interpretation

`RoundProbe` traces only the `run_rounds` frame, locating phase boundaries by source
markers. Actors includes map wait and a separate pickle probe; merge covers result
flattening, ordering and counting; learner includes minibatch preparation, masks,
forward, backward and optimizer steps; boundary includes checks, metrics, checkpoint
and log writes. Setup/progress are separate. Run wall time additionally includes
network/optimizer construction, pool setup/teardown and initial manifest loading.

The pool wrapper preserves map ordering and measures each actor job's compute time.
`wall - actor_max` is a **wait residual, not an IPC measurement**: cold residual also
contains process startup, and check jobs run in multiple waves, so their residual
contains scheduling and compute. For the warm training round there is one job per
worker; that residual is a useful upper bound on non-overlapped transfer/scheduling.
Serialization is separately probed by dumping/loading each job and result independently
(no cross-job memoization). Byte counts are ordinary pickle payload proxies, not
actual shared-memory multiprocessing wire sizes. These probes add overhead to the
reported round/check/wall times; their durations are recorded, and must remain enabled
in final comparisons. Serialization and actor compute overlap in production; do not
add these probe times to map wait a second time.

## Pre-optimization equality harness

The frozen golden configuration uses seed 35, mixed targets, the real league and
measuring panel, four rounds of eight matches, two generations of 16 matches,
minibatch 32, eight validation situations, and a small complete 2/2-pair check.
Two league blocks per round exercise independent actor jobs; 26 updates exercise
optimizer state. This intentionally smaller fixture runs comfortably under the cap.
It collects 754 learner samples without the head and 757 with it.

For each head setting, `golden-off.json` / `golden-on.json` were recorded with one
worker before optimization. The harness compares typed, unrounded tensor/array byte
hashes, semantic log, all retained checkpoints (including optimizer, counters,
league/panel history and RNG states), and raw generation snapshot file bytes.
Only elapsed time, elapsed_seconds and resolved filesystem locations are normalized;
portable panel paths are represented by basename with their SHA identities retained.
No model float, loss, validation metric, action, promotion decision or RNG state is
discarded. This is semantic checkpoint equality, not equality of serialization
container metadata. No expected hashes should be regenerated after an optimization.

```powershell
python proyecto/bench_phase2.py golden --workers 12 --resume --reference docs/perf/phase2-baseline/golden-off.json --output training_output/perf35/equal-off.json
python proyecto/bench_phase2.py golden --workers 12 --auxiliary --resume --reference docs/perf/phase2-baseline/golden-on.json --output training_output/perf35/equal-on.json
```

The reference fixture is uninterrupted with one worker. `--resume` stops a separate
process after round one, before the first generation finishes, and resumes from
disk in a fresh process in the same directory; no actor or inference caches survive.
The checked-in `equal-*-resume-12.json` results match their reference in every field.
For later chunks also run the cross-product of head off/on, workers 1/12, and
uninterrupted/resumed by omitting `--resume` or selecting `--workers 1` as appropriate.
The benchmark's larger run also records hashes for both head settings.

All eight comparisons in that cross-product passed using the original reference
fixtures, including fresh-process partial-generation resumes. Every `equal-*.json`
file records its worker count and resume/head setting. Existing phase 1 guards also
passed: `test_phase1_golden.py` (1 test, 0.786 s) and
`test_phase1_extraction_golden.py` (1 test, 0.560 s). Python compilation and staged
diff whitespace checks passed. No full-suite claim is made in this baseline chunk.

## Actor profile

Four single-process self-play matches, seed 35, mixed targets, 206 decisions:
2.027 s under cProfile / 7,502,276 calls. These are instrumented costs, not
unprofiled per-decision latency estimates; cumulative rows overlap.

| Component | Cumulative seconds | Per decision |
|---|---:|---:|
| Entire league block | 2.027 | 9.84 ms |
| Phase 2 encode | 1.955 | 9.49 ms |
| Incremental posterior update (two beliefs) | 1.409 | 6.84 ms |
| `seen_cards` filtering | 1.119 | 5.43 ms |
| `v3_summaries` | 0.520 | 2.52 ms |
| `_summary_arrays` | 0.415 | 2.01 ms |
| Network forward | 0.027 | 0.13 ms |
| Engine outside policy calls | about 0.017 | 0.08 ms |

Filtering repeatedly performs millions of Python set intersections/subset checks,
including calls with unchanged evidence. Encoding accounts for 96.4% of profiled
block time; forward accounts for 1.3%. Event encoding is not a first priority.

## Validation profile

400 mixed-target situations, cold then cached: **28.127 / 11.116 s under cProfile**,
39.242 s combined / 98,437,665 calls. The cold fixed replay alone takes 16.907 s,
56 matches and 1,674 decisions. Across both passes:

| Component | Calls | Cumulative seconds |
|---|---:|---:|
| Exact posterior reconstruction | 2,400 | 10.258 |
| `prediction_metrics` | 2,400 | 7.591 |
| Learned `card_distribution` | 800 | 6.568 |
| All `seen_cards` calls, including replay | 5,748 | 18.017 |
| Python hand-ID tensor construction | 1,040 | 1.278 |

Every situation rebuilds the same exact posterior three times: once explicitly,
once inside weighted inference, and once inside learned `card_distribution`.
Exact/weighted distributions and their scores depend on the fixed validation
situations, not on the learner. Even the learned candidate hand list is static.
The Python metric loop scans candidate hands and adds card marginals in a defined
order. Replacing this with a generic floating reduction risks changing both metric
floats and tie rankings; caching existing results is safer.

## Learner profile

The parent-process cProfile pass uses the same 96-match rounds and eight updates,
with a shorter 12/24-pair check to bound profiling time. Learner round times are
1.245 / 1.127 s (0.2966 s/update). The report includes lazy torch initialization,
validation and pool management; use the explicitly filtered learner rows, rather
than interpreting the whole report as learner time. The standalone actor/validation
profiles avoid this broader initialization noise.

| Learner operation | Calls | Cumulative seconds | Mean/update |
|---|---:|---:|---:|
| Compatible-hand masks | 8 | 1.098 | 137.3 ms |
| Backward | 8 | 0.382 | 47.7 ms |
| Cross entropy | 8 | 0.024 | 3.0 ms |
| Adam step | 8 | 0.015 | 1.9 ms |

Masks consume about 46% of learner time, while backward and Adam are much smaller.
Remaining learner time includes tensor/index preparation and scoring the 9,880
hands. Precomputing integer/boolean membership columns could save about 0.8–1.0 s
over this two-round run (roughly 0.1 s/update) without touching float arithmetic.

## Ranked plan for following chunks

Estimates below are planning ranges based on the profile, not measured improvements.
They overlap and must not be added. Keep one change per chunk, remeasure the fixed
full benchmark and pass the original goldens in the entire worker/resume matrix.

| Order | Candidate | Expected saving | Bit-identity risk / gate |
|---|---|---|---|
| 1 | Skip repeated evidence filtering when no excluded/new opponent cards exist, retaining the ordered posterior | 20–40% of actor/check compute; roughly 4–8 s/check and several seconds of cold validation | Low: only skip an idempotent dict filter; preserve played-card state and all reset cases; shared inference requires phase 1 goldens |
| 2 | Cache fixed validation preprocessing: exact/weighted metric results and compatible learned hand IDs; reuse original scalar arithmetic | Roughly 5–8 s per warm 400-situation validation; first-call benefit smaller | Low/medium: key every schedule/target/temperature/count input; never cache learned outputs; resumed cold path must match cached path |
| 3 | Precompute boolean card-membership columns for `compatible_mask` | About 0.8–1.0 s over eight auxiliary updates (roughly 0.1 s/update) | Low: boolean operations and hand order only; no float changes; retain every exact-evidence case |
| 4 | Reduce repeated ordered hand-ID/summary-array construction per unchanged posterior | Around 10–20% actor/check compute after earlier changes | Medium: invalidate after weights, cards and evidence change; preserve Python total-weight sum, operation order and ordered cumulative sums |
| 5 | Cache immutable learned hand-index tensors / optimize validation marginals with order-preserving additions | About 0.5–1 s per validation for IDs; metric loop ceiling about 3.8 s/pass | Medium/high for marginals: identical summation order and tie ordering required; reject changed floats |
| 6 | Compact sample transfer / avoid repeatedly packaging immutable weights | Warm round IPC residual is already milliseconds; likely small wall-time gain | Medium complexity for little measured gain; measure true transfer before implementation |
| 7 | Event lookup tables or batched forwards | Small: all non-encoding actor work is under 4% in this profile | Batching may change float results and action/RNG ordering; defer unless golden proves exactness |

No check scheduling, panel, league, validation count, precision, posterior-hand count
or asynchronous actor changes are proposed. A faster float-changing implementation
must be measured then reverted and reported for explicit approval.

The final multi-chunk note, permanent suite golden, upstream merge and full suite
remain for the final chunk; this document records the baseline only.
