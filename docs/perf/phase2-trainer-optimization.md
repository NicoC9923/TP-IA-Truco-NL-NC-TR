# Phase 2 trainer optimization

Ticket #35 / spec #15. Baseline production revision:
`7ed6b76078e7cb8c1527200ed0c9a7f3b732cda5`. All kept changes preserve the original
model/optimizer tensors, semantic generation log, checkpoint state and generation
snapshot bytes for the fixed configurations, including fresh-process resume and
worker-count changes. No defaults, precision, posterior-hand counts, league/check
semantics or phase 1 results changed.

## Machine, fixed protocol and reproduction

Windows 11, AMD64 Family 26 Model 68 Stepping 0, 16 logical CPUs, 12 actor workers,
one torch/BLAS thread per process, Python 3.12.10, torch 2.15.0+cpu. Exactness is
promised on this machine/library configuration, as in ADR 0042.

Every full timing uses seed 35, two generations with one 96-match round each,
mixed targets 15/30, 30% random starts, default league shares, four-match blocks,
minibatch 512, temperature 0.1, progress on, measuring panel v0, 100 mirrored pairs
per member/candidate and 200 direct pairs. Generation 1 establishes the champion;
generation 2 checks it: 800 pairs / 1,600 check matches. Auxiliary validation uses
all 400 fixed situations each generation. Champion export is disabled to protect
the repository champion. The head-off/on runs make 3,861/3,855 learner samples and
eight updates respectively; this remains constant at every step.

```powershell
python proyecto/bench_phase2.py bench --output training_output/perf35/off.json
python proyecto/bench_phase2.py bench --auxiliary --output training_output/perf35/on.json
python proyecto/bench_phase2.py actor --output training_output/perf35/actor.json
python proyecto/bench_phase2.py validation --output training_output/perf35/validation.json
python proyecto/bench_phase2.py auxiliary --output training_output/perf35/auxiliary.json
```

Each invocation starts a fresh supervised process with a 175-second process-tree
deadline. Runs are sequential; no timing/profile CPU jobs overlap. Temporary runs
stay on the repository drive and are removed after successful completion. No
individual measurement runs beyond three minutes. These are fixed local timings,
not confidence intervals or a substitute for the later measuring run (#28).

Training matches/s and samples/s divide by the entire run, including checks and
validation. Round timings include their boundary work; total wall additionally
includes setup/teardown. Serialization probes remain enabled in all comparisons.
Ordinary pickle sizes are payload proxies, not actual shared-memory wire sizes;
map wait minus the longest job is a scheduling/transfer residual, not pure IPC.
The full measurement definitions and source-marker instrumentation are documented
in [phase2-baseline.md](phase2-baseline.md).

## Baseline versus final

| Measurement | Off baseline | Off final | On baseline | On final |
|---|---:|---:|---:|---:|
| Total wall, seconds | 29.212 | 19.738 | 67.342 | 39.386 |
| Training matches/s | 6.573 | 9.727 | 2.851 | 4.875 |
| Learner samples/s | 132.170 | 195.609 | 57.245 | 97.877 |
| Round 1 / round 2, seconds | 4.481 / 23.123 | 3.606 / 14.538 | 30.497 / 35.193 | 21.480 / 16.320 |
| Full check, seconds | 21.064 | 13.212 | 20.667 | 12.733 |
| Cold / warm validation, seconds | disabled | disabled | 24.652 / 11.236 | 17.353 / 1.859 |
| Mean learner update, seconds | 0.01064 | 0.01078 | 0.29744 | 0.11582 |
| Actor-only matches/s, including cold wait | 30.355 | 40.847 | 29.072 | 40.277 |

Whole-run time is **32.4% lower head off / 41.5% lower head on**. The head-on learner
uses **61.1% less time per update**, and warm validation takes **83.5%
less time**. Check times fall about 37–38%. The fixture is intentionally check-heavy;
learner savings would matter more with more training matches between checks.

## Measured steps and motivating profiles

| Step | Off wall, s | On wall, s | On matches/s | Decision |
|---|---:|---:|---:|---|
| Original | 29.212 | 67.342 | 2.851 | Frozen baseline |
| 1: skip idempotent evidence filters | 21.583 | 53.065 | 3.618 | Keep: -26.1% / -21.2% wall |
| 2: fixed validation preprocessing | 21.494 | 43.609 | 4.403 | Keep: -17.8% on wall |
| 3: membership masks + ordered combination scores | 21.032 | 42.333 | 4.535 | Keep: -59.3% learner time |
| 4: reuse unchanged summary arrays | 19.738 | 39.386 | 4.875 | Keep: -6.2% / -7.0% wall |

Percentages compare each step with its predecessor; small unrelated timing changes
are ordinary variation. Raw JSON, cProfile text and frozen equality records remain
in `phase2-baseline/` and `phase2-step1/` through `phase2-step4/`.

1. **Evidence filters.** The initial four-match actor profile spent 1.955/2.027 s
   encoding, including 1.119 s filtering cards. Reapplying unchanged evidence is
   idempotent. Return early only with no exclusions and no new opponent cards,
   retaining the ordered posterior and all reset behavior. The same actor profile
   falls to 1.150 s and filtering to 0.295 s.
2. **Validation preprocessing.** Profiling found three exact-posterior rebuilds per
   situation and repeated baseline metric scans. Cache fixed exact/weighted scores
   and ordered learned-hand IDs by the complete replay protocol, bounded to one
   protocol. Never cache learned logits, probabilities or metrics. Current-model
   predictions, original scalar metric addition order and tie rankings remain.
   Warm validation falls from 9.839 to 1.826 s in this step.
3. **Masks and auxiliary loss.** The 512-sample operation profile measured 135.7 ms
   in masks, 95.8 ms scoring 9,880 combinations, 49.4 ms backward and only 3.5 ms in
   cross entropy. Precomputed membership cuts masks to about 3.9 ms; exact faster
   scoring cuts score construction to 16.9 ms. Combined measured update operations
   fall from 299.4 to 113.0 ms. Actual full-run updates fall from 284.6 to 115.7 ms.
4. **Summary arrays.** After filtering improved, `_summary_arrays` still cost
   0.406/1.150 s in the actor profile. Cache the derived tuple and invalidate on all
   evidence, played-card and weight mutations. The unchanged builder preserves its
   ordered weight sum, operations and arrays. Actor time falls to 1.038 s and array
   construction to 0.296 s; full checks and cold validation replay also improve.

Detailed reports: [step 1](phase2-step1.md), [step 2](phase2-step2.md),
[step 3](phase2-step3.md), [step 4](phase2-step4.md).

## Custom backward and rejected float-changing proposals

The ordinary score expression materializes a batch x 9,880 x 3 gather. Gathering
three columns separately and adding is cheaper, but ordinary autograd changes the
order in which duplicate-card gradients accumulate. The kept custom function starts
the forward additions at positive zero (preserving signed-zero bits) and reconstructs
the original gather backward using row/hand/card ordered `index_put_` accumulation.
It preserves both independent trunk forwards and all loss/optimizer operations.
Backward itself is slower (about 74 versus 50 ms), but the roughly 80-ms score
saving leaves a substantial net improvement.

Tests compare raw forward and gradient bytes for float32/float64, multiple batch
sizes through 512, all hands and noncontiguous hand subsets, negative zero and
arbitrary upstream gradients. Eight-update probes also compare all gradient hashes,
losses, model and Adam tensors, followed by the original full training goldens.

Rejected measurements, with new masks:

| Candidate | Update operations, ms | Exact? | Reason |
|---|---:|---|---|
| Original score expression | 168.9 | Reference | Comparison |
| Flattened `index_select` | 166.0 | Yes in probe | Marginal 1.7% gain; inferior to kept path |
| Three gathers, ordinary autograd | 76.1 | No | First gradients differ despite equal first scalar loss |
| Shared Q/auxiliary trunk forward | 160.7 | No | Changes gradient rounding despite equal first scalar loss |
| Three gathers, original backward order | 113.0 | Yes | Kept after complete golden gates |

The float-changing candidates are reproducible benchmark counterexamples, never
production paths. No approximate beliefs, fewer hands, float16, asynchronous actors,
changed check scheduling or altered league/panel semantics were introduced.

## Equivalence and permanent regression guard

Original fixtures were recorded before production changes, with head off/on, seed 35,
mixed targets, real league and v0 panel, four eight-match rounds, two generations,
minibatch 32 and a small complete check. The 26 updates collect 754/757 samples.
Every kept step passes the original hashes with workers 1/12, uninterrupted runs,
fresh-process partial-generation resumes and generation-boundary resumes. The last
case compares a warm uninterrupted validation cache with a cold rebuilt cache.
Both larger fixed benchmark hash sets also equal the original baseline at every step.

The typed hashes cover tensor/array bytes, model and Adam state, semantic log,
checkpoint counters/league/panel/RNG state and raw generation snapshot bytes.
Only documented elapsed-time fields and resolved filesystem locations are normalized;
no loss, metric, action, promotion, RNG or model float is dropped or rounded.

`test_phase2_optimization_golden.py` permanently compares both heads after a fresh
generation-boundary restart with two actors against the original one-worker fixtures.
The manual benchmark matrix covers 12 workers. Existing phase 1 original match/log,
encoding and six encoder/worker training goldens also remain exact. Dedicated tests
cover evidence fast paths, summary-cache invalidation, changed-model/cold/warm/evicted
validation caches, boolean masks and the custom backward.

## Deferred items 5–7

- Fixed validation IDs are already cached. All remaining warm validation is under
  two seconds; further marginal vectorization has a small ceiling and must preserve
  scalar addition and tie ordering. Deferred by coordinator direction.
- Warm map transfer/scheduling residuals are about three milliseconds, so additional
  IPC machinery is not justified by this profile. Payload proxies are not wire sizes.
- Actor forward was about 1.3% of its profile and event work was small. Batching
  introduces float/RNG-order risk for little evidence of benefit. Deferred.

## Integration and full suite

Implementation checkpoints: baseline harness `cbc8d10`, evidence filters `aee2a2c`,
validation cache `4e73106`, auxiliary masks/scoring `19885c4`, and summary reuse plus
the permanent original-golden guard `322058d`.

After `git fetch origin`, `git merge origin/feat/ia-entrenada` reported **Already up
to date** (upstream `7ed6b76`). The full suite ran from the repository root:

```powershell
python -m unittest discover -s proyecto -p "test*.py"
```

**333 tests passed in 166.079 seconds**, with no skips or failures. The suite includes
the permanent pre-optimization head-off/on resume guard, all phase 1 goldens, cache
invalidation/reference tests and byte-level custom-backward checks. A 175-second
process-tree watchdog bounded this full-suite run as well. No process started for
this work remains running; no pushes, PRs or issue changes were made.

## Ticket #35 acceptance evidence

- **Baseline/final throughput:** same fixed machine/configuration; matches/s,
  samples/s, round/check times and head-off/on results recorded above and in raw JSON.
- **Golden equality including resume:** permanent suite test plus 12-case worker/
  resume matrix and both full benchmark hash sets match the pre-optimization code.
- **Performance note:** every kept step has its motivating profile, measured effect,
  equivalence evidence and a linked detailed report; rejected proposals are explicit.
- **Full suite:** 333 tests green after fetching/merging the required upstream branch.
- **Bounded CPU work:** every profile/timing run is under three minutes, supervised
  with a 175-second deadline; runs are sequential and the full suite also finishes
  below that limit.
