# Phase 2 optimization step 2: cache fixed validation preprocessing

Ticket #35, chunk 3. Kept: the full head-on benchmark takes **17.8% less time**
than step 1 and matches every original semantic hash. The baseline protocol and
interpretation remain in [phase2-baseline.md](phase2-baseline.md); this step's raw
measurements and equality records are in `phase2-step2/`.

## Change and identity

`_validation_preprocessing` caches one fixed validation protocol, keyed by schedule
seed, target, temperature and situation count. It retains each situation's encoded
inputs, true/exposed cards, ordered compatible hands, integer hand-index tensor and
the exact/weighted baseline metric results. These depend only on the existing fixed
replay policy and public evidence, never on the learner. The cache holds at most one
protocol to bound the memory occupied by hand lists and index tensors; changing a
protocol evicts it. The existing replay cache is unchanged.

Every invocation still calls the current model once per situation, gathers its
current card logits in the same ordered hand-index tensor, performs the identical
three-card sum and softmax, and scores the resulting learned probabilities with the
unchanged scalar metric loop. No learned logits, probabilities, scores, or generation
reports enter the cache. `card_distribution` itself is unchanged; validation no
longer rebuilds its fixed evidence and hand indices or constructs its unused card
marginals. Cold preprocessing still computes exact and weighted distributions through
their original functions, preserving every weight and its iteration order.

Metric aggregation keeps the original per-situation `+=` sequence, division and
standard-error computation. The reference test also uses sequential additions:
Python 3.12's compensated `sum` is not interchangeable with that original sequence.
Returned report dictionaries do not expose mutable cached baseline results.

## Fixed full benchmark

Same machine and configuration as step 1: 12 workers, seed 35, two 96-match rounds,
mixed targets, league/progress on, measuring panel v0 with the full 100/200-pair
check, and 400 validation situations when enabled. Runs are sequential and retain
the original benchmark's serialization probes and 175-second process-tree limit.

| Measurement | Off step 1 | Off step 2 | On step 1 | On step 2 |
|---|---:|---:|---:|---:|
| Total wall, seconds | 21.583 | 21.494 | 53.065 | 43.609 |
| Training matches/s | 8.896 | 8.933 | 3.618 | 4.403 |
| Learner samples/s | 178.892 | 179.628 | 72.647 | 88.400 |
| Full check, seconds | 14.481 | 14.312 | 14.774 | 14.363 |
| Cold validation, seconds | disabled | disabled | 19.135 | 18.338 |
| Warm validation, seconds | disabled | disabled | 9.839 | 1.826 |
| Learner seconds/update | 0.01098 | 0.01144 | 0.29118 | 0.28459 |
| Samples / updates | 3,861 / 8 | 3,861 / 8 | 3,855 / 8 | 3,855 / 8 |

Head-on throughput rises **21.7%** in this step; warm validation takes **81.4% less
time**. Cold validation improves only 4.2%, because the cache must first construct
the fixed replay, baseline metrics and hand indices. The approximately 8-second
warm-validation saving accounts for most of the 9.46-second whole-run reduction;
small differences in actors, checks and learner timings are ordinary run variation.
Head-off time is effectively unchanged (0.4% difference).

Compared with the original head-on baseline, wall time is now **67.342 -> 43.609 s**,
**35.2% less time**, and throughput **2.851 -> 4.403 matches/s**. Round splits and
IPC proxy data are retained in the JSON files; no throughput/IPC semantics changed.

## Golden and regression gates

Both fixed full benchmark `hashes` objects equal the original chunk-1 baseline,
including model/optimizer tensors, semantic log, retained checkpoints and raw
generation snapshot bytes. No original fixture was regenerated.

The original eight golden cases cover head off/on, workers 1/12, and uninterrupted
versus fresh-process partial-generation resume. The harness now additionally accepts
`--resume-round 2`, stopping after the first complete generation and restarting the
process. This compares a cold rebuilt preprocessing cache after resume with the
uninterrupted run's warm second-generation cache against the same original fixture:

```powershell
python proyecto/bench_phase2.py golden --workers 12 --auxiliary --resume --resume-round 2 --reference docs/perf/phase2-baseline/golden-on.json --output training_output/perf35/generation-resume.json
```

All **12** original-golden comparisons passed: the eight original cases plus four
generation-boundary restarts (head off/on, workers 1/12). The additional records are
named `equal-*-generation-resume-*.json`. They hash the original fixed protocol;
the added resume cutoff is a harness control, not a changed training setting.

Independent reference tests retain the pre-cache validation path, including the old
`card_distribution` reconstruction. They compare exact metric floats for cold, warm,
evicted/rebuilt caches and changed model weights, verify a current forward for every
warm situation, verify output mutation cannot affect the cache, and vary each of the
four protocol key inputs. The existing deterministic-validation test clears both
caches so that it continues to test a genuinely cold replay.

Checks passed:

- Original golden matrix and generation-boundary resumes: 12/12 comparisons.
- Both full benchmark hash sets equal their original baseline counterparts.
- `test_validation_preprocessing.py`: 2 tests, 2.634 s.
- `test_phase2_card_metrics.py`: 5 tests, 0.789 s.
- All `*golden*.py` suite tests: 4 tests, 39.515 s, including all six phase 1
  encoder/worker training comparisons and their checkpoint contents.
- Python compilation and staged whitespace checks pass; all CPU runs are below
  three minutes and no process started for this chunk remains running.

## Next

Keep this change. Plan item 3 (boolean learner masks) remains a separate chunk; it
must also profile the other auxiliary-loss work, particularly combination scoring,
and report any exactly preserving opportunity. Final integration, the permanent
golden suite test, final performance note and full suite remain for the final chunk.
