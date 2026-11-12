# Phase 2 optimization step 1: skip already enforced card evidence

Ticket #35, chunk 2. Kept: both fixed full benchmarks improve and every frozen
semantic hash matches the pre-optimization baseline. Baseline details, commands,
timing caveats and original fixtures remain in [phase2-baseline.md](phase2-baseline.md).
Raw results for this step are in `phase2-step1/`.

## Change and exactness

`Inference.seen_cards` returns immediately when there are no excluded cards and
the supplied opponent cards are a subset of `self.played`. The existing posterior
already enforces these cards. All intervening inference operations either remove
hands or change weights; none restore a filtered hand. Empty initial evidence is
also an identity operation. The original filtering code remains unchanged for any
new opponent card or nonempty exclusion list, including contradictory evidence.

The fast path does not reconstruct the dictionary, touch floats, change ordering,
or add cached state. `self.played` already contains exactly the supplied subset,
so skipping its update is also an identity operation. Incremental inference's reset
conditions remain unchanged: hand change, bluff-rate change, decreasing call count,
changed/removed reveal, and explicit reset all rebuild an empty played-card set.
This shared helper is also used by phase 1; its frozen encodings and training
artifacts must therefore still match.

The actor baseline profile attributed 1.119 of 2.027 seconds to repeated card
filtering. That motivated this small fast path instead of an array rewrite.
The identical four-match, 206-decision cProfile pass now takes **1.150 s** and
3,704,604 calls (baseline 2.027 s and 7,502,276 calls). Card filtering drops from
**1.119 to 0.295 s**, while summary-array construction stays near 0.41 s. This
confirms the speedup comes from the intended hotspot, with no decision-count change.

## Fixed full benchmark

Same machine and harness as the baseline: 12 workers, seed 35, two 96-match rounds,
mixed targets, league on, progress on, measuring panel v0, full 100/200-pair check,
and 400 validation situations with the head on. No CPU timing jobs overlap.
Each benchmark invocation has the unchanged 175-second process-tree deadline.

| Measurement | Off baseline | Off step 1 | On baseline | On step 1 |
|---|---:|---:|---:|---:|
| Total wall, seconds | 29.212 | 21.583 | 67.342 | 53.065 |
| Training matches/s | 6.573 | 8.896 | 2.851 | 3.618 |
| Learner samples/s | 132.170 | 178.892 | 57.245 | 72.647 |
| Actor matches/s, including cold map wait | 30.355 | 36.701 | 29.072 | 37.024 |
| Full check, seconds | 21.064 | 14.481 | 20.667 | 14.774 |
| Cold validation, seconds | disabled | disabled | 24.652 | 19.135 |
| Warm validation, seconds | disabled | disabled | 11.236 | 9.839 |
| Learner seconds/update | 0.01064 | 0.01098 | 0.29744 | 0.29118 |
| Samples / updates | 3,861 / 8 | 3,861 / 8 | 3,855 / 8 | 3,855 / 8 |

Head-off wall time decreases **26.1%** (throughput increases **35.3%**);
head-on wall time decreases **21.2%** (throughput increases **26.9%**).
Learner time is essentially unchanged, as expected. Single measurements establish
a large useful effect here, not tight confidence bounds on these percentages.
Round-level actors, merge, learner, persistence and IPC proxy data remain in the
JSON files; the baseline's overlap and serialization caveats still apply.

## Equality and regression evidence

Both full benchmark `hashes` objects equal their original counterparts, including
model/optimizer tensors, semantic generation log, all retained checkpoints and raw
generation snapshot bytes. The original small fixtures are never regenerated.
The eight golden cases cover head off/on, workers 1/12, and uninterrupted/fresh-process
partial-generation resume; checked-in `equal-*.json` files retain their settings.
All eight cases passed against the original fixtures from chunk 1.

The scalar-reference tests exercise the original filter independently of the fast
path across repeated/empty/new evidence, exclusions after a card was already played,
contradictory exclusions, weighted calls and all incremental reset cases. Ordered
weights and distributions must compare exactly, not approximately.

Checks passed:

- `python -m unittest discover -s proyecto -p 'test*inference*.py'`: 13 tests, 0.398 s.
- `python -m unittest discover -s proyecto -p '*golden*.py'`: 4 tests, 39.305 s;
  includes phase 1 original match/log fixtures, extraction log, stateless/incremental
  v1/v2/v3 encodings, and all six encoder/worker training golden comparisons with
  their real checkpoint files.
- All eight original phase 2 golden cases and both full benchmark hash sets match.
- Staged whitespace check passes. The longest timing run takes 53.065 s; no process
  started for this chunk remains running.

## Next

Keep this change. Plan item 2 (fixed validation preprocessing) and item 3 (boolean
learner masks) remain for separate chunks. During item 3, also profile the remaining
auxiliary loss, particularly the minibatch-by-9,880 combination scores, and report
whether an exactly preserving speedup exists. The permanent suite golden, final
performance note, upstream integration and full suite remain for the final chunk.
