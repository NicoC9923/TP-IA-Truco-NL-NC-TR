# Phase 1 trainer optimization

Baseline: commit 94dc17e, Windows, Python, 12 worker processes. A cold cProfile
of two v3 matches (seed 1, target 15) took 2.552 s / 11,122,597 calls.
Cumulative: v3_summaries 2.230 s, incremental update .304 s, seen_cards .224 s.
Internal: v3_summaries .826 s, built-in sum .410 s, card rank .388 s.
There were 1,228,211 card-rank calls and 4,623,116 outcome generator calls.
The network and engine outside inference are negligible in this sample.

The trainer creates pools for league, finalist retest, and each promotion check;
map uses default chunksize 1 and transfers full decision logs. These are secondary
candidates; first target is posterior summaries, retaining ordered float additions.
Baseline production measurements are recorded in before-v3.txt / before-v2.txt.
The benchmark uses temporary directories which are removed automatically.

Determinism fixture: six individuals, two pairs, two finalists, eight validation
pairs, ten generations, target 1, seed 0, all three encoders. Ten generations
exercise the actual checkpoint writer; SHA-256 covers raw log bytes, every
champion and sidecar, and each checkpoint member's uncompressed bytes (ZIP
container timestamps are deliberately excluded). Workers 1 and 2 must both
match the same baseline. No timing fields occur in generations.jsonl.

## Change and exactness

Precomputed card IDs, ranks and brava flags replace repeated rule calls.
A small lazy table stores the unchanged strength formula for remaining ranks.
NumPy computes each posterior contribution; `np.cumsum` deliberately performs
sequential additions instead of `np.sum`'s pairwise reduction. Integer comparison
counts may be reduced freely. Multiplication precedes division exactly as before.
Posterior dictionaries, iteration order, Python total-weight summation, evidence
filtering, bluff updates, policy RNG and all trainer code remain unchanged.

The scalar-reference test checks exact equality of every summary over weighted,
filtered posteriors with zero through three cards played. All 236 tests passed
(82.744 s), including existing encoder goldens and training golden comparisons
for v1/v2/v3, workers 1 and 2. Checkpoint member contents and champion bytes match
the pre-change fixture exactly; generation log bytes are identical.

## Measurements

Production settings: population 64, pairs 16, finalists 8, finalist multiplier 4,
validation pairs 1000, check batch 200, workers 12, seed 0, default target and gate.
Each fresh two-generation run uses an automatically removed temporary directory.
The promotion check stopped after 200 pairs in both generations-2 comparisons.
Times below are seconds. Retest is total minus league minus check, and includes
the small champion/log-writing overhead; pool creation, serialization and transfer
remain included in their respective phases.

| Encoder / generation | League before | League after | Retest before | Retest after | Check before | Check after | Total before | Total after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 / 1 | 80.78 | 20.06 | 35.43 | 9.08 | 0 | 0 | 116.21 | 29.14 |
| v3 / 2 | 71.81 | 18.72 | 33.82 | 8.71 | 11.33 | 3.18 | 116.96 | 30.61 |

V3 full-run wall time: **233.232 -> 59.821 s**, **74.35% less time (3.90x)**.
Generation 2 alone: 73.83% less time. The first baseline generation partially
overlapped the single-process tiny golden recording, so its timing has slight
contention; the uncontended second-generation comparison independently clears 50%.
No other timing runs overlapped. Optimized v3 is below baseline v2 (96.32/98.32 s).

After cProfile, same seed/genome/two target-15 matches: .488 s, 1,830,111 calls;
v3_summaries .123 s cumulative / .010 s internal, update .347 s, seen_cards .230 s.
Compared with 2.552 s / 11,122,597 calls before, summary processing is 18.1x faster.

Pool/serialization probe: cold pool + 24 trivial jobs .255 s, warm .00456 s;
representative job 29,976 bytes / .0098 s for 1000 pickle round trips, full result
13,800 bytes / .1466 s for 1000 round trips. These are microbenchmarks, not an
additive estimate of production overhead. Pools are created per league/retest/check,
chunksize remains 1 and full decision logs still return. Their cost is too small
relative to measured inference to justify additional changes for this goal.

| Encoder / generation | League before | League after | Retest before | Retest after | Check before | Check after | Total before | Total after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 / 1 | 67.08 | 18.76 | 29.24 | 8.19 | 0 | 0 | 96.32 | 26.95 |
| v2 / 2 | 60.60 | 16.68 | 28.79 | 8.31 | 8.93 | 2.84 | 98.32 | 27.83 |

V2 full-run wall time: **194.707 -> 54.851 s**, **71.83% less time (3.55x)**.
The >=50% v3 goal is met. No trainer settings, network topology, encodings,
fitness, game rules or promotion criteria changed. No phase-2 work, pushes or PRs.
