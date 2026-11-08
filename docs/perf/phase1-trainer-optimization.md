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
