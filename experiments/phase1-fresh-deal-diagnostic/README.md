# Phase 1 fresh-deal diagnostic

The [rule](RULE.txt) was registered on 2026-11-07 before running: compare seed 1 near-miss generations 66, 87, 121 and final generation 123 with seed 1 champion generation 23. A candidate passes if its win rate minus its margin exceeds 50%. The network-only comparisons used 2,000 mirror pairs each, validation-partition deals and schedule seed 4242. The [head-to-head script](../phase1-seed1/h2h.py) is identical to the script used here.

| Candidate generation | Win rate ± margin |
|---|---|
| 66 | 54.85% ± 1.38 |
| 87 | 54.80% ± 1.41 |
| 121 | 55.35% ± 1.39 |
| 123 | 52.15% ± 1.38 |

All lower bounds exceed 50%, so the rule passed: these near-misses replicated on fresh deals, and the 55% gate was blocking modest real gains. See [results](results.txt). These candidates were selected, so their intervals do not account for selection. Distinct schedule seeds give different deals in practice but cannot guarantee that no individual deal recurs.
