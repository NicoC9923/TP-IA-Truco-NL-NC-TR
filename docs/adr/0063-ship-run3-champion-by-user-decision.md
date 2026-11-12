# Ship run 3's champion by user decision, despite one failed guardrail

Status: accepted (user decision, 2026-11-11). Overrides the ADR 0061 decision for run 3's exam only; ADR 0061 stays the rule for future runs.

## Context

Run 3's champion, generation 73, failed one guardrail in its final exam (experiments/phase2-run3-exam/README.md). Against the near-untrained phase 1 network v1s2-g2 (a generation-2 network), it scored 90.3%, against 94.0% for the shipped run 2 generation 38: −3.7 points, interval [−5.1, −2.3]. It gained 13.7 points against fixed Adaptive Utility (85.8% vs 72.2%), and was level with g38 against Reactive, the shipped phase 1 network and the other held-out network.

## Decision

The user judged that, in practice, generation 73 is the better network for the game, and decided to ship it. A descriptive all-vs-all round robin (experiments/all-vs-all, frozen before play, 1,000 mirror pairs per pairing, all phase 2 networks greedy) was run first as a safety check. Generation 73 beat generation 38 directly in 65.5% of pairs [63.8, 67.2], and had the best equal-weight mean against the other seven agents (74.0%). Shipping was conditional on generation 73 not losing that direct duel.

Generation 73 ships greedy, with profile prior 5 and encoder v3 (1,635 inputs). Generation 38 is retired.

## Consequences

- At ship time the full suite passed: 633 tests OK (`py -m unittest discover -s proyecto -p "test_*.py"` from the repository root, rerun 2026-11-11).
- This was decided after seeing the exam, so it is a judgement call, not a pass of the pre-registered rule. Reports say so.
- The known weakness, a small but clear loss against an erratic, barely trained opponent, is carried into run 4's goals: a more varied league, and a diagnosis of which change caused it.
