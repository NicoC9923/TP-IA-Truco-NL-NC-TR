# 0031. Frozen network panel and pre-registered experiment decision

## Context

A single champion comparison can miss strategic cycles and specialization. The 53% gate experiment needs a common, pre-registered evaluation before its result is known. Reference opponents must never select champions (ADR 0021).

## Decision

Freeze panel-v1 on 2026-11-07 before the 53% experiment result: run 1 generations 22 and 48 (an earlier promotion and the shipped champion), run 2 generation 12, seed 1 generations 9 and 23 (an earlier and final champion), and seed 2 generation 2. These provide historical and independent-run network opponents without choosing them after observing this experiment. Manifest hashes identify the exact files; a changed panel requires panel-v2.

Compare the final candidate to the seed 1 generation 23 baseline using 4,000 mirror pairs and to each panel member using 1,000 pairs, with the baseline also measured against each member on the same panel schedule. Report pair-level 95% margins. The decision is promising only when the direct paired lower bound is above 50%, the candidate's mean exceeds the baseline's, and no member's candidate rate falls more than two percentage points below baseline. Otherwise report no demonstrated benefit. If the candidate or the baseline has the same hash as a member, skip that member for both means and the regression criterion (a baseline playing itself is about 50% by construction and would distort them; the direct 4,000-pair comparison already covers it); if no members remain, the decision cannot be promising. This is an experiment decision, not the separate shipping rule.

Reserve validation schedule seeds 9001 for the direct final comparison and 9002 for panel comparisons. Gates and reference evaluations use their run seeds (0, 1, 2); published head-to-heads used 777 and 4242. Reference evaluation defaults to the run seed, though its CLI permits overrides: these reserved seeds must not be used there or in other head-to-heads. Even seeds select validation partition 0 and training uses partition 1.

Distinct schedule seeds provide independent pseudorandom streams, not guaranteed exclusion of repeated individual card combinations: the deal space is finite and different seeds can generate the same deal. We accept incidental overlap here because these are fixed-size final comparisons, not a confirmation conditional on individual screen deals, and retaining all previous deal histories would add complexity and potentially alter the sampling distribution. This does not establish strict deal-disjointness; do not reuse these schedules to select subsequent candidates. A conditional confirmation test would need explicit exclusion or another independently justified design.

## Alternatives and practice

Champion-only comparison can reward a specialist that regresses elsewhere. Reference opponents are excluded from selection by ADR 0021. Re-picking the panel after seeing outcomes invites selection bias. Fixed held-out evaluations, common random numbers, paired analysis, and prespecified criteria are common practices for controlling comparison noise and post-hoc selection; repeated experimentation on this same panel can still overfit it.
