# Phase 2 final exam: one fixed test decides which network ships

Status: accepted (user decision, 2026-11-10). Amends ADR 0058 for every run after its closed one-time test, and ADR 0041 for the shipped temperature.

## Why

The project's goal is a network that beats every other game mode (ADR 0039). Run 2's acceptance (ADR 0058) asked a narrower question: is the run 2 champion clearly better than run 1 generation 28 in a direct duel? Generation 38 scored 51.85% with a lower bound of 49.975%, so generation 28 stayed shipped. That test never measured Adaptive Utility, which generation 28 beats only 37% of the time (four-mode round robin), while run 2's development checks put its late generations around 54–57% against it on 100 pairs each.

So the question that decides shipping changes. It is no longer "does it beat the champion in a duel?" but "does it do better than the champion against the game's opponents, starting with Adaptive Utility, without getting clearly worse against any of them?" The ADR 0058 test stays closed: it is not rerun, and the direct duel is not part of the new exam.

## The exam

The same rule judges every finished phase 2 run.

- **Opponents (five):** Adaptive Utility (fixed, ADR 0059), the primary opponent; Reactive; shipped phase 1 run 1 generation 48 (a game mode that run 2 never trained or selected against); and the two held-out phase 1 networks of ADR 0060, v1s1-g23 and v1s2-g2. Phase 1 networks use their own encoder and sampling.
- **Baseline:** the shipped phase 2 champion, at its shipped settings.
- **Candidates:** fixed before any game. Candidates are ranked; the first one that passes ships.
- **Design:** for each opponent, 1,000 mirror pairs on validation deals (partition 0), half to 15 points and half to 30. Every candidate and the baseline play the same deals with the same action seeds, so their paired difference per pair is the measured quantity. Every game builds fresh policies, profiles and random generators. Seeds are new and audited against every earlier namespace.
- **Pass, for a candidate:**
  1. **Primary:** its paired gain over the baseline against Adaptive Utility is above 0, with the one-sided Holm correction across the candidates (overall one-sided 2.5%).
  2. **Guardrails:** against each of the other four opponents, the 95% lower bound of its paired gain is above −2 points.
  3. **Balance:** its equal-weight mean gain over the five opponents is above 0.
- **Shipping:** the highest-ranked passing candidate replaces the shipped champion, with its temperature written in its sidecar. The old champion is archived. If none passes, the champion stays and the exam is not repeated for those candidates.

Win rate against Adaptive Utility above 50% is reported, not required: requiring it could block a network that is far better than the shipped one.

## Temperature before the exam

The fixed T = 0.05 was chosen from 34 decisions of an eight-minute network (ADR 0041) and was never measured on a trained one. Before the exam, one exploratory test on separate seeds plays the first candidate at T = 0.05, 0.02 and greedy (always the highest legal Q) against Adaptive Utility and phase 1 generation 48, 300 mirror pairs each. The best equal-weight mean replaces 0.05 only if it beats 0.05 by at least 2 points; the chosen value is used for every candidate in the exam. The baseline keeps its own shipped temperature. A lower temperature makes the network more predictable to a human, which is why a small gain is not enough to change it.

## First use: run 2

Candidates: run 2 generation 38 (first, the run's own selected champion) and generation 49 (second, the last fully checked generation). Both were named after their development checks were seen; that is disclosed, and expected to inflate their check scores (winner's curse). Fresh seeds make the exam itself unbiased for these two named candidates. Baseline: run 1 generation 28 at T = 0.05, profile prior 5.

The frozen settings, seeds and analysis are in `experiments/phase2-final-exam/RULE.txt`.

## Considered Options

- **Rerun or relax the ADR 0058 test:** shopping for a pass. Rejected.
- **Require more than 50% against Adaptive Utility:** the ideal, but it could block a large real improvement.
- **Only Adaptive Utility:** run 2 trained against it (ADR 0056), so a gain there can be specialization; the guardrails catch that.
- **Choose the best candidate after the exam:** checkpoint shopping. The ranking is fixed beforehand.
- **Tune the temperature inside the exam:** each temperature would be one more candidate.
