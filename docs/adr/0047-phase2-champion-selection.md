# Phase 2 promotes on measured improvement against a fixed selection panel

Status: accepted (user decision, 2026-11-08). Thresholds are pilot values. Amends ADR 0017 for phase 2.

Phase 1's rule (≥55% of 1,000 pairs against the reigning champion alone) asks a single opponent whether the challenger is stronger. In a game that can be non-transitive, beating the incumbent's specific habits does not mean being stronger in general, and a panel that changes with every promotion moves the bar. Phase 2 instead asks whether the challenger does **better than the reigning champion against the same fixed opponents**.

## Rule

- **Selection panel v1,** frozen before the first check: shipped phase 1 run 1 generation 48, v2 seed 1 generation 46, v2 seed 2 generation 25 (each with its own encoder and profile semantics, hashes frozen), and the phase 2 snapshot at the end of the measuring run (ADR 0044). Panel members are reserved: they are never sampled in the training league (ADR 0043).
- **Each check (one generation), about 1,000 mirrored pairs:** the challenger and the reigning champion each play 100 pairs against each panel member on identical deals (800 pairs), plus 200 pairs challenger vs champion. Schedules rotate every check with fresh validation seeds; the champion is re-evaluated each time on the same deals as the challenger, so luck is shared. Results are cached only when checkpoint, schedule, action seeds, profile protocol and panel version are all identical.
- **Promotion** when both hold:
  1. the equal-weight panel gain (challenger minus champion, averaged over the four members) is at least 1 percentage point and its paired 95% lower bound is above zero, with uncertainty computed by clustering the per-deal differences across members;
  2. the challenger scores at least 47.5% against the champion directly. This is a screen against clear regressions, not a proof of non-inferiority.
- All checks run to completion at first (no early promotion). If checks take more than their time share (ADR 0044), training between checks is increased. A single halfway rejection for a clearly negative gain may be added later, accepting some false rejections.
- **Panel saturation:** when the reigning champion scores at least 70% against every member, pooled over its last three checks with at least 300 pairs per member (pooling resets when the champion changes), a panel v2 is frozen before the next check: keep the two original phase 1 members the champion beats least, replace the other two with two recent phase 2 snapshots (one strong general performer, one behaviorally distinct) chosen by a documented deterministic ranking, excluding the reigning champion and duplicates, and reserve them from the league. Challenger and champion are re-evaluated on the new panel, one overlap evaluation of the champion is run on both panels, and the evolution page marks the boundary; a score drop there is a change of yardstick, not a loss of ability. If too few distinct snapshots exist, replacement is postponed.

The log records per-member results for challenger and champion, the panel gain with its interval, the direct result, the panel version and the decision. Reactive and Adaptive Utility never enter the check (ADR 0021): the panel is a development yardstick used for selection, not an independent test; the reference evaluation remains the exam.

## Considered Options

- **Phase 1's 55% against the reigning champion only (ADR 0017):** convention from AlphaGo Zero, not justified by Truco evidence; vulnerable to non-transitivity. AlphaZero later dropped gating altogether.
- **55% aggregate over a rolling panel of the last 3 champions plus ≥50% direct (coordinator's first proposal):** the bar drifts with panel members, and ≥50% direct can reject broadly stronger networks.
- **Reuse cached champion results across checks:** saves ~40% of check time but loses the shared-luck pairing and can freeze a lucky champion.
- **Replace the fourth member with the latest snapshot every check:** keeps the panel fresh but destroys the fixed yardstick.

## Amendment (2026-11-09)

Panel members share each deal schedule, so their results may be correlated. For each pair index, average the challenger-minus-champion differences equally across members; compute the standard error across these per-pair averages and use 1.96 standard errors for the 95% interval. The gain and direct thresholds are unchanged.
