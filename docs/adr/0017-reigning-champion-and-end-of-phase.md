# Reigning champion and end of phase by self-play

Progress and the end of a phase are decided only with networks from the phase itself, without the existing agents (ADR 0021):

- There is a **reigning champion**, initially the first champion.
- At the end of each generation, its champion plays up to 1,000 pairs of mirror matches against the reigning champion on the validation deals. The check can stop early only to reject a challenger; promotion requires all pairs. If its observed win rate is at least 55%, it is **promoted**: it becomes the reigning champion and enters the hall of fame.
- The phase **ends** after 20 consecutive generations without a promotion. The number is a starting point, to be tuned once the engine's speed is measured.
- The app's network modes load their phase's reigning champion.

With 1,000 pairs the margin of error is about ±2 percentage points, so 55% indicates a real improvement and not card luck. Phase 2 has no generations: it runs the same check every fixed number of training matches, and each check counts as a generation on the evolution page.

This is AlphaGo Zero's method: a new network replaced the best one only if it won at least 55% of 400 games against it.

## Considered Options

- **End when the champion beats Adaptive Utility 55% of the time** (the original version): puts an existing agent in the training loop, since it decides when to stop and which champion reaches the app. Rejected by ADR 0021. Adaptive Utility moves to the reference evaluation (ADR 0022).
- **Measure over few matches (say, 50):** can't tell skill from chance.
- **Promote every generation's champion without a check:** a generation's champion can be worse than an earlier one, and the hall of fame would fill with near-copies.

Promotion checks inspect up to 1,000 validation pairs in batches of 200. The default stops early only when the upper paired interval is below the threshold; otherwise promotion uses the observed win rate after all pairs. Repeated looks at ordinary 95% intervals inflate error rates, making early promotions overconfident. The rejection rule can still discard a promising challenger; it is a compute tradeoff, not a calibrated sequential test. Batch size zero disables early stopping. The historical `both` mode preserves early promotion for existing runs.

A 400-pair minimum still permits repeated uncalibrated promotion decisions. A calibrated sequential test would need a designed error budget and changes the decision rule. Always playing 1,000 pairs wastes work on clear failures. These alternatives were rejected for this change.

## Amendment (2026-11-08)

Phase 2 promotes on measured improvement against a fixed selection panel instead of 55% against the reigning champion alone, and ends after 8 hours or 40 checks without promotion (ADRs 0044, 0047).
