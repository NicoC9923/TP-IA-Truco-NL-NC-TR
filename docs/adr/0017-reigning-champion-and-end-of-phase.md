# Reigning champion and end of phase by self-play

Progress and the end of a phase are decided only with networks from the phase itself, without the existing agents (ADR 0021):

- There is a **reigning champion**, initially the first champion.
- At the end of each generation, its champion plays up to 1,000 pairs of mirror matches against the reigning champion on the validation deals. If the check establishes a result above or below the threshold it can stop early; otherwise it uses all pairs. If its observed win rate is at least 55%, it is **promoted**: it becomes the reigning champion and enters the hall of fame.
- The phase **ends** after 20 consecutive generations without a promotion. The number is a starting point, to be tuned once the engine's speed is measured.
- The app's network modes load their phase's reigning champion.

With 1,000 pairs the margin of error is about ±2 percentage points, so 55% indicates a real improvement and not card luck. Phase 2 has no generations: it runs the same check every fixed number of training matches, and each check counts as a generation on the evolution page.

This is AlphaGo Zero's method: a new network replaced the best one only if it won at least 55% of 400 games against it.

## Considered Options

- **End when the champion beats Adaptive Utility 55% of the time** (the original version): puts an existing agent in the training loop, since it decides when to stop and which champion reaches the app. Rejected by ADR 0021. Adaptive Utility moves to the reference evaluation (ADR 0022).
- **Measure over few matches (say, 50):** can't tell skill from chance.
- **Promote every generation's champion without a check:** a generation's champion can be worse than an earlier one, and the hall of fame would fill with near-copies.

Promotion checks now inspect up to 1,000 validation pairs in batches of 200. A check stops early when the 95% paired interval is wholly below or at least the promotion threshold; otherwise the final decision uses the observed win rate. This saves matches when the result is clear. Repeated looks slightly inflate the error rate, as in sequential testing; AlphaGo Zero used 400 games. Always playing a fixed 1,000 pairs wastes work on clear outcomes, while a fixed 200 gives insufficient precision for close outcomes. Batch size zero disables early stopping.
