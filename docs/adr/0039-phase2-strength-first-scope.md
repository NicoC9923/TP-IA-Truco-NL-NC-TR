# Phase 2 is designed for playing strength, not for symmetry with phase 1

Status: accepted (user decision, 2026-11-08).

When phase 2 was designed (grilling of 2026-11-08, with independent advice from a Codex `gpt-6.1-sol` advisor), the user stated that phase 2 is not bound to earlier decisions: the goal is the most effective and efficient network for playing Truco on this CPU-only PC. Earlier ADRs written for both phases (0004, 0013, 0015, 0017, 0019) still describe phase 1; for phase 2 they are amended by ADRs 0040–0048. Each amendment was weighed on playing strength per CPU hour, and the advisor was asked to flag recommendations that were only anchored to earlier decisions.

What stays shared with phase 1: the symmetric engine and the policy interface, the exact opponent-card filter (ADR 0024), the trainer log format with one line per generation, checkpoints in `training_output/` and weights-only export (ADR 0018), the separate reference evaluation (ADR 0022), and the rule that Reactive and Adaptive Utility never take part in training or selection (ADR 0021).

## Consequences

- The two phases are no longer an isolated comparison of learning algorithms: phase 2 may train against phase 1 networks (ADR 0043) and uses a different input, network and selection rule. The final comparison is presented as a comparison of the two playable network modes (ADR 0046).
- ADR 0015's "phase 2 never plays phase 1 networks" no longer holds for phase 2's training league and selection panel.
