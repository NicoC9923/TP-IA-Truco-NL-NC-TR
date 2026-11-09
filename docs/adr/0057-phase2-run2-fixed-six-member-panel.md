# Phase 2 run 2: a fixed six-member selection panel without rotation

Status: accepted (user decision, 2026-11-09, on a Codex recommendation). Amends ADR 0047 and ADR 0051 for run 2.

A promotion check in run 2 compares challenger and reigning champion against six equal-weight members. The panel is frozen before training and never rotates.

| Member | Kind | SHA-256 (prefix) |
|---|---|---|
| v2s1-g46 | phase 1 network, encoder v2 | `9659d27a3c70` |
| v2s2-g25 | phase 1 network, encoder v2 | `3f25fb0b59b5` |
| run 1 generation 12 | phase 2 network | `681f456ba206` |
| run 1 generation 28 (the run 1 champion) | phase 2 network, temperature 0.05, prior 5 | `66ed791f4dd1` |
| Reactive | reference, code identity | — |
| Adaptive Utility | reference, code identity | — |

## Why these four networks

- **Easy members say little.** By the end of run 1, gen28 beat three of the four members of its final panel by about 75%. A member beaten that easily barely separates a better candidate from a worse one.
- **Generation 12 and gen28 still challenge.** Generation 12 held gen28 to 55.6%, and gen28 against itself is about 50%. Both still reward real progress.
- **Gen28 is the target.** It is the network run 2 must beat (ADR 0058).
- **Not all phase 2.** The two phase 1 v2 networks keep their seats, so the network members are not all phase 2 checkpoints.

Panel v1, with run1-g48 and the measuring-run network, was considered. Its extra encoder variety does not make up for losing the generation 12 challenge.

## Rules

- **Thresholds and pairs per member as in ADR 0047.**
  - The equal-weight gain over the six members must be at least 1 point, with a clustered 95% lower bound above zero.
  - The direct result must be at least 47.5%.
- **No rotation.**
  - A persisted switch turns off both saturation tracking and panel replacement, so saturation neither rotates the panel nor ends the run.
  - Patience (40 checks) and the 12 h cap still apply. 12 h is a cap, not a promise.
- **Network members are reserved.** They are never drawn in the league, neither as plain nor as styled snapshots. The references are the deliberate exception (ADR 0056).
- **The held-out networks never appear** anywhere in training or selection (ADR 0060).

## Risks

- With six members, a check plays about 50% more matches.
- Rewarding wins against gen28 can favor a specialist that beats gen28 without playing better in general. The held-out part of the acceptance test (ADR 0058) guards against this.
