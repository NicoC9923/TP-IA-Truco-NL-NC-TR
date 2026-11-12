# Phase 2 run 2: the reference opponents train with the model

Status: accepted (user decision, 2026-11-09). Amends ADR 0021 and ADR 0043 for phase 2 from run 2 on. Phase 1 and run 1 are unchanged.

The project's priority is now the strongest possible play (ADR 0039). So from run 2 on, Reactive and Adaptive Utility are **training opponents** and **selection panel members** for phase 2.

Run 1 showed why this matters:

- About 90% of its training opponents were closely related networks: itself, its snapshots and phase 1 networks.
- Its panel held only networks too.
- So the run learned to beat networks like itself, and it stopped improving against the hand-written strategies after generation 10.
- Against Adaptive Utility it scored about 35%. It lost points mostly on bets: accepted envidos, and folds and refusals of truco (`experiments/phase2-run1/README.md`).

A network only learns to handle a strategy it meets.

## Run 2 league (share of training matches)

| Opponent | Share |
|---|---|
| Current network (self-play) | 30% |
| Snapshots of the run | 20% |
| Phase 1 networks (same four as run 1) | 10% |
| Styled snapshots: bluffer, cautious, aggressive, calculating | 5% each, 20% in total |
| Reactive | 10% |
| Adaptive Utility | 10% |

How the references play in the league:

- They play on training deals (partition 1), like every other league opponent.
- Only the learner's decisions become training samples.
- They are identified by their code and configuration, not by a weights file.
- Their random numbers come from a seeded generator, and they start each block fresh.

## What this costs

The original reason for ADR 0021 still holds. Once the network practices against the references, a score against them no longer measures strength against unseen play. From run 2 on, reference scores are reported as **results against familiar opponents**, not as an independent exam. ADR 0060 says what can still be claimed and how.

## Considered Options

- **Keep ADR 0021 (references never train):** honest reference scores, but the network keeps losing to the strategy family it never meets. Rejected for strength.
- **New hand-written opponents similar to the references:** keeps the exam clean, but costs design time and may not cover what the references do. Deferred.
- **References in training only, not in the panel:** selection would still reward beating only related networks, which is how run 1 failed.
