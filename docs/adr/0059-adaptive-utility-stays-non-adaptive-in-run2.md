# Adaptive Utility stays non-adaptive in phase 2 run 2

Status: accepted (user decision, 2026-11-09).

The policy wrapper in `proyecto/engine_policies.py` never calls Adaptive Utility's online adjustment. So in the symmetric engine it plays a fixed strategy, the same in every match. All earlier reference scores, from phase 1 and run 1, were measured against this fixed version.

Run 2 keeps it that way, in the league, in the panel and in every report:

- Changing two things at once would hide which one made the difference.
- A fixed opponent is also easier to make deterministic and resumable (ADR 0042).

Turning the adjustment on, so that the network faces an opponent that learns during a match, is a separate later experiment with its own ADR. Reports say "Adaptive Utility (fixed)" wherever the difference matters.
