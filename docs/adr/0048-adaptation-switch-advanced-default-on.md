# The adaptation switch stays, default on, in advanced settings

Status: accepted (user decision, 2026-11-08). Amends ADR 0012 and spec #15 decisions 11 and 13 on placement.

Normal play always adapts: the network uses the opponent profile and bluff-rate weighting. The "Adaptarse al rival" switch is kept for both network modes, inside a collapsed advanced/presentation section, on by default. While off, the game shows a clear "Adaptación desactivada" status; a new session starts with it on again, so one visitor's choice never carries over to the next. Off means neutral profile inputs and neutral-rate weighting; exact evidence and the current hand's event history are still used, so the network still reads in-hand behavior.

Phase 2 trains about 5% of its learner episodes with a neutral profile, so the network plays well both with the switch off and against a new opponent with no history yet. The reference evaluation measures the RL champion with adaptation on and off on paired deals; that, not a live demonstration, is the evidence that adaptation helps.

Why keep it: removing it would not make the network stronger (same network and policy when on, and neutral-profile play is needed anyway for new players), while it lets the presenter show the effect live after presenting the measured numbers, and the Neuroevolution mode already has it.

## Considered Options

- **Remove the switch from the UI; keep "no adaptation" only as an exam option:** simpler screen, same strength and same measured evidence, but no live demonstration and an inconsistency with the Neuroevolution mode.
- **Keep the switch on the main screen (spec #15 decision 13):** casual players should not have to decide which mode is stronger.
