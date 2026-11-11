# Spectator mode runs on the symmetric engine with a human-paced driver

Status: accepted (user decision, 2026-11-11). Builds on ADR 0002; does not change it.

## Context

Spectator mode shows two AIs playing each other, each with its own robot. ADR 0002 ties `PartidaTruco` to "human vs robot": it embeds the AI, is hard-wired to one robot and one human seat, and drives the simulator. The symmetric engine (`proyecto/symmetric_engine.py`) is the opposite: player A vs player B, no UI, no robot, no threads, reproducible from a seed. It calls a policy at each decision with that player's own cards and the public state, so it already treats both seats the same.

## Decision

Spectator mode uses the symmetric engine for the rules and a separate driver for the show. The driver sits between the engine and the outside world:

- it slows each decision to human pace (a visible thinking pause, then the arm movement), so the engine's speed never shows;
- it turns the engine's decisions into what the viewer sees and hears: robot motion, spoken calls and the web page's state;
- it handles the page's controls (pause, step, speed x0.5 / x1 / x2) by holding or releasing the next decision;
- it applies the chosen match length (points, or a number of hands with one tie-break hand).

`PartidaTruco` stays the human-vs-robot match and is not modified for this. Both flows share the rules in `truco_reglas.py`, and the existing test that they score the same points on fixed hands (ADR 0002) covers the shared ground.

## Considered options

- **Extend `PartidaTruco` with two AI seats:** puts a second robot, a second AI and a paused/stepped flow inside a class built around one human, and risks changing the human match that already works. The new seat would also need its own proof that it behaves like the engine's.
- **Run the engine at full speed and replay the result:** loses pause, step and speed control, and the viewer could not follow a decision as it happens.

## Consequences

- The AIs play the same way as in training and in the network-vs-network checks, since the policy and the engine are the same ones.
- Two drivers exist (human match, spectator), each owning its own pacing. Shared pieces (speech, gestures, cards) are reused, not duplicated.
- The engine today ends a match at a points target only; the "N hands" length needs an end condition added there or in the driver, plus a tie-break hand if points are level.
