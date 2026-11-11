# The robot's turn is paced like a person's and runs asynchronously

Status: accepted (2026-11-11). Step 3 of the 3D card and arm work; builds on ADR 0066 and 0067.

## Context

The robot answered inside the same HTTP request, so every move was instant and indistinguishable from a program. The page faked a 3 s pause in the browser, and spoke whatever sentence the game had written, reasoning included.

## Decision

- **Pacing is a pure module (`ritmo.py`).** Thinking takes 1.5-3 s (2.5-4.5 s for truco, retruco, vale cuatro, real/falta envido, or answering a big call); the card takes the arm 0.45 s to reach plus 0.8 s to lay (the 3D `Ritmo`, kept in sync by a test); short pauses after a baza and between hands. A speed multiplier (x0.5, x1, x2) scales everything; `Ritmo.instantaneo()` is zero delay and synchronous. `PartidaTruco` defaults to instant, so tests stay fast; the server opts in (`--ritmo=`, `TRUCO_RITMO`, `POST /truco/ritmo`).
- **What is said is a catalog (`voz_jugador.py`).** Calls, answers, the tanto when the rules require it (the mano always says its number; the pie says its number only if it beats it, else "son buenas") and short phrases when laying a card (sometimes silence), independent of the card. Each seat has a voice profile (preferred voice names, pitch, rate). The old free-text bubble stays on screen, but is never spoken.
- **Async turn.** The robot's decision is computed by a worker, but nothing it causes (state, phrase, gesture, 3D move) is revealed until the think time has passed: the public state stays frozen at what it was, with `robot.estado = pensando`, then `actuando` while the card is laid. The delay is chosen after deciding, from the action taken. This keeps the existing synchronous game code intact; the human's part still runs inside the request.
- **Human input is rejected while the robot acts** (clear message), not queued. A reset or new game cancels the pending turn through an epoch counter.
- **Speech.** Each phrase reaches the page once, with an increasing id; the page picks the es-AR voice by preference, then any Spanish one.

## Consequences

- Both robots of spectator mode can use `ritmo.py` and `voz_jugador.py` unchanged.
- The robot's decision uses the state at the moment the human acted; since the human cannot act meanwhile, it equals deciding after the wait.
- A chain of robot moves (answer, tanto, card) takes the sum of its think times.
