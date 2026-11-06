# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Single static HTML file (`proyecto/voz.html`), served by the stdlib Python server `proyecto/servidor_truco.py` at `http://127.0.0.1:8080`. No Node, no build step. Launched with `jugar_truco.bat` / `jugar_truco.sh`. External assets must work from that local server (Chrome or Edge, for the Web Speech API).

## Users

Primary: the student team presenting their Artificial Intelligence course project (UADE, TPO "Inteligencia Artificial, Jueves Mañana") to professors and classmates, usually on a laptop or a projector. One person plays a real game of Truco against the robot while the audience watches; the UI has to make the AI legible to that audience in the moment.

## Product Purpose

Play a complete match of Argentine Truco (two players, to 15 or 30 points) against **Ottoman**, a simulated Unitree G1 humanoid sitting at a table in MuJoCo, and show *why* the AI does what it does. Success: a full game is playable without friction, and an observer can see Ottoman's reasoning, its bluffs, and how it adapts to the human over matches.

## Positioning

Not a card-game app: an AI agent you can play against and inspect. Its distinguishing mechanisms are a utility agent built on a 474,000-hand statistical corpus, a dynamic bluff (farol) policy, online modelling of the human opponent's bluffing and folding, explainable decision traces, and switchable agent architectures (reactive, utility-based, hybrid adaptive) for comparison. A physical humanoid in simulation mirrors the game.

## Operating Context

- Two terminals: the MuJoCo simulator (`sim.bat`) and the web server. The UI shows whether the robot simulator is connected and keeps working without it.
- Input channels: click a card, show a physical Spanish-deck card to the webcam (detection constrained to the hand, plus a free test-detect mode), push-to-talk voice in Rioplatense Spanish ("¡Envido!", "¡Truco!", "¡Quiero!", "Al mazo", card names), or typed text.
- Ottoman speaks its lines through browser TTS (toggleable, repeatable).
- A second, out-of-game mode orders the robot by voice or text (wave, celebrate, shake hands, sit, stand, deal, stop, walk), in Spanish or English, with a log of what was executed, blocked, or not understood.

## Capabilities and Constraints

All current functionality must be preserved; the UI form is free. Server endpoints (unchanged):
`GET /truco/estado`, `GET /truco/historial`, `GET /cartas/<valor>_<palo>.png`, `POST /truco/jugar | envido | truco | responder | mazo | voz | nuevo | reiniciar | modo_ia | detectar`, `POST /comando`.

Game state exposes: score and goal (15/30), who is mano, round (baza 1-3), turn, the table by baza with winners (incl. parda), human hand and envido points, available calls (envido, real envido, falta envido, next truco level), a pending call from Ottoman with response options, truco stake, Ottoman's last line, match bluff stats, end-of-hand reveal of Ottoman's three cards (played / kept) with its real envido and bluff verdicts, end-of-match summary, a 25-line log, and the AI learning summary (mode, bluff quotas for envido and truco, threshold offsets, detected human profile, reasoning trace of up to 8 steps with phase/decision/metric/threshold/bluff policy/explanation, and the last 30 matches).

The client simulates a 3-second "thinking" pause before Ottoman's moves and polls state every 2 s.

Terminology is Argentine Spanish and must stay so: mano, baza, parda, envido, real envido, falta envido, truco, retruco, vale cuatro, quiero / no quiero, irse al mazo, farol, tantos.

## Brand Commitments

- Ottoman is the robot's name; it is a Unitree G1.
- The card faces are the Spanish deck PNGs in `proyecto/cartas/` (262x400, Fournier-style), 40 cards: values 1-7, 10-12 of Espada, Basto, Oro, Copa. These are the cards; do not redraw them.
- UI language: Spanish (Argentina), voseo.

## Evidence on Hand

- Real match history in `proyecto/historial_partidas.json`; corpus stats in `proyecto/truco_stats.json` (474k hands).
- No testimonials, metrics, or claims beyond what the code computes; do not invent any.

## Product Principles

1. The game is always playable first: the current decision (play a card, answer a call) is never buried under AI exposition.
2. Make the AI's mind visible: every Ottoman decision and bluff should be inspectable at the moment it happens and afterwards.
3. Honest reveal: when a hand ends, show what Ottoman really held and whether it lied.
4. Every input channel (click, camera, voice, text) leads to the same game; none is required.
5. Readable from across a room: a projector audience must follow score, stake, and whose turn it is.

## Accessibility & Inclusion

Must read on a projector (large type for score and state, strong contrast). Keyboard-operable controls; push-to-talk must also work with touch and keyboard, not only mouse.
