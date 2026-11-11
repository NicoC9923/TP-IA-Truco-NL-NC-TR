# A project-owned deck, and kinematic Truco cards on the 3D table

Status: accepted (2026-11-11). Step 1 of the 3D card and arm work; step 2 adds a second G1 in seat 2, step 3 tunes pacing.

## Context

The web's 40 card images were cut from a template image of unknown origin (`template_naipes.jpg`), so the project could not say it owned them. The MuJoCo scene showed the robot at the table, but the cards on it were two plain white boxes that never changed: the 3D view did not show the match being played.

## Decision

**Deck.** `proyecto/generar_baraja.py` draws all 40 cards and a card back with Pillow from code alone: no template, no font files (the index digits are a small stroke font). Traditional Spanish layout: pip layouts for 1-7, simple original figures for sota, caballo and rey, corner indices, and the broken frame that tells the suit (oros none, copas one break, espadas two, bastos three). Output is 524 x 800, the web's 262/400 ratio, under the same file names, so the web is unchanged. The template image is deleted.

**Cards in 3D.** The same PNGs texture thin card meshes in the scene. Cards are mocap bodies moved kinematically every frame, never grasped physically:

- the robot holds its three cards fanned in the left hand, faces toward itself;
- to play, the right hand reaches the chosen card, the card rides on the right palm, and it is laid face up in the next table slot of the robot's side; earlier bazas stay in a row;
- a human's card (there is no human body) slides in from the human's edge to that side's slot;
- a new hand resets every card.

Arm poses come from a small damped least-squares IK solved once at startup, then interpolated. The place motion lasts 0.8 s (`Ritmo.llevar`), reach and return 0.45 s each, all configurable. The two-hand handoff worked, so no fallback was needed.

**Seats and cameras.** Seat 1 (Ottoman) faces seat 2 across the table; seat 2 has a chair and a camera already, for the second robot of step 2. Three named cameras: `general`, `detras_robot_1`, `detras_robot_2`. F7 in the viewer cycles them; the game server can ask for one through the simulator socket (`Robot.elegir_camara`).

**Human vs robot.** In that mode (the default) the behind-robot cameras are disabled: asking for one returns the overview, and the viewer is pulled back if MuJoCo's own `[`/`]` keys land on one. As a second guarantee, the robot's cards in hand show their backs on both faces, so not even a free camera reveals them.

**Game hook.** `PartidaTruco` queues the deal and every played card (human and robot) to one worker thread that sends them to the simulator in game order. Without a simulator it does nothing, and the web game never waits for the 3D.

## Consequences

- The deck is the project's own and can be regenerated on any machine with the same pixels. Pillow joins `requirements.txt`, only to redraw the deck and to take screenshots; the PNGs are committed.
- `py -3 -m sim.capturas <dir> jpg` (from `entorno/`) renders each camera at key moments offscreen with the live pose code; `docs/screenshots/step1/` holds the reviewed set.
- The robot's left arm keeps the holding pose while it has cards, so gestures only move the right arm then.
- Card faces in the robot's hand are hidden in human mode even from a camera behind the robot that a later mode might allow; robot-vs-robot mode shows them.
