# The AIs speak only what a real player would say aloud

Status: accepted (user decision, 2026-11-11).

## Context

Both modes now give the robots a voice (a different voice and pitch per robot), and spectator mode adds a panel showing what each AI is considering. A tempting design is to have the robots narrate their reasoning ("tengo 33", "tengo buenas cartas"). In truco that is private information: saying it would leak a hand that the opponent should not hear.

## Decision

A robot says aloud only what a real player may say at the table:

- calls and responses: truco, quiero retruco, quiero vale cuatro, envido, real envido, falta envido, quiero, no quiero, me voy al mazo;
- the tanto, when the rules require it to be announced ("treinta y tres", "son buenas");
- short phrases when playing a card.

Thinking is shown as a visible pause and a gesture, never as verbal reasoning. Gestures also accompany calls.

The reasoning panel (the top action probabilities of each AI) exists only in spectator mode, is shown on screen only, can be toggled off, and is never spoken. It does not appear in human-vs-robot mode, where it would reveal the robot's hand to the human.

## Considered options

- **Narrate the AI's reasoning aloud:** more informative, but it leaks private information and does not sound like a player.
- **Silent robots with text only:** safe, but loses the sense of a real game and of two distinct characters.

## Consequences

- What is heard is what a human opponent would hear, so voice never gives the human an advantage.
- Explanation for the audience stays a separate, optional, on-screen feature of spectator mode (compare the reasoning trace in CONTEXT.md).
