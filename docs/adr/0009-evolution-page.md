# A separate evolution page, fed by two logs

An HTML page separate from the game UI shows training. It reads two logs: the trainer's log, one line per generation, and the reference-evaluation log, written by a separate process (ADR 0022). Per generation it shows:

- maximum, average and minimum fitness;
- the succession of reigning champions (which generation had a promotion and with what win rate);
- the reference-evaluation curve: the champion against Reactive and against Adaptive Utility, with its margin of error;
- emerging behavior (envido, bluff and truco-acceptance rates);
- population diversity;
- a selector to view an example hand played by any generation's champion, decision by decision.

Phase 1 and phase 2 are shown side by side and, once phase 2 exists, the three predictions of the opponent's cards are compared (ADR 0019). The page was kept separate from the game UI so training doesn't depend on a match running.
