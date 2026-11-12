# Styled opponents: copies of the agent itself with fixed biases

For the network to learn to read the opponent profile (ADR 0012), training must include opponents that play differently. Those opponents are **copies of the phase's own networks** (clones or hall-of-fame members) with a fixed bias added to their outputs while they act as opponents:

- **Bluffer:** bonus to calling truco and envido with weak cards.
- **Cautious:** penalty on "quiero".
- **Aggressive:** bonus to raising.
- **Calculating:** penalty on bluffing.

Each bias's size is configured and recorded so runs are comparable. After the first promotion, about half of training matches are against styled opponents, 12.5% against each when the full eight-slot cycle is used (ADR 0004). No strategy code is written for this, and nothing from the existing agents is used (ADR 0021).

## Considered Options

- **Four variants of the Utility agent with fixed parameters** (the original version): the trained agent would learn to exploit Utility's logic through the back door, and its evaluation against Adaptive Utility would be inflated.
- **Four new bots written from scratch:** independent of the app, but it is hand-written strategy to maintain, and biases on the agent's own networks give enough variety.
- **No styles (pure self-play):** standard when strength is the only goal, but clones play alike, the profile would carry no signal, and adapting to the opponent would learn almost nothing.

At style strength 1.0, the logit adjustment is +3 for objectively weak envido or truco calls by the bluffer and -3 for those calls by the calculating opponent. The cautious opponent receives -3 for quiero; the aggressive opponent receives +3 for raises (real envido, falta envido, retruco, vale cuatro). These are additive softmax logits and do not reveal style to the opposing network.

## User-approved phase 1 deviation from story 17

Phase 1 defaults to `--style-strength 0.5`, scaling those offsets to ±1.5, and `--styles-after-first-promotion`: the league is entirely unbiased until a challenger is promoted, then uses the 50/50 mix. `--no-styles-after-first-promotion` restores styles from generation 1 and strength 1 restores ±3. This is a user-approved deviation from story 17's 50/50 mix from the start. The Sonnet learning review (F3–F4) observed near-random early policies and reasoned that ±3 caricatures can reward exploiting styles instead of learning to read them, pulling selection away from the unbiased promotion gate. The empirical learning review similarly noted the offsets change action odds about twentyfold, but did not measure their effect on selection; this is a design rationale, not an established measured improvement.

## Amendment (2026-11-08)

Phase 2 uses 20% styled opponents from the start with small offsets in Q units, ramped up, instead of 50% after the first promotion (ADR 0043).
