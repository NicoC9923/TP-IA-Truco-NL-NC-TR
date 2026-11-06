# Styled opponents: copies of the agent itself with fixed biases

For the network to learn to read the opponent profile (ADR 0012), training must include opponents that play differently. Those opponents are **copies of the phase's own networks** (clones or hall-of-fame members) with a fixed bias added to their outputs while they act as opponents:

- **Liar:** bonus to calling truco and envido with weak cards.
- **Cautious:** penalty on "quiero".
- **Aggressive:** bonus to raising.
- **Calculating:** penalty on bluffing.

Each bias's size is fixed and documented so runs are comparable. About half of training matches are against styled opponents, 12.5% against each (ADR 0004). No strategy code is written for this, and nothing from the existing agents is used (ADR 0021).

## Considered Options

- **Four variants of the Utility agent with fixed parameters** (the original version): the new agent would learn to exploit Utility's logic through the back door, and its evaluation against Adaptive Utility would be inflated.
- **Four new bots written from scratch:** independent of the app, but it is hand-written strategy to maintain, and biases on the agent's own networks give enough variety.
- **No styles (pure self-play):** standard when strength is the only goal, but clones play alike, the profile would carry no signal, and adapting to the opponent would learn almost nothing.
