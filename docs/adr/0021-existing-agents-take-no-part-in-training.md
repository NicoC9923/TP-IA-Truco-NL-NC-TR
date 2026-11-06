# Existing agents take no part in training

Reactive, Adaptive Utility and anything built from their code are kept out of training the trained agent: they are not league opponents, they don't feed fitness, they don't decide when a phase ends and they don't pick which champion reaches the app. Training uses only networks from its own phase (ADR 0004, 0013 and 0017).

The reason is that the app's modes should be independent architectures. If the trained agent practiced against the existing ones, it would learn to exploit them specifically, and comparing them live would no longer be fair: one would have studied the other.

The existing agents are used **afterwards**, as the reference evaluation (ADR 0022): they measure the champion, but the trainer never reads the result. This is standard practice: AlphaZero trained only against itself and its strength was then measured against Stockfish, which it never saw in training; DouZero was compared against existing bots it didn't train with; Libratus and Pluribus were measured against professional poker players.

What training does reuse is the rules module (card ranking, tantos, falta envido), because that is the game, not a strategy.

## Considered Options

- **A league with Reactive, Adaptive Utility and four Utility variants** (the original design): external anchors against strategy cycles, but with exactly the coupling we want to avoid.
- **Validate and end the phase against Adaptive Utility:** doesn't change the weights directly, but decides when to stop and which champion is used, so it puts the existing agent in the loop.
