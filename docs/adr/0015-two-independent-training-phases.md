# Training in two independent phases: neuroevolution and reinforcement learning

Both phases train a neural network; what differs is how its weights are adjusted. The goal is a strong AI, not just an academic demo, so training has two phases that share the policy interface, the engine, the league rules, the end-of-phase rule and the evolution page:

1. **Neuroevolution** (ADR 0001): a genetic algorithm adjusts the weights. It is in the course syllabus, produces generations that can be shown, is quick to get working, and exercises all the scaffolding. Its ceiling is medium, because a genetic algorithm doesn't scale to large networks.
2. **Deep reinforcement learning with self-play:** DouZero-style Deep Monte-Carlo, in PyTorch on CPU. It adjusts the weights by gradient descent, the usual way to train neural networks.

The two phases are **independent runs**. Phase 2 starts from random weights and its league is built only from its own networks (ADR 0004 and 0013): it never plays phase 1's champion. They are compared only at the end, in the reference evaluation (ADR 0022) and on the evolution page. That keeps the genetic-algorithm vs reinforcement-learning comparison clean: the only difference is how each network learns.

This is common practice: DouZero and AlphaZero trained by pure self-play from scratch, with no opponents from another method.

## Considered Options

- **Use phase 1's champion as a phase 2 opponent** (the original version): phase 2 would start against a strong opponent, but would learn to beat phase 1 specifically, biasing the comparison between phases.
- **MCCFR:** theoretically the strongest (converges to a Nash equilibrium), but only on a simplified version of the game. It also doesn't exploit weak opponents and, being a table, can't take the opponent profile, so adapting to the opponent would be lost. It doesn't reuse the network either.
- **PPO** instead of Deep Monte-Carlo: equally valid, but more delicate to tune. Deep Monte-Carlo has proven stable for card games trained on CPU.
- **Keep only one of the two phases:** neuroevolution alone is medium-strength; reinforcement learning alone loses the syllabus's genetic algorithm and the generations to present.
