# Neuroevolution for the trained agent

This is phase 1 of the two-phase training plan (ADR 0015).

The trained agent is a small neural network (a multilayer perceptron) whose weights are tuned by a genetic algorithm over generations of self-play. It was chosen because it covers two techniques from the course syllabus at once (genetic algorithms and neural networks), it produces real generations that can be shown in a presentation, and it needs no examples of the "correct move": it learns only from match outcomes, so it can discover bluffs nobody programmed.

## Considered Options

- **A genetic algorithm over the utility agent's thresholds:** simple, but it only tunes hand-written rules. Low ceiling.
- **Supervised learning:** needs examples labelled with the correct move, and there are none. The T1K22 dataset has deals, not decisions. Labelling with the current agent's decisions would only teach the network to imitate it, never to beat it.
- **MCCFR:** the most theoretically sound option for games with bluffing, but it advances by iterations rather than generations, needs an abstraction of the game state, and fits the syllabus less well.
- **Reinforcement learning (PPO, DQN) with PyTorch:** powerful, but much harder to tune and adds a heavy dependency (see ADR 0003). Later adopted as phase 2 with Deep Monte-Carlo (ADR 0015).
