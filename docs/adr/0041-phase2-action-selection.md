# Phase 2 chooses actions by softmax over Q with one fixed temperature

Status: accepted (user decision, 2026-11-08); the temperature is a pilot value.

The network turns its legal Q values into probabilities, p ∝ exp(Q / T), and samples, with the same fixed temperature T in training rollouts, promotion checks, reference evaluation and the game. The method is described as Deep Monte-Carlo with Boltzmann (softmax) action selection. T starts at 0.1 (a 0.2 vs 0.0 value gap gives roughly 88/12) and is fixed from the measuring run before the real run (ADR 0044), then recorded.

Why: Truco rewards being unpredictable, so a network that always plays the same move in the same spot is easy for a human to read; training and play use the same policy, so the Monte-Carlo targets describe the policy that is actually deployed; it matches ticket #24 and phase 1's stochastic policy (ADR 0011).

Limits recorded with the decision (advisor): softmax spreads randomness over every decision, including tactically obvious ones, so it can keep making small mistakes; it is not a theory of correct bluff frequencies; a cold temperature can stop exploring actions that were undervalued early. The run logs tactical mistakes and overly concentrated probabilities, and temperature/exploration is revisited before enlarging the network if they persist. Style offsets (ADR 0043) are added to Q before dividing by T, so they are in Q units: phase 1's ±1.5 logit offsets would overwhelm Q values in [−1, 1].

## Considered Options

- **DouZero's ε-greedy training and greedy play:** the standard DMC method, kept as the baseline for a later optional experiment (equal budget, several seeds); not run now because each comparison costs overnight runs.
- **ε-greedy in training, softmax in the game:** learns one policy and deploys another.
