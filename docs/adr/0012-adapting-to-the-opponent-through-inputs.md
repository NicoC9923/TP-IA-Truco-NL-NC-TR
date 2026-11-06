# Adapting to the opponent through inputs and inference, not retraining

The network's weights don't change while it plays a human: training needs thousands of matches and a human plays few. The agent adapts by reading the opponent, in two ways:

1. **Opponent profile as an input:** rates of bluffs seen, folds, truco accepted and envido called. It changes **how it plays** against that opponent.
2. **Weighted inference** (ADR 0010): the bayesian module weighs the opponent's truco calls by their measured bluff rate. It changes **what it believes** about their cards.

Profile rules:

- It is computed from what is observed, by the same code in training and in play. The network is never told an opponent's true style, because against the human it will never know it.
- Each statistic starts at a neutral value with a prior worth about 20 observations, and moves toward the observed rate as evidence accumulates, so the agent doesn't overreact to the first hands.
- The bluff rate counts only hands where the opponent's cards were revealed, using the objective bluff criterion (ADR 0011).
- There is a single human profile, saved across sessions, with a button to reset it when someone else plays. The app has no user accounts.
- With the **Adapt to opponent** switch off (UI: "Adaptarse al rival", ADR 0005), both ways use neutral values, so the two positions differ only by adaptation.

For the network to learn to read the profile, it faces opponents with different styles in training (ADR 0013).

## Considered Options

- **Retrain on the human's matches:** far too few matches, and the result is unstable.
- **Portfolio of champions** (pick, during the session, the one doing best against the human): needs several clearly different champions, and switching only pays off after many matches.
- **Profile input only, or weighted inference only:** each shows adaptation from a single angle. Together they show it changes both how it plays and what it believes.
