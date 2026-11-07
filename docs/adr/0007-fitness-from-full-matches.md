# Fitness = full matches won, plus point difference

Each individual plays full matches, to 15 or 30 points (ADR 0016). Its fitness is the percentage of matches won plus 0.1 × the normalized point difference. Full matches are needed so it learns to play the score (when falta envido pays off, when to take risks at the end of a match). The point-difference term breaks ties and reduces noise.

## Considered Options

- **Point difference per hand only:** less noisy, but never learns to play the score.

## Phase 1 learning review amendment

The default remains win rate plus 0.1 times normalized point difference (`--fitness-mode classic --points-weight 0.1`). For experiments, `classic` computes win rate + w × normalized difference; `mixed` computes (1 − w) × win rate + w × normalized difference, with w between 0 and 1. These settings affect league and finalist ranking, not the promotion check. The Sonnet learning review (F1) identified high sampling noise and proposed testing a stronger point signal; the empirical learning review measured Spearman 0.138 between independent 16-pair evaluations of the same 64 genomes. That single comparison does not establish that a mixed signal is better, so it is opt-in.
