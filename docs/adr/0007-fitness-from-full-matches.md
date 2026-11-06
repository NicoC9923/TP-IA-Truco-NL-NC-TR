# Fitness = full matches won, plus point difference

Each individual plays full matches, to 15 or 30 points (ADR 0016). Its fitness is the percentage of matches won plus 0.1 × the normalized point difference. Full matches are needed so it learns to play the score (when falta envido pays off, when to take risks at the end of a match). The point-difference term breaks ties and reduces noise.

## Considered Options

- **Point difference per hand only:** less noisy, but never learns to play the score.
