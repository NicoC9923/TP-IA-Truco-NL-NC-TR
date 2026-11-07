# Fitness = full matches won, plus point difference

Each individual plays full matches, to 15 or 30 points (ADR 0016). Its fitness is the percentage of matches won plus 0.1 × the normalized point difference. Full matches are needed so it learns to play the score (when falta envido pays off, when to take risks at the end of a match). The point-difference term breaks ties and reduces noise.

## Considered Options

- **Point difference per hand only:** less noisy, but never learns to play the score.

## Phase 1 learning review amendment

The default remains win rate plus 0.1 times normalized point difference (`--fitness-mode classic --points-weight 0.1`). For experiments, `classic` computes win rate + w × normalized difference; `mixed` computes (1 − w) × win rate + w × normalized difference, with w between 0 and 1. These settings affect league and finalist ranking, not the promotion check. The Sonnet learning review (F1) identified high sampling noise and proposed testing a stronger point signal; the empirical learning review measured Spearman 0.138 between independent 16-pair evaluations of the same 64 genomes. That single comparison does not establish that a mixed signal is better, so it is opt-in.

## Decision after fitness measurements

The user chose to keep classic fitness (win rate + 0.1 × normalized point difference) as the default; `--fitness-mode mixed` and `--points-weight` remain available for experiments, but mixed fitness is not enabled by default. Measurements on the generation-70 population were inconclusive. On a wider population of 48 champions and 16 random genomes, classic fitness with the common schedule had reliability Spearman 0.53 and alignment to a fixed-panel win-rate estimate 0.74; mixed weight 0.5 had 0.54 and 0.74, points-only had 0.48 and 0.73, and the independent-schedule baseline had 0.51 and 0.58. The small numerical reliability advantage for mixed 0.5 is not evidence of a meaningful improvement over classic in this single comparison. The panel and schedules differ between the baseline and common-schedule runs, so these figures do not isolate the effect of weighting or establish a general ranking.

A heavier margin weight can also favor a policy with a lower match win rate: for example, policy A with win rate 0.60 and normalized difference 0.02 scores 0.602 under classic and 0.31 under mixed 0.5, while policy B with win rate 0.55 and normalized difference 0.20 scores 0.570 and 0.375 respectively. Classic ranks A higher, whereas mixed 0.5 ranks B higher despite its lower win rate; promotion remains based on match wins.
