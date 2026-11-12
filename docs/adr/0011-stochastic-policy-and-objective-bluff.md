# Stochastic policy and objective bluff detection

The network picks its action at random in proportion to its scores (softmax), both in training and in play. That lets it learn mixed-frequency bluffs ("bluff 25% of the time with this hand"). A deterministic policy would be predictable and exploitable.

Since a network has no inspectable "intent" to bluff, a bluff is judged from the outside, with the same criterion for every mode: an envido call or raise with fewer than 24 tantos, or a truco call or raise with a hand strength (the corpus percentile) below 0.44. These are the thresholds the game already uses to judge the human's bluffs, so the bluff rates of every mode and of the human are comparable.
