# Opponent card inference in three stages

Working out the opponent's cards is separated from choosing the move, and each part uses the technique that suits it:

1. **Bayesian module**, in two layers kept separate:
   - **Exact filter:** uses only certain evidence (declared tantos, "son buenas", seen cards) to rule out impossible hands among the 7,770 combinations. A ruled-out hand never comes back.
   - **Weighting:** gives the remaining hands more or less weight according to the opponent's behavioral signals (a truco call, for example), in proportion to the bluff rate measured for that opponent (ADR 0012). If the opponent rarely bluffs, their truco weighs a lot; if they bluff often, it weighs little. With adaptation off, a neutral rate is used.
2. **Supervised inference network** (phase 2, ADR 0019): sees the full history and predicts the opponent's cards. It is trained on the real cards from simulated matches, so labels come for free. Its prediction goes through the exact filter so it never proposes an impossible card.
3. **Decision network:** takes the weighted inference's summaries as input, computed by the same code in training and against the human.

Truco calls are still not certain evidence, because they may be bluffs: they enter as weighted evidence and never rule a hand out.

## Considered Options

- **Let the decision network learn inference on its own from the raw history:** with neuroevolution that requires much larger networks, and a genetic algorithm doesn't scale to that size.
- **Exact filter only** (the original version): ignores what the agent already knows about how often this particular opponent bluffs.
