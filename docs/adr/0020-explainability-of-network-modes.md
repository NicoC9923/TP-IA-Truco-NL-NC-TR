# Explainability of the network modes

A network has no thresholds to show, so in the two network modes (ADR 0005) the reasoning trace shows:

1. The three most likely actions, with their percentages.
2. What the weighted inference believes about the opponent's cards (ADR 0010).
3. Whether the move was a bluff, by the objective criterion (ADR 0011).
4. With **Adapt to opponent** on, the opponent profile in use (ADR 0012).

This keeps the product's promise of showing why Ottoman does what it does, even when the policy is a network.
