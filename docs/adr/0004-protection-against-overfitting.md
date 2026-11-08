# Protection against overfitting in self-play

Pure self-play has two problems: strategy cycles (A beats B, B beats C, C beats A) and specializing in the quirks of one's own clone. To avoid them without using the agents already in the app (ADR 0021):

1. Each individual plays a league made only of networks from its own phase: roughly half its matches against **unbiased opponents** (clones from its generation and the hall of fame) and the other half against **styled opponents**, copies of those same networks with fixed biases (ADR 0013).
2. The hall of fame keeps only champions that were promoted to reigning champion (ADR 0017), so it is a short list of real improvements rather than hundreds of near-copies.
3. Progress is measured on validation deals that are never used for training.
4. Every deal is played as mirror matches, swapping seats, to cancel out the luck of the cards.

This is common practice: OpenAI Five played about 80% of its games against its current self and 20% against its own past versions, and AlphaStar trained in a league made only of its own versions.

## Considered Options

- **Play only against the previous generation:** simplest, but produces exactly the cycles and specialization we want to avoid.
- **Add Reactive, Adaptive Utility and four Utility variants to the league** (the original version of this ADR): external anchors help against cycles, but the trained agent would learn to exploit the existing agents, and comparing them in the app would no longer be a comparison of independent architectures. Rejected by ADR 0021.

## Amendment (2026-11-08)

Phase 2 replaces the hall of fame with a bounded snapshot pool that admits a snapshot every generation, promoted or not (ADR 0043).
