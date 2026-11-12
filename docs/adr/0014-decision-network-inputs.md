# Decision network inputs

The decision network sees six groups of data, encoded as numbers (about 60 inputs):

1. **Its cards:** rank, suit, whether already played, and each card's tantos.
2. **The table:** cards played in each trick, who won each, pardas, and the current trick.
3. **The calls:** envido state (what was called, by whom, whether resolved, declared tantos) and truco state (level and who holds the quiero).
4. **The score** (ADR 0016).
5. **Inference about the opponent:** probability they hold a brava, expected strength of their cards, and probability of winning the next trick, computed with the weighted inference (ADR 0010).
6. **The opponent profile** (ADR 0012): neutral when adaptation is off.

Groups 5 and 6 are computed by the same code in training and against the human. The network gets precomputed summaries (inference, score) instead of the raw history, because a small evolved network cannot derive them on its own.

## Encoder version clarification

ADR 0035 distinguishes the existing face-value encoding from experimental held-card Truco rank encoding.
