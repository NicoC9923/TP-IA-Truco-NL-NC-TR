# Phase 2 run 3: explicit hand facts, a per-hand target and greedy selection

Status: accepted (user decision, 2026-11-10, with a Claude Sonnet 5.5 advisor opinion). Amends ADR 0040 (inputs and target), ADR 0041 (action selection) and ADR 0047 (checks) for run 3. Shipped networks play greedy (ADR 0061).

## Why

Three weaknesses were found after run 2:

1. **The network never sees its own envido total.** It gets each card's suit and tanto digit and must rebuild "same suit + 20" by itself, while Adaptive Utility uses the exact total. Accepted envidos are where run 1 lost the most points against it.
2. **Every decision gets the whole match's result as its target.** One envido at 0-0 in a 30-point match barely changes who wins, so its signal is buried in noise. Run 2's training loss stopped improving at around 0.73 from generation 13.
3. **Sampling at T = 0.05 threw games away.** The final exam's Part T showed that generation 38 wins 73% against Adaptive Utility when greedy, against 52% at T = 0.05. The value ranking was good; the sampling was not.

## Decisions

- **New inputs** (zero-initialised columns, so the warm-started network starts unchanged):
  1. own envido total from the original three cards, normalised;
  2. a flush-pair bit;
  3. the truco strength percentile of the three-card hand, from a standalone table computed from the rules over every hand, never from Adaptive Utility code;
  4. the exact percentile of the cards still unplayed, zero when none remain;
  5. points at stake for envido and real envido (falta is already an input);
  6. the probability that the own tanto beats the opponent's, from the exact-filter beliefs, only if it slows training by less than 10%.
- **Target:** 0.5 × match result (+1/−1) + 0.5 × match equity after the hand. Equity comes from a fixed table, P(win | scores after the hand, target 15 or 30, who is mano), built from self-play of the warm-start network played greedy. It is read at the state after the hand's points, so they are not counted twice, and sparse cells are smoothed. Before use, the table must rise with the own score, and must predict held-out results better than the raw score difference (Brier score).
- **Warm start** from the shipped network (run 2 generation 38).
- **League and panel:** unchanged from run 2 (ADRs 0056, 0057). Phase 1 generation 48 and the two held-out networks stay out of training and selection, so the exam can still detect specialisation.
- **Action selection:** training rollouts keep T = 0.05 for exploration. Promotion checks and the panel play challenger and champion greedy, so selection measures the deployed policy. The trainer logs how often the sampled action differs from the greedy one.
- **Pilot before the night:** about 1,000 updates from the warm start with all changes, then 300 greedy mirror pairs against fixed Adaptive Utility on separate seeds. Below 68% (the exam measured 71.5%), the learning rate is lowered and the pilot repeated; the night starts only after a pass.
- **Budget:** 12 hours or 40 checks without promotion (ADR 0044). Speed-ups are accepted only if a fixed-seed golden run gives exactly the same matches, samples and weights.
- **Judgement:** the ADR 0061 exam with fresh seeds. Candidates are run 3's champion, then its last checked generation; the baseline is the shipped network played greedy.

## Considered Options

- **The network's own value as the equity (bootstrap):** elegant, but it learns from its own errors and can drift.
- **Pure equity target:** lowest variance, but any error in the table would become the network's objective. The blend keeps the real result as an anchor.
- **Training greedy or at a lower temperature:** narrows exploration; the advisor recommended keeping T = 0.05 or a mild epsilon-greedy.
- **Starting from scratch:** cleaner to explain, but throws away twelve hours of learning.
- **Some randomness against humans:** less predictable, but it cost more than 20 points against Adaptive Utility. Rejected for strength; it can be revisited if people find the network easy to read.
