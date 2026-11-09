# The supervised inference network is built in phase 2, as an auxiliary output

In phase 1 the decision network uses only the bayesian module. Supervised prediction of the opponent's cards (ADR 0010) is added in phase 2 as an auxiliary output of the same network, trained together with the decision. Doing it in phase 1 would have meant hand-writing gradient computation in numpy; in phase 2, PyTorch provides it. Training both together also tends to improve the decision, and it is standard in DouZero-style card AIs.

The evolution page compares three predictions of the opponent's cards on the validation deals: exact filter only, inference weighted by bluff rate (ADR 0010), and the learned output. It shows how much a network that learned to read the opponent adds over a formula.

## Amendment (2026-11-08)

The phase 2 network and auxiliary head are specified in ADR 0040; the head is optional and kept only if it measurably helps play.

## Amendment 2026-11-08: phase 2 card prediction

The auxiliary card head is off for the measuring and real phase 2 runs. The measured head-off throughput is approximately twice head-on throughput, potentially allowing twice as many training matches in eight hours; there is no evidence that the auxiliary task compensates for that cost, and DouZero does not use this head. The implementation remains available as an option. The evolution page’s three-way card comparison will instead use a post-training linear probe on the frozen final champion, compared against the same probe on an untrained network and labelled “post-training probe”; this is future work.
