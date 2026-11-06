# The supervised inference network is built in phase 2, as an auxiliary output

In phase 1 the decision network uses only the bayesian module. Supervised prediction of the opponent's cards (ADR 0010) is added in phase 2 as an auxiliary output of the same network, trained together with the decision. Doing it in phase 1 would have meant hand-writing gradient computation in numpy; in phase 2, PyTorch provides it. Training both together also tends to improve the decision, and it is standard in DouZero-style card AIs.

The evolution page compares three predictions of the opponent's cards on the validation deals: exact filter only, inference weighted by bluff rate (ADR 0010), and the learned output. It shows how much a network that learned to read the opponent adds over a formula.
