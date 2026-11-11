# Phase 2 run 3 and its final exam

## Run 3

Run 3 followed its frozen [rule](../phase2-run3/RULE.txt) (ADR 0062 and amendments). It warm-started from the shipped run 2 generation 38 and added six explicit hand-fact inputs. Its target blended the match result 50/50 with match equity, from a fixed [table](../phase2-run3/equity-table.json). Rollouts were sampled at T = 0.02, and checks played greedy with 300 pairs per panel member. It ran 43,200 s of active training (12-hour cap; `done.json` train 0): 12,487,920 matches and 100 checked generations, with generation 101 cut by the cap. It was moved once into an Orca terminal by deterministic resume (see the operational note in the rule). Promotions: generations 2, 11, 19, 22, 56, 61 and 73. The run champion is generation 73. Pilots and their go/no-go are in [pilots/](../phase2-run3/pilots/).

## Exam

The exam followed the frozen [rule](RULE.txt) and [config](config.json) (ADR 0061). An independent advisor approved it before play, under the user's standing authorization. It ran once with 12 workers; the raw records and results are in [part-e/](part-e/). The coordinator independently recomputed every gain and lower bound from `part-e/partial.jsonl`: 5,000 unique records, 1,000 unique deal seeds per opponent and 500 pairs at each target.

All three networks played greedy. Each cell is 1,000 mirror pairs on shared deals and action seeds. Gains are paired differences against the baseline, with 95% intervals.

| Opponent | g38 (baseline) | Run 3 g73 (C1) | Gain C1 | Run 3 g100 (C2) | Gain C2 |
|---|---:|---:|---:|---:|---:|
| Adaptive Utility (fixed) | 72.2% | 85.8% | +13.7 [11.6, 15.7] | 85.0% | +12.8 [10.8, 14.9] |
| Reactive | 99.5% | 99.7% | +0.1 [−0.3, 0.6] | 100.0% | +0.4 [0.1, 0.8] |
| Phase 1 g48 | 92.2% | 92.6% | +0.4 [−1.1, 1.8] | 91.9% | −0.4 [−1.8, 1.1] |
| Held-out v1s1-g23 | 92.2% | 92.2% | +0.0 [−1.4, 1.4] | 91.4% | −0.8 [−2.2, 0.7] |
| Held-out v1s2-g2 | 94.0% | 90.3% | **−3.7 [−5.1, −2.3]** | 89.0% | **−5.0 [−6.5, −3.5]** |

Both candidates pass the primary test against Adaptive Utility (Holm) and the balance test, with mean gains of +2.1 and +1.4 points. Both **fail the guardrail**: against held-out v1s2-g2, the lower bound is below −2 points for both. C2 also misses it narrowly against v1s1-g23. By the rule, **run 2 generation 38 stays shipped**. The exam is not repeated for these candidates.

## Reading

- **Run 3 learned to beat the opponent it trained against.** Against fixed Adaptive Utility, a run 3 training opponent, it rose from 72% to 86%. Against Reactive and the phase 1 networks it was essentially unchanged, and it got worse against v1s2-g2, a generation-2 phase 1 network never seen in training or selection. This is the specialisation the guardrails exist to catch.
- The loss is against a weak, unusual network rather than a strong one. It still shows that run 3's changes do not transfer evenly to unseen styles of play. Which change caused it (inputs, equity target, temperature or check size) is not identified, because run 3 changed several at once.
- These results describe this protocol and these opponents. They do not establish general strength. Raw checkpoints are archived outside Git.
