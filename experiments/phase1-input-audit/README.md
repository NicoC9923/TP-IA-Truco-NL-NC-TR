# Phase 1 input audit

This diagnostic evaluates the three reigning champions and 16 fixed, uniformly sampled members of the frozen seed 1 generation 70 population. The observation bank contains 2,782 decisions, including 2,000 card-play decisions, from 128 scheduled population matches. The audit used 12 workers, no training, a fresh default opponent profile per observation, and the reserved odd seed namespace `2*(10^20 + match_index)+1`; the run took 26.39 seconds. The full per-network measurements, seeds, hashes, sample counts, intervals and alias results are in [results.json](results.json). The decision rules were committed beforehand in [ADR 0034](../../docs/adr/0034-phase1-input-audit.md).

```text
python proyecto/input_audit.py --workers 12 --output experiments/phase1-input-audit/results.json
```

The run set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The checkpoint path is recorded in the JSON; sampled population indices were 24, 4, 39, 38, 11, 48, 55, 3, 62, 63, 35, 34, 54, 13, 2 and 58. The two seed families select independent streams, not disjoint deals.

## Results

- **H1: supported for 16 of 19 networks by the preregistered probe rule.** On 600 decisions with a known opponent card and a winning card, mean probability mass across networks was 51.79% on a winning card, 35.63% on a cheapest winning card and 16.15% on a stronger winning card. The mean is over networks, and mass is over all legal actions, including calls and folding. There were 162 order-agreeing and 390 order-disagreeing winning-versus-losing card pairs; no network satisfied the separate twofold error-ratio rule. Ratios ranged from 0.908 to 1.154, while the mean probe correlations were 0.233 with face value and 0.052 with Truco rank.
- **H2: supported for 11 of 19 networks.** There were 252 decisions where the first available card was uniquely better by at least 0.3, and 324 where another card was. Across networks, the mean card-restricted argmax rates for the better card were 58.42% and 29.95% respectively. This comparison is descriptive: card-slot preferences and differences between the two groups can also contribute to the gap.
- **H3: not supported.** Zero of 2,782 observations had a strategically different exact encoding or one within the preregistered L2 radius of 0.01. This result depends on the sampled bank, selected facts and radius.
- **Mistakes:** there were three opportunities for rejecting or folding a conservatively guaranteed hand and 18 for declining envido with 31–33 points. Mean legal-action probability mass across networks was 7.09% and 5.09%, respectively; the shipped champion assigned 22.87% and 15.43%. There were zero opportunities for accepting an envido already known to be lost or for choosing a losing card over a winning one in the last trick, so neither rate is estimated.

| Reigning champion | H1 | H2 | Winning mass | Cheapest winning mass | H2 first / other argmax | Face / rank correlation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Shipped | No | Yes | 47.58% | 31.83% | 86.51% / 8.64% | -0.280 / 0.108 |
| Seed 1 | Yes | No | 39.19% | 26.88% | 20.24% / 57.10% | 0.163 / -0.004 |
| 53% gate seed 1 | Yes | No | 54.87% | 36.42% | 9.92% / 63.27% | -0.062 / -0.088 |

## Proposed change

Following the preregistered priority, replace each held card's normalized face-value input with its normalized Truco rank. This stays within **64 inputs**, but changes the meaning of existing inputs and therefore breaks compatibility with current genomes and panel-v1. It would require from-scratch training, a new ADR and possibly a spec #15 change; the audit does not implement it. H2 also fired for some networks, but the preregistration selects one proposal.

## Limits

The H1 probe changes card identity while keeping the rest of the encoding, including inference summaries, fixed; some resulting inputs contain impossible duplicate cards. The H1 support rule only compares correlations and may fire even when both correlations are small or negative; its result is not evidence of a twofold card-choice error. H2 mixes exact outcomes against a played card with inference otherwise, and the latter assumes the opponent chooses uniformly among remaining cards; strict wins exclude ties. Bank decisions are correlated within hands and matches, so the decision-bootstrap and normal confidence intervals can overstate precision. The guaranteed-hand classifier is sufficient rather than exhaustive, high-envido declines are a heuristic rather than a universally proven mistake, and declaration timing explains why known-losing accepted envido may be unobservable. In the last trick each seat normally has only one card left, leaving no choice between winning and losing cards. These descriptive results do not establish causality or prove that a proposed input will improve training.
