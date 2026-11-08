# Phase 1 frozen champion round robin

The [pre-registered rule](RULE.txt) was written before the run. The [runner](run_rr.py) and [results](results.json) preserve a network-only round robin: 28 pairs, each playing 1,000 mirror pairs on a fresh validation schedule seed from 9301 through 9328. Margins below are paired 95% margins in percentage points.

## Frozen networks

| Network | Encoder | SHA-256 | Mean win rate over seven opponents |
| --- | --- | --- | ---: |
| gate53-g26 | v1 | `b00e64118ce54e8063b7eb61c8085ec6093f732c00112d61a26a608ee666f5fd` | 52.56% |
| v2s1-g46 | v2 | `9659d27a3c70f369751d48a9ba4d97f555f4d8cf20931cbbd23f06c90ecd7fe6` | 51.75% |
| v2s2-g25 | v2 | `3f25fb0b59b5c7a10c4666bb2450717f822ad6d13c8283a217069645a7f68a06` | 51.14% |
| v3s2-g50 | v3 | `6664e68379ef90db275ab6432b2a2b3df3562104bfe24a239e671452d77bae38` | 51.13% |
| v3s1-g55 | v3 | `24773046ba6753f1f13614fda74d15a31b1756c2976a76cc6eb384504a311763` | 50.16% |
| v1s1-g23 | v1 | `413542520b893418371fa0ff29b4b693d0675ef60cd4be55f8e782dd544db31c` | 49.81% |
| run1-g48 | v1 | `0b3e04fad09334f60d2d416b9c6848c7b3aabcfa6bb0d6593ac825feb32e47f7` | 48.14% |
| v1s2-g2 | v1 | `5f00bfbaa27d7ee9fcedd7610ca36b092c71b8e935218a8f337a7cc1e25a3e79` | 45.31% |

## Pair results

Win rate is for the first network.

| First | Second | Schedule seed | Win rate ± margin |
| --- | --- | ---: | ---: |
| run1-g48 | v1s1-g23 | 9301 | 48.15% ± 1.96 |
| run1-g48 | v1s2-g2 | 9302 | 53.40% ± 1.97 |
| run1-g48 | gate53-g26 | 9303 | 45.65% ± 1.93 |
| run1-g48 | v2s1-g46 | 9304 | 45.10% ± 2.00 |
| run1-g48 | v2s2-g25 | 9305 | 48.85% ± 1.73 |
| run1-g48 | v3s1-g55 | 9306 | 48.30% ± 1.80 |
| run1-g48 | v3s2-g50 | 9307 | 47.50% ± 1.86 |
| v1s1-g23 | v1s2-g2 | 9308 | 55.10% ± 1.99 |
| v1s1-g23 | gate53-g26 | 9309 | 48.20% ± 1.98 |
| v1s1-g23 | v2s1-g46 | 9310 | 47.70% ± 2.05 |
| v1s1-g23 | v2s2-g25 | 9311 | 47.60% ± 1.83 |
| v1s1-g23 | v3s1-g55 | 9312 | 50.60% ± 1.98 |
| v1s1-g23 | v3s2-g50 | 9313 | 47.65% ± 1.91 |
| v1s2-g2 | gate53-g26 | 9314 | 43.55% ± 1.93 |
| v1s2-g2 | v2s1-g46 | 9315 | 43.80% ± 2.03 |
| v1s2-g2 | v2s2-g25 | 9316 | 45.65% ± 1.86 |
| v1s2-g2 | v3s1-g55 | 9317 | 45.95% ± 1.93 |
| v1s2-g2 | v3s2-g50 | 9318 | 46.70% ± 1.90 |
| gate53-g26 | v2s1-g46 | 9319 | 52.05% ± 2.13 |
| gate53-g26 | v2s2-g25 | 9320 | 49.85% ± 1.76 |
| gate53-g26 | v3s1-g55 | 9321 | 52.10% ± 1.96 |
| gate53-g26 | v3s2-g50 | 9322 | 51.30% ± 1.89 |
| v2s1-g46 | v2s2-g25 | 9323 | 50.85% ± 1.84 |
| v2s1-g46 | v3s1-g55 | 9324 | 49.95% ± 1.99 |
| v2s1-g46 | v3s2-g50 | 9325 | 50.10% ± 1.96 |
| v2s2-g25 | v3s1-g55 | 9326 | 50.85% ± 1.73 |
| v2s2-g25 | v3s2-g50 | 9327 | 49.95% ± 1.63 |
| v3s1-g55 | v3s2-g50 | 9328 | 48.90% ± 1.85 |

There are **no clear cycles**: no three-network cycle has all edges with a 95% lower bound above 50%. This gives no evidence of non-transitive (rock-paper-scissors) strength among these champions; it does not rule it out. Newer champions cluster near 50% against each other. Older v1 networks, including shipped run1-g48, sit below them overall: run1-g48 scored 45.10% ± 2.00 against v2s1-g46 and 45.65% ± 1.93 against gate53-g26. No challenger reaches the 55% shipping rule against run1-g48 in these results. This is descriptive only: nothing was selected or replaced.
