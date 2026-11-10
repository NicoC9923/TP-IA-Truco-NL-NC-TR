# Four-mode round robin

One pre-registered descriptive run compared Reactive, fixed Adaptive Utility, shipped phase 1 run 1 generation 48, and shipped phase 2 run 1 generation 28. The phase 2 run 2 candidate did not meet its separate frozen acceptance rule; generation 28 remained shipped. See [RULE.txt](RULE.txt) for identities, policy settings, seed audit, construction and limitations.

Each row reports the first mode's match win rate across 1,000 mirror pairs (2,000 games), and a normal 95% interval computed from the sample standard deviation of the 1,000 pair scores. There were 500 pairs at target 15 and 500 at target 30 per row; intervals are unadjusted for multiple comparisons.

| First mode | Opponent | Win rate | 95% interval |
|---|---|---:|---:|
| Reactive | Fixed Adaptive Utility | 16.85% | [15.36%, 18.34%] |
| Reactive | Phase 1 | 54.45% | [52.31%, 56.59%] |
| Reactive | Phase 2 | 31.85% | [29.99%, 33.71%] |
| Fixed Adaptive Utility | Phase 1 | 79.70% | [77.99%, 81.41%] |
| Fixed Adaptive Utility | Phase 2 | 63.05% | [61.26%, 64.84%] |
| Phase 1 | Phase 2 | 21.40% | [19.77%, 23.03%] |

These are familiar-opponent, descriptive results: phase 2 training included phase 1 networks and both reference modes. They do not establish independent general strength or isolate the effect of the training algorithms. The pairwise intervals describe this protocol and opponent set; they are not simultaneous confidence intervals. No champion selection, retry or extension was performed.

The runner exited 0 and wrote the completion marker. An independent verification recalculated all six intervals from the raw seat winners, checked 6,000 unique indexed pairs, 6,000 unique deal seeds, 24,000 unique action seeds, target counts, all recorded source/model/sidecar hashes and the raw file hash; see [verification.json](verification.json). The eight synthetic pre-run protocol tests passed, including fresh construction, mirrored scoring, persistence, interrupted-run behavior and seed bounds. The run's configuration and full raw pair records are in [config.json](config.json), [results.json](results.json), and [partial.jsonl](partial.jsonl). The lock is retained to prevent an accidental second run. Logs and an independently hash-verified copy of the full evidence are archived in `E:/Tools/GitHub-Repositories/TP-IA-Truco-training-archive/four-mode-round-robin`.
