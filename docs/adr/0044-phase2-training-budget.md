# Phase 2 training budget: a measuring run first, then up to 8 hours

Status: accepted (user decision, 2026-11-08). Numbers are fixed after the measuring run and recorded in an amendment here before the real run.

1. **Measuring run** (~20–30 min, asked for first, 12 workers): matches per second, learner samples per second, duration of one generation check, the fixed softmax temperature (ADR 0041) and the profile prior (ADR 0045). Its final snapshot becomes the fourth selection anchor (ADR 0047).
2. **Real run:** the number of training matches per generation is chosen so one generation takes about 5–10 minutes and checks use at most 15% of the time. The run ends at the first of **8 hours** of training or **40 consecutive checks without a promotion**. The stop reason is logged. Pilot starting values from the advisor: 512-sample minibatches, Adam with learning rate 3e-4, fresh data used once (no stale replay).

Why: in reinforcement learning a network often improves gradually without beating the champion for a while, so stopping after spec #15's 20 failed checks could cut a run off while it is still learning. The 40-check limit is a safety net so a run that is truly stuck doesn't use the whole night. Reaching the time cap does not mean the network has converged; useful learning may take more than one night.

## Considered Options

- **20 checks without promotion (spec #15):** too early for gradual RL progress.
- **Time cap only (advisor):** the safety net costs nothing when progress is real.
- **~3 hours while the PC is in use:** rejected in favor of overnight runs.

## Measuring-run amendment 2026-11-08

The preregistered measuring rule selects **57,120 training matches per generation**, estimating 450.4 seconds per generation and 2.6% check overhead. The eight-minute head-off run measured 24.73 whole-run matches/s and 327.69 samples/s; pre-check training throughput was 130.24 matches/s, while the separate benchmark estimated 11.79 seconds/check and 10.13 ms/update. These figures come from an undertrained network and early short matches; longer future matches could reduce throughput and increase generation time. See the [measuring record](../../experiments/phase2-measuring-run/README.md).
