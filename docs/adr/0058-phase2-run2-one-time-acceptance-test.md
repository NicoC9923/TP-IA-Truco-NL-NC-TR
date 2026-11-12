# Phase 2 run 2 ships only if it passes a one-time acceptance test

Status: accepted (user decision, 2026-11-09).

Run 1's champion, generation 28, is the baseline. Run 2's champion replaces it in the app only if it passes a test whose rule is written in `experiments/phase2-run2/RULE.txt` before training starts. The test runs once: no retries, no picking among checkpoints.

## The test

All matches use validation deals (partition 0) and fresh, pre-registered seeds.

1. **Direct match:** run 2 champion vs gen28, 1,000 mirror pairs.
2. **Held-out check:** run 2 champion and gen28 each play the same 1,000 mirror pairs against each of the two held-out networks (ADR 0060), half to 15 points and half to 30. The score is the paired difference, run 2 minus gen28, averaged with equal weight over the two networks.

**Run 2 ships only if both hold:**

- the direct match's 95% lower bound is above 50%;
- the held-out difference's 95% lower bound is above −3 points.

The first condition asks for a clear win. The second only asks that run 2 is not clearly worse against networks it never saw.

## Reporting

- **Ships:** run 2 becomes the new phase 2 champion.
- **Not shown better:** gen28 stays, and the result is reported as inconclusive, not as "run 2 is worse".
- **Clear regression** (the interval lies entirely below the threshold): gen28 stays, and the result is reported as a regression.

This test decides shipping. It is not an unbiased estimate of run 2's strength, because the champion was itself chosen by selection.

## Considered Options

- **Accept the trainer's own promotions as proof:** the panel is a development yardstick, and run 1 showed that its late gains did not carry over.
- **Re-run or extend the test on a close result:** that is shopping for a pass. Rejected.
