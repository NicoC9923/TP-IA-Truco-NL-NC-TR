# Phase 1 frozen-population ranking reliability

Measured the 64-individual population saved by seed 1 at generation 70, with its two hall entries, reigning champion, checkpoint fitness settings, and styled league. Each of three splits used two independent shared league schedules at each pair count; an independent 64-pair league measurement supplied reference fitness. The top-eight reference score averages the two selected sets per split. The 12-worker run took 668.94 seconds (11.15 minutes).

| Mirror pairs | Split Spearman correlations | Mean Spearman | Top-eight overlaps | Mean reference fitness of top eight |
| --- | --- | ---: | --- | ---: |
| 16 | 0.0952, 0.1042, 0.2142 | 0.1379 | 1, 0, 3 | 0.525098 |
| 32 | -0.3114, 0.0973, -0.1657 | -0.1266 | 0, 1, 0 | 0.526133 |

The pre-registered rule requires a mean correlation increase of at least 0.10 and a higher mean top-eight reference fitness at 32 pairs. The reference score is slightly higher, but the correlation difference is -0.2645, so the verdict is **do not proceed** to a 32-pair training run under this rule.

This is one frozen population with three splits; the reference is itself noisy and league schedules differ between measurements. These findings do not establish general performance across populations or training seeds. The JSON records the full reference scores and seed namespace.
