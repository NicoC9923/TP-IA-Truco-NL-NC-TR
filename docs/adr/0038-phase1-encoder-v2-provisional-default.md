# Phase 1 encoder v2 as the provisional experiment default

Status: accepted provisionally (user decision, 2026-11-08, after advisor review).

## Context

ADR 0035 introduced v2 as an experiment, changing held-card inputs to Truco strength. ADR 0036 introduced v3-core, a bundle of additional input and accounting changes. The frozen network panel and pre-registered experiment criteria in ADR 0031 assess development evidence separately from the shipping rule.

## Decision

Future phase 1 experiments use `--encoder v2` unless they explicitly test another encoding. The trainer's CLI default remains v1 for backward compatibility; this decision makes no code change. The shipped champion remains run 1 generation 48, using v1, and game #23 remains unchanged. The 55% shipping rule remains unchanged.

## Evidence and limits

V2 versus v1 was promising under the pre-registered panel rules on two training seeds. For seed 1, the direct result was 53.79% and the panel gain was 2.07 percentage points on reused schedule seeds; the panel result was re-confirmed on fresh schedules in the v3 comparison. For seed 2, the direct result was 54.69% ± 0.92 percentage points and the panel gain was 4.21 points on fresh schedule seeds 9211 and 9212. See the [v2 seed 1 record](../../experiments/phase1-encoder-v2-seed1/README.md), [v3 seed 1 record](../../experiments/phase1-encoder-v3-seed1/README.md), and [matched seed 2 record](../../experiments/phase1-seed2-matched/README.md). Distinct schedule seeds do not guarantee disjoint individual deals.

The [matched seed 3 comparison](../../experiments/phase1-seed3-matched/README.md) found no demonstrated v2 benefit: direct 50.29% ± 0.96 and panel difference −0.14 points. V2 was promising on two of three seeds; the decision remains provisional.

This adoption is provisional: three training seeds are limited evidence, and the repeatedly used panel is development evidence, not a shipping test. Reference opponents are descriptive only under ADR 0021 and disagree with the panel in places. V2 retains the confirmed B1 pending-envido stakes defect and B2 opponent-profile counting defect, both corrected in v3 (ADR 0036). V3-core bundles changes, so individual effects are unknown; it showed no demonstrated additional panel benefit over v2: seed 1 was worse, and the seed 2 loss did not replicate. This is not evidence of equivalence.

The separate shipping tests did not reach 55%: v2 seed 1 generation 46 scored 52.5% ± 1.0 against the shipped champion on a fresh test. The seed 2 panel comparison against that champion was also below 55% and was not a shipping test. These outcomes do not establish a generally strongest player.

## Considered options and revision

- **Adopt v3:** its additional panel benefit has not been demonstrated; its correctness fixes remain relevant for future investigation.
- **Replace the shipped champion:** the existing 55% shipping criterion has not been met.
- **Change the shipping rule after seeing these results:** this would require an explicit prospective policy revision, not a retroactive success.

Revisit this decision if a v3 component benefit is replicated, or if a panel-based shipping rule is prospectively registered and adopted. A future shipping policy must remain distinct from the experiment panel decision and respect ADR 0021.
