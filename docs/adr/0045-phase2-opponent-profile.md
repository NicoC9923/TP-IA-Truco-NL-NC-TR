# Phase 2 opponent profile: rates over opportunities, prior tuned in the pilot

Status: accepted (user decision, 2026-11-08). Phase 1 profile semantics are unchanged (per-policy semantics, ADR 0036).

Phase 2 keeps the four observed rates (bluff, fold, accept, envido) and adds their confidence as inputs (ADR 0040), with these changes, computed by the same code in training and in the game (spec #15 decision 7):

- **Rates are counted over opportunities.** The fold rate is folds over the times the opponent could fold, not over all hands; an opponent who was never raised has no fold evidence, not a low fold rate. The same applies to accept and envido.
- **The prior is a pilot value.** It starts at the current neutral prior worth ~20 observations (spec #15 decision 9) and is compared in the measuring run against weaker priors and recency-weighted counts on held-out prediction and play; the choice is recorded before the real run.
- **Bluffs are still counted only on revealed hands** with the objective criterion (envido under 24 tantos, truco under the 0.44 percentile; ADR 0011). The rate means "bluffs that were caught", not the opponent's true bluff frequency, because which hands get revealed depends on play.

Why: the profile is what lets the network adapt to the opponent; counting non-opportunities as reluctance, or an arbitrary prior, feeds it misleading numbers.

## Measuring-run amendment 2026-11-08

The preregistered held-out comparison selects neutral **prior strength 5** for the real run: mean log loss was 0.591 for 5, 0.623 for 10 and 0.651 for 20. Only 15 held-out opportunities were scored from nine short replays of an undertrained eight-minute network, with none for revealed bluffs; this is weak evidence for a stable choice. See the [measuring record](../../experiments/phase2-measuring-run/README.md).
