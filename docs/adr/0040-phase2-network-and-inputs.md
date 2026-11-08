# Phase 2 network: one MLP over the current hand's ordered events

Status: accepted (user decision, 2026-11-08). Sizes are implementation targets to verify, not tuned values.

## Inputs (about 1,700 numbers)

- **Summary block:** the corrected v3-core features (ADR 0036), not v2. Phase 2 starts from scratch, so it is not bound to phase 1's provisional v2 default (ADR 0038), and v3-core fixes bugs B1 (pending envido stakes) and B2 (envido interruptions counted as truco resolutions). Added: profile-confidence features (how many observations back each rate) and one "in the buenas" flag per player, because points remaining alone cannot tell a 15-point match from a 30-point one, and falta envido pays differently.
- **Own cards by identity:** three 40-card one-hot slots, aligned with the engine's card slots, kept alongside the rank features. Identity captures suits (envido) and exact cards; rank gives the ordering directly.
- **Legal actions:** 12 bits.
- **Ordered events of the current hand:** up to 24 events (the implementer verifies the engine's maximum and rejects overflow instead of truncating), each with relative actor, action category, public card identity, trick index, a presence bit and declared tantos / "son buenas". The opponent's card events carry the public card, never the private slot it came from. Earlier hands reach the network only through the opponent profile (ADR 0045).
- **Beliefs about the opponent's cards:** both the exact-filter-only summaries and the bluff-rate-weighted summaries (ADR 0010, 0024), with a switch that drops the weighted ones. The weighting is a hand-written formula that may misread strategic opponents; the network should be able to learn to discount it.

Nothing the player could not know at that moment enters: no style label, no opponent identity, no true opponent cards.

## Network

One shared MLP, about 1,700 → 256 → 128 → 12, ReLU, float32, roughly 470,000 weights (about 2 MB, within ADR 0018's "a few MB"). The 12 outputs are Monte-Carlo action values Q(state, action), one per action slot, with a legal-action mask. Outputs are linear (no tanh) because saturation slows the correction of confident errors; out-of-range values are logged. The final layer starts small so first choices are near uniform. No batch normalization or dropout.

**Target:** the match result from the acting seat's point of view, +1 win / −1 loss, discount 1, mean squared error on the action actually taken. Points per hand are not the target: what is right depends on the score (protect a 14–5 lead, gamble at 3–13, falta envido decides the match), and maximizing hand points ignores that.

**Auxiliary card head (ticket #25, ADR 0019):** a 40-logit branch from the shared trunk. It scores every three-card combination compatible with the exact filter (so joint evidence such as declared tantos is respected, not only card-by-card exclusion), normalizes over them and derives per-card marginals. Labels are the real cards from simulation, including folded hands; they never update the observable profile. Loss weight starts around 0.1. The head can be turned off, and it stays only if it measurably helps play. Its prediction is not fed back as an input. Accuracy is reported separately for cards still hidden, so exposed cards don't inflate it.

## Considered Options

- **Reuse a phase 1 encoder alone (64 inputs):** cheap, but it compresses calls into counts and loses order, which is exactly the behavioral signal phase 2 should read.
- **Raw history of earlier hands:** much larger input; the profile already summarizes earlier hands, and the CPU budget is unlikely to learn more from raw history.
- **DouZero's action-as-input scoring:** needed for DouDizhu's thousands of actions; Truco has 12 fixed actions, so one pass with 12 outputs is cheaper and equally valid.
- **LSTM over events (DouZero) or a small attention block:** the current hand has at most a few dozen short events, and recurrent layers are slow on CPU. Revisit only if an order-sensitive probe fails, compared at equal wall time.
- **Separate networks or heads for envido, truco and card play:** waste samples; the decisions share facts such as hand strength.
