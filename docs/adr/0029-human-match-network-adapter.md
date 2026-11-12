# Project the human match onto the shared policy observation

The human match continues to own its existing rules, gestures and reveal flow. At each network decision it projects the visible match state into the symmetric engine's observation and legal-action vocabulary, then applies the selected action through the existing match methods. The projection contains the robot's cards and public evidence only; the human's cards enter the profile's bluff calculation only when all three have been played. A single local JSON file stores the same smoothed profile counts used in training, and an atomic replacement saves each completed hand. The adaptation switch supplies either those counts or a neutral profile to both the encoder and weighted inference.

The phase 2 mode remains unavailable until its policy implementation and champion format exist, even if an arbitrary file appears in its directory. This keeps the game independent of PyTorch and avoids claiming that a phase 2 champion can be loaded before that implementation exists.

## Considered options

- Replace the human match with the symmetric engine: this duplicates or disrupts the established robot, reveal and UI integration and risks changing game behavior.
- Implement a second network or inference path for the human match: this risks training and play interpreting the same observations differently.
- Store profile rates alone: counts are needed to preserve the prior and weight future observations correctly.

## Field practice

Adapters translate a legacy application's state into a shared domain interface while keeping policy inference separate from presentation and device effects. Persisting sufficient statistics preserves online estimates without keeping a full event history; atomic file replacement prevents an interrupted write from leaving a partially written profile.

A request to select a recognized but unavailable agent mode returns HTTP 409 and leaves the active mode unchanged. Unknown mode codes return HTTP 400.
