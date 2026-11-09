# 0055. Accept the lower truco stake when raising

A retruco in response to truco accepts the two-point stake; vale cuatro in response to retruco accepts the three-point stake. The raising player holds the raise right for that accepted level until the opponent responds. Envido may interrupt the first pending truco before a trick, but cannot be introduced after either raise.

The symmetric training engine previously left the accepted stake at its old value on a raise. It consequently exposed illegal envido actions and incorrect observations, and could score the resulting interruption incorrectly. With the current phase 2 network, the illegal envido was offered in 0.7% of decisions and selected in 0.26%, approximately once per 14 matches.

Phase 1 results and the aborted phase 2 run 1 were produced with this bug. The phase 2 re-run uses the fixed engine. Historical results are retained as records of the engine used at the time.

## Training integrity and recovery

Phase 2 training deals use canonical partition 1; selection, overlap and card-validation deals use partition 0, matching phase 1's convention. Different seeds alone do not prevent the same deal from recurring. See ADR 0004.

The promotion interval clusters shared deals: average the challenger-minus-champion differences across panel members for each pair, then calculate the standard error across pairs. The gain and direct thresholds are unchanged. See ADR 0047.

Champion weights and metadata are each written to a temporary file and atomically replaced. On resume, the export is reconciled with the champion stored in the restored checkpoint, including after an interruption between export and checkpoint. The phase 2 checkpoint configuration identifies the engine/observation contract and encoder versions; checkpoints without matching versions, including older checkpoints without those keys, cannot resume. A round with no learner samples advances without an optimizer update, and invalid numeric run settings are rejected before output creation.

## Confirmed training-engine rule decisions

- Truco and raises may be called only on the caller's own card turn, or as an answer to a pending bet. Calling between card turns, which the human game allows, is not modeled.
- "El envido va primero" is allowed as an answer to the first truco, even after the answering player has played their first card, as long as no trick is complete. This matches the human game.
