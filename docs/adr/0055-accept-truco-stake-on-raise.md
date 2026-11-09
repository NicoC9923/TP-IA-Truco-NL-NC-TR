# 0055. Accept the lower truco stake when raising

A retruco in response to truco accepts the two-point stake; vale cuatro in response to retruco accepts the three-point stake. The raising player holds the raise right for that accepted level until the opponent responds. Envido may interrupt the first pending truco before a trick, but cannot be introduced after either raise.

The symmetric training engine previously left the accepted stake at its old value on a raise. It consequently exposed illegal envido actions and incorrect observations, and could score the resulting interruption incorrectly. With the current phase 2 network, the illegal envido was offered in 0.7% of decisions and selected in 0.26%, approximately once per 14 matches.

Phase 1 results and the aborted phase 2 run 1 were produced with this bug. The phase 2 re-run uses the fixed engine. Historical results are retained as records of the engine used at the time.
