# Phase 2 encoder layout and decision cost

Status: accepted (2026-11-08).

The phase 2 encoder uses the index map documented in `proyecto/phase2_network.py`: 64 v3-core values, four profile confidence counts, two buenas flags, 120 own-card identity values, 12 legal-action bits, 24 event rows of 59 values, and five exact plus five weighted belief values, for 1,628 float32 values. A 15-point match is treated as in buenas for both players because falta envido is played for the match; in a 30-point match each player's flag starts at 15 points. This preserves the distinction relevant to falta envido.

The symmetric engine permits at most 17 events in a completed hand: six card plays, three truco calls and three responses, and four envido calls and one response. The encoder reserves the 24 rows selected in ADR 0040 and raises on overflow. A card event records the public table card in place of its private slot, and accepted envido declarations are attached to the resolution event.

The policy maintains separate incremental exact and weighted posteriors and resets them after each hand. A default styled-opponent offset of 0.1 Q units is small relative to the match-result target; temperature remains the fixed pilot value from ADR 0041. The offset can be tuned before training.

A small warm CPU loop, one PyTorch thread, measured 5.63 ms per decision with weighted beliefs and 2.93 ms with exact-only beliefs (25 calls each, encode plus forward and sampling). A separate 100-call breakdown measured encoding at 5.576 ms and forward at 0.040 ms with weighting, versus 2.819 ms encoding and 0.039 ms forward without it. These are local planning measurements, not a training throughput benchmark: posterior summaries dominate and full matches, worker startup and learning updates will affect ticket #31 throughput.
