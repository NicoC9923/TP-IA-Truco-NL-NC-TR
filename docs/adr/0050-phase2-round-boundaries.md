# Phase 2 round boundaries and sample storage

Status: accepted (2026-11-08).

A generation rounds its requested match count up to a whole number of synchronous rounds. This preserves a fixed round size and makes generation boundaries checkpoint boundaries; the log records the matches actually played. Each learner sample contains the float32 encoded observation, the chosen action index and the final match result from that decision's seat; the seat is retained for inspection. Both seats contribute in self-play, and samples are consumed once in a deterministic shuffled order.

An interruption during a round discards its unfinished work: resume starts from the last completed round checkpoint, truncating later generation log entries. A checkpoint is atomically replaced after its round and includes partial-generation statistics, so an interruption between generations does not change semantic results.

Phase 2 keeps ADR 0016's mixture of targets 15 and 30 and 30% random legal starting scores by default; both the target and scores are drawn from the run seed and logical match id. The target and random-start share are configurable, making small tests practical without changing the default distribution.
