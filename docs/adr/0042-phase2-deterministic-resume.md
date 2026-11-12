# Phase 2 resume is bit-identical at round boundaries

Status: accepted (user decision, 2026-11-08).

Phase 2 collects experience in synchronous rounds: 12 single-thread actor processes play a fixed batch of matches with frozen weights, results are merged in a fixed order, then one learner updates and broadcasts the new weights. Every match is seeded by its logical id. Checkpoints are written only between rounds and hold the model, optimizer, counters, league pool, profiles, schedules and every Python / NumPy / torch generator state, so resuming continues exactly as if the run had not stopped. PyTorch runs with deterministic algorithms and fixed intra/inter-op and BLAS thread counts.

The test (ticket #24) compares an uninterrupted and a resumed small run with at least two actors: model and optimizer tensors and the semantic log must be equal, excluding timestamps. Exactness is promised only on the same machine and library versions (PyTorch does not guarantee it across versions or platforms); scientific conclusions still rely on more than one seed.

Why: the ticket requires it, synchronous rounds make it cheap, an exactly repeatable run is much easier to debug, and it is a convincing claim in the presentation. This is stricter than common practice: DouZero restores models, optimizers and counters but not actor RNG state, and does not promise identical continuation.

## Considered Options

- **Statistically equivalent resume with asynchronous actors:** a little faster (no waiting for the slowest actor), but a resumed run can no longer be checked against an uninterrupted one.
