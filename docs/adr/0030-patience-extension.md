# 0030. Explicit bounded patience extension

## Context

Seed 1 stopped after 50 generations without promotion, despite near-promotions at generations 65, 66 and 71 (53.8%, 54.95% and 54.3% against the 55% gate).

## Decision

Allow an explicit patience extension when resuming a checkpoint. The first resumed generation records the previous and new patience, and subsequent checkpoints retain the new value. This bounded experiment leaves the shipped champion and default configuration unchanged. The pipeline uses the latest checkpoint in its output directory.

## Alternatives

Silently changing patience defaults obscures the experiment; restarting with a new seed does not continue this run; manually editing checkpoints is error-prone and conceals the change.
