# Phase 1 encoder v2 experiment

Status: accepted for an experiment; no shipped champion is changed.

ADR 0034 selected held-card strength as the first input experiment, but the audit does not establish a cause for the performance plateau. Compare training from scratch with the existing v1 baseline; this change is not evidence of improvement and does not modify spec #15, whose stories permit rule-based ranks.

Encoder v1 uses face value divided by 12 for each held card. Encoder v2 replaces only those values, at zero-based indexes 0, 4 and 8, with `valor_carta_truco(value, suit) / 13`: the rule function returns integers from 0 (weakest) to 13 (strongest), with equal strengths tied. The three fixed hand slots remain in their original order even after a card is played; unavailable cards retain their identity and have availability zero. The observation contract requires three held-card slots, so missing held cards have no encoding convention and are not supported. Missing table cards retain three zero features. Table cards continue to use face value in v2: the approved experiment changes only held cards, isolating that intervention.

All other inputs, including suit, availability, tantos, table features and inference, stay unchanged. There are 57 populated features and seven padding zeros, with the same 64–24–12 topology and 1,860 weights. A genome trained under one version must be evaluated using that version; mixed matches dispatch each policy independently.

Bare `.npy` genomes and checkpoints without encoder metadata mean v1. A v2 genome is accompanied by a same-stem `.encoder.json` sidecar containing `{"encoder": "v2"}`; an explicit encoder must agree with an existing sidecar. Reference evaluation uses the run configuration when neither a sidecar nor an explicit encoder is available. Training writes the version to `run_config.json` and checkpoint state, and v2 generation entries include it (missing means v1); resuming with a different version is rejected. Publishing copies the reigning champion sidecar if present and removes an obsolete sidecar for v1. Existing panel and champion artifacts are unchanged; panel manifests may specify an encoder per member, otherwise a sidecar or v1 is used.

The input audit is v1-only: its face-value probe is hardcoded and must be updated before auditing v2. This experiment does not alter fitness, promotion gates, or shipped artifacts; a separate evaluation is required before considering adoption.


The [encoder v2 seed 1 experiment](../../experiments/phase1-encoder-v2-seed1/README.md) records the result and shipping decision.
