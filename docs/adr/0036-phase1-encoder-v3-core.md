# Phase 1 encoder v3-core experiment

Status: accepted for an experiment; no shipped champion is changed. V3-core bundles several information and accounting fixes in one training run, so its effects cannot be attributed separately. The 64–24–12 network topology, fitness, promotion gate and v1/v2 behavior remain unchanged.

V3 uses Truco rank for public table cards as well as held cards. Owner distinguishes an empty table slot from a weakest-rank card. Each held slot now has a current-trick outcome: exact +1, 0 or -1 against a visible opposing card in the current trick; otherwise the posterior expectation P(win) − P(loss), assuming a uniformly selected remaining opponent card. A parda is zero even for mano, and unavailable slots are zero with their availability retained. Brava probability and expected opponent strength remain once each. The completed-trick count disambiguates an unplayed trick and a parda.

Pending envido features encode both winner-dependent accepted amounts and the decline amount, using the engine's payable amounts. Falta envido uses the score and target, including winner-dependent amounts when a 30-point match is still in malas. No pending envido means three zeros. Direct real envido therefore differs from an envido-envido-real envido sequence even when their other fields coincide.

## Complete v3 input map

All indexes are zero-based. Rank is `valor_carta_truco / 13` (0 through 13 before normalization); suit is its index in `PALOS` divided by 3; tantos digit is `min(value, 10) % 10 / 7`. The three held slots persist after play, and availability identifies legal cards. Table owner is +1 self, -1 opponent, or 0 absent; absent table rank and suit are also zero.

| Index | Feature |
| --- | --- |
| 0 | Held slot 0 rank |
| 1 | Held slot 0 suit |
| 2 | Held slot 0 availability |
| 3 | Held slot 0 tantos digit |
| 4 | Held slot 1 rank |
| 5 | Held slot 1 suit |
| 6 | Held slot 1 availability |
| 7 | Held slot 1 tantos digit |
| 8 | Held slot 2 rank |
| 9 | Held slot 2 suit |
| 10 | Held slot 2 availability |
| 11 | Held slot 2 tantos digit |
| 12 | Table slot 0 rank |
| 13 | Table slot 0 suit |
| 14 | Table slot 0 owner |
| 15 | Table slot 1 rank |
| 16 | Table slot 1 suit |
| 17 | Table slot 1 owner |
| 18 | Table slot 2 rank |
| 19 | Table slot 2 suit |
| 20 | Table slot 2 owner |
| 21 | Table slot 3 rank |
| 22 | Table slot 3 suit |
| 23 | Table slot 3 owner |
| 24 | Table slot 4 rank |
| 25 | Table slot 4 suit |
| 26 | Table slot 4 owner |
| 27 | Table slot 5 rank |
| 28 | Table slot 5 suit |
| 29 | Table slot 5 owner |
| 30 | Trick 1 winner |
| 31 | Trick 2 winner |
| 32 | Trick 3 winner |
| 33 | Current truco points / 4 |
| 34 | Envido resolved flag |
| 35 | Pending envido flag |
| 36 | Pending truco flag |
| 37 | Self is mano flag |
| 38 | Self owns truco raise flag |
| 39 | Self points to target / 30 |
| 40 | Opponent points to target / 30 |
| 41 | Falta envido points if self wins / 30 |
| 42 | Signed envido call count |
| 43 | Signed real envido call count |
| 44 | Signed falta envido call count |
| 45 | Signed truco call count |
| 46 | Signed retruco call count |
| 47 | Signed vale cuatro call count |
| 48 | Self envido declaration |
| 49 | Opponent envido declaration |
| 50 | Posterior probability of an opponent brava |
| 51 | Posterior expected remaining opponent strength / 24.7 |
| 52 | Current-trick outcome for held slot 0 |
| 53 | Current-trick outcome for held slot 1 |
| 54 | Current-trick outcome for held slot 2 |
| 55 | Opponent bluff rate |
| 56 | Opponent fold rate |
| 57 | Opponent truco acceptance rate |
| 58 | Opponent envido call rate |
| 59 | Completed tricks / 2: 0, 0.5 or 1 during play |
| 60 | Pending envido accepted points if self wins / 30 |
| 61 | Negative pending envido accepted points if opponent wins / 30 |
| 62 | Negative pending envido declined points / 30 |
| 63 | Reserved zero |

Trick winners are +1 for self, -1 for opponent, 0 for parda or unplayed. Each call count adds +1 for self and -1 for opponent. Declarations are numeric tantos / 33, -1 for son buenas and 0 if absent. Profile rates use the existing smoothed prior. Stakes follow `puntos_falta_envido` when applicable and the pending call's accepted and declined fields otherwise.

## Profile and compatibility

V3 policies count a truco resolution only for quiero, no quiero, fold, retruco or vale cuatro while responding to pending truco. Envido interruptions are not resolutions; the later response is counted. Legacy v1/v2 policies retain their previous counting even in mixed matches. The trainer's behavior metric counts resolutions for logging only. A human match keeps the original persisted `opponent_profile.json` for legacy policies and separately accumulates `opponent_profile_v3.json` from a neutral prior for v3. Old aggregate counts include unknown interruptions and cannot be corrected retrospectively; neither loading nor selecting v3 resets the old profile.

Bare genomes and checkpoints without encoder metadata remain v1. V3 genomes have a same-stem `.encoder.json` containing `{"encoder": "v3"}`. Explicit versions must agree with sidecars; checkpoints and run configuration record the version, and resume rejects mismatches. Each policy in mixed evaluation uses its own encoder and corresponding profile semantics. The shipped champion remains v1. Panel and reference evaluations select encoders per policy. Panel seeds default to historical 9001 and 9002, but these and 9101 are already used and require `--allow-reused-seeds`; fresh seeds can be supplied via `--baseline-seed` and `--panel-seed`.

The leak contract is that an encoding is determined only by the player's own cards and public observation, including public plays, calls, scores and declarations. The posterior never reads actual hidden opponent cards. Its expectation remains a model under uniform remaining-card play, not a claim about the opponent's strategic choice.

The [encoder v3 seed 1 experiment](../../experiments/phase1-encoder-v3-seed1/README.md) records training and the frozen panel decision.

The [matched encoder seed 2 record](../../experiments/phase1-seed2-matched/README.md) documents the subsequent comparison and decision.
