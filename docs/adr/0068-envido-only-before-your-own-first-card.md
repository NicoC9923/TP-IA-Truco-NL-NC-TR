# Envido can only be called before your own first-baza card

Status: accepted (user decision, 2026-11-11).

## Context

In Argentine truco, a player who has already played their card of the first baza can no longer call envido, including the "el envido está primero" response to a truco. The mano may lose the right by playing first, while the pie may still call envido after the mano played, as long as the pie has not played yet.

The game (`PartidaTruco`) allowed envido for as long as the first baza was open, so a human or a robot could call it after playing. The training engine (`symmetric_engine.py`) has the same permissive rule, and the shipped networks were trained on it.

## Decision

Envido is legal only while the caller has not played a card in the first baza. This is enforced in the game layer only:

- `puede_cantar_envido(quien)` checks that `quien` has no card on the table in baza 1.
- Human: the envido buttons, the voice command and the "el envido está primero" response to a truco are all not offered, and the calls are rejected with "No podés cantar envido ahora."
- Robot: the classic agents and the network modes only consider envido when the robot has not played. For the networks, the envido actions are removed from the legal-action list before the network chooses, so it picks the best legal action.
- The training engine and every trained network are not changed.
- Spectator mode (ADR 0064) runs on the symmetric engine and will enforce this rule in its own driver layer, by masking the illegal envido actions.

## Considered options

- **Change the training engine and retrain:** the cleanest rules, but it invalidates the trained champions and the run results already documented.
- **Keep the permissive rule:** no work, but the game would allow a call a real table would reject.

## Consequences

- Games against a human follow the real rule, and the networks only lose options they rarely needed after playing.
- The training engine and the game now differ on this rule on purpose; the difference is documented here and lives in one place per layer.
