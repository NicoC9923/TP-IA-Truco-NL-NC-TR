# Short matches: a points target of 30, and tie-break hands until the points differ

Status: accepted (2026-11-11). Applies to spectator mode (ADR 0064) and to the human-vs-robot match.

## Context

A short match lasts a chosen number of hands (1 to 10). Two questions were open:

- **The extra hand can also end level.** A hand can score for both players (one wins the envido, the other the truco), so "one extra hand decides" does not always decide.
- **The match still needs a points target.** Falta envido is worth what the leader needs to reach the target, and both trained networks read the target and the points to it among their inputs; they were trained only with 15 and 30.

The training engine (`play_match`) ended a match only at a points target.

## Decision

- After the chosen number of hands, the match ends as soon as the points are not level. While they are level, one more hand is played, as many times as needed. The page labels them "desempate".
- A short match uses 30 as its points target for falta envido and for the networks' inputs. Reaching 30 also ends it, which can only happen with a very high-scoring hand.
- The engine gets two optional keyword arguments, `max_hands` and `on_event` (a callback at each deal and hand end). Both default to `None`, which leaves every training, evaluation and golden result unchanged; a test checks that a match played with `on_event` is identical to one without.
- `PartidaTruco` gets the same rule (`reiniciar_partido(manos=N)`), counted on the hands it already records.

## Considered options

- **Exactly one extra hand, then a draw:** a match with no winner, which neither the end screen nor the history models.
- **Target 15 for short matches:** falta envido would often be worth the whole match, and nothing in a short match is close to 15 points in a way that matters.

## Consequences

- A short match may last more hands than chosen; the extra ones are visible as tie-break hands.
- The "short match" definition in CONTEXT.md says extra hands are played until the points differ.
