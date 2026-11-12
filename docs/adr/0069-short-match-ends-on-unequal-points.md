# Short matches: a points target of 30, and tie-break hands until the points differ

Status: accepted (2026-11-11). Applies to spectator mode (ADR 0064) and to the human-vs-robot match.

## Context

A short match lasts a chosen number of hands (1 to 10). Two questions were open:

- **The extra hand can also end level.** A hand can score for both players (one wins the envido, the other the truco), so "one extra hand decides" does not always decide.
- **The match still needs a points target.** Falta envido is worth what the leader needs to reach the target, and both trained networks read the target and the points to it among their inputs; they were trained only with 15 and 30.

The training engine (`play_match`) ended a match only at a points target.

## Decision

- After the chosen number of hands, the match ends as soon as the points are not level. While they are level, one more hand is played, as many times as needed. The page labels them "desempate".
- A short match uses 30 as its points target for falta envido and for the networks' inputs, and reaching 30 also ends it. In a short match the leader is always in malas (under 15), so falta envido is worth the whole match: **an accepted falta envido always ends a short match**, and its winner goes to 30. This was an explicit user decision (2026-11-11, Q26 = A) over options B and C below; falta envido stays a deliberate all-or-nothing call in short matches.
- The engine gets two optional keyword arguments, `max_hands` and `on_event` (a callback at each deal and hand end). Both default to `None`, which leaves every training, evaluation and golden result unchanged; a test checks that a match played with `on_event` is identical to one without.
- `PartidaTruco` gets the same rule (`reiniciar_partido(manos=N)`), counted on the hands it already records.

## Considered options

- **Exactly one extra hand, then a draw:** a match with no winner, which neither the end screen nor the history models.
- **Target 15 for short matches:** the networks would read a target they were trained with, but it changes nothing about falta envido, which with target 30 already ends the match when accepted.
- **B. Falta envido worth what the leader lacks to 15** (or to 30 once both are past 15): a bounded falta that would not end the match, at the cost of a special scoring rule found nowhere else. Rejected by the user in favour of A.
- **C. No falta envido in short matches:** removes the all-or-nothing call, but takes a legal envido call away from the players and from the networks. Rejected by the user in favour of A.

## Consequences

- A short match may last more hands than chosen; the extra ones are visible as tie-break hands.
- A short match can end before its chosen number of hands: an accepted falta envido ends it on the spot (a test checks this).
- The "short match" definition in CONTEXT.md says extra hands are played until the points differ.
