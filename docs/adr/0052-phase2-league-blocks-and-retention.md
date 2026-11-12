# Phase 2 league blocks and snapshot retention

Status: accepted. This supplements ADR 0043 with execution details.

League draws select one opponent for a block of four matches by default. The block size is configurable; a round is rounded up to a whole number of blocks, and a generation to whole rounds. Both policies keep their opponent profiles during a block. The learner alternates seats by logical match ID so first-hand mano is balanced independently of worker scheduling.

When the phase 2 snapshot pool is empty, its unstyled share is reassigned to current self. Styled draws continue from the start, using the current frozen round network as the opponent until a snapshot is available; the style changes only that opponent's action values. An unavailable phase 1 share is likewise reassigned to self after panel exclusions.

The pool retains the eight most recent eligible snapshots by default and up to four older snapshots. An older snapshot's diversity is the smallest L1 distance from its score vector to a recent snapshot with the same nonempty panel-member dimensions; absent or incompatible vectors have distance zero. Older snapshots with the largest distances are retained, with ascending hash as the tie-break. This deterministic rule favors behavioral differences from the recent set without comparing incompatible panel versions, and both limits are configurable.
