# Phase 2 league: own self, a bounded snapshot pool with phase 1 networks, and gentle styles

Status: accepted (user decision, 2026-11-08). Proportions are pilot values. Amends ADRs 0004, 0013 and 0015 for phase 2.

Training opponents, by share of matches:

- **50% the current network** (self-play).
- **30% frozen snapshots:** phase 2 snapshots plus about 10 points of the total from phase 1 networks. Phase 2 saves a snapshot every generation, promoted or not, so the pool keeps strategies that are different without being stronger. The pool is bounded (recent snapshots plus a few behaviorally distinct older ones) so the useful recent ones keep being drawn.
- **20% styled snapshots,** 5% for each of the four styles (bluffer, cautious, aggressive, calculating), present from the start with small offsets in Q units (about 0.05–0.15) that ramp up over the first ~20,000 matches. Offsets act only on the opponent's choice, never on the learner's targets. Their behavior is checked, not assumed.

The phase 1 training opponents, split equally within the ~10%, are gate53 seed 1 generation 26 (v1), v3 seed 1 generation 55, v3 seed 2 generation 50 and matched v2 seed 3 generation 23, each loaded with its own encoder and profile semantics, with file hashes frozen. The networks used as fixed selection anchors (ADR 0047) are reserved and never sampled in the league.

Sometimes the learner plays a block of consecutive matches against the same frozen opponent, keeping that opponent's measured profile, so the profile fills up and the network learns to use it, as when a person plays several matches in a row.

Reactive and Adaptive Utility never take part (ADR 0021): they are the reference exam, and training against the exam would make its score meaningless and teach the network to exploit two fixed rule sets rather than to play well.

Why styles at all: pure self-play plus past snapshots can produce a family of networks with similar habits; if all opponents bluff and accept alike, the opponent profile carries no signal and adaptation cannot be learned. 50% styled (phase 1's split) spends too much time against caricatures with artificial mistakes. A 0% vs 20% comparison can be run later; styles stay only if they help.

Why phase 1 networks: competent opponents from the first match, at no extra training cost. Using opponents other than oneself is common (AlphaStar's league); keeping evaluation opponents out of training is standard practice. Consequence: phase 2 is "assisted" by phase 1, which the comparison discloses (ADR 0046).

## Considered Options

- **Only its own networks (ADR 0015 as written):** a clean algorithm comparison, but slower early learning; rejected for strength.
- **Also train against Reactive and Adaptive Utility:** strongest early opposition, but contaminates the exam.
- **Only promoted champions in the pool (hall of fame, ADR 0004):** the 55% filter discards useful variety.
- **Styles only after the first promotion (phase 1's deviation, ADR 0013):** phase 2 instead starts styles gently from the beginning with ramped offsets.
- **Randomized opponent temperature, multiple independent agents, dedicated exploiters (AlphaStar):** vary mistake rate rather than style, or cost scarce training time; deferred until weaknesses persist.
