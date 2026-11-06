# The reference evaluation runs in a separate process

The new agent's strength is measured against the existing agents, Reactive and Adaptive Utility, in a **process separate from the trainer**. That process takes each saved champion, plays 1,000 pairs of mirror matches against each reference opponent on the validation deals, and writes its own log with the win rate and its margin of error. The evolution page merges that log with the trainer's (ADR 0009).

The separation is enforced by structure, not discipline: the trainer has no code path to the existing agents, and a test checks that the trainer module doesn't import them. For the presentation: "the exam is graded in another room."

So the yardstick doesn't move, Adaptive Utility starts every run from the same fixed initial state, without what it learned against humans, and never writes its persistent history. A measurement from one week can then be compared with the next, and evaluation matches don't pollute what it learned from humans.

## Considered Options

- **Evaluate inside the trainer with a "don't read the result" rule:** works, but depends on nobody breaking it; with a separate process it can't be broken without the test catching it.
- **Measure strength only against its own versions:** good enough for deciding promotions (ADR 0017), but results among one's own versions can go in circles and don't prove the agent is strong.
- **Measure against humans:** would need about 2,000 matches for a ±2-point margin.
