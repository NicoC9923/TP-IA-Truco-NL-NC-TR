# What phase 2 run 2 can claim, and the two held-out networks

Status: accepted (user decision, 2026-11-09).

Once Reactive and Adaptive Utility train with the model (ADR 0056), no hand-written opponent is left unseen. That changes what we can claim:

- **Reference scores are familiar-opponent results.** They show how well the network plays against strategies it practiced against. They do not show strength against unseen play.
- **The direct match against gen28** compares the two champions on fresh deals. Gen28 is also a panel member, so it is a familiar opponent too.
- **Two held-out networks are the only unseen check.** They are the only phase 1 champions never used in any league or panel:
  - `v1s1-g23`: `experiments/phase1-seed1/reigning_champion.npy`, SHA-256 `41354252…`;
  - `v1s2-g2`: `experiments/phase1-seed2/reigning_champion.npy`, SHA-256 `5f00bfba…`.

  Both use encoder v1, and one is weak, so this is a limited check. It can catch a clear loss of general play. It cannot prove general strength.

**Lockout:** both hashes are banned from every network path in run 2: the league's snapshots, its phase 1 slice, the styled versions and the panel. Loading either one anywhere in training is an error.

In the presentation, run 2 is described as "stronger against the opponents it trained with, and not worse against two networks it never saw". Stronger claims need human games or new unseen opponents.
