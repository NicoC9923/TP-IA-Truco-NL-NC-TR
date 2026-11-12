# Resumable training, with only the reigning champions in git

Each generation saves its champion's genome, and every 10 generations the full population is saved together with the reigning champion and the hall of fame. That way a long run can be stopped and resumed, any generation can be replayed on the evolution page, and it is always known which reigning champion the network modes load (ADR 0017). Phase 2 does the same with network checkpoints.

All of that goes to `training_output/`, which git ignores. Only what the app and the presentation need is committed, in `champions/phase1/` and `champions/phase2/`: the phase's reigning champion (weights only, without optimizer state) and a copy of the trainer log and the reference-evaluation log. A fresh clone can then play both network modes and show the evolution page without retraining. While a run is going, the evolution page reads the live logs in `training_output/`; otherwise it reads the committed copies.

## Considered Options

- **Commit all training output:** phase 1 alone reaches tens of MB per run and phase 2 hundreds of MB to about 1 GB. Weights are effectively random numbers, so git can't compress or diff them, and every committed run stays in the history forever. GitHub rejects files over 100 MB and expects repositories to stay under about 1 GB.
- **Commit nothing from training:** every machine would have to retrain before the network modes work, including the presentation machine.
