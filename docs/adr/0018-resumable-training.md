# Resumable training

Each generation saves its champion's genome, and every 10 generations the full population is saved together with the reigning champion and the hall of fame. That way a long run can be stopped and resumed, any generation can be replayed on the evolution page, and it is always known which reigning champion the network modes load (ADR 0017). Phase 2 does the same with network checkpoints.
