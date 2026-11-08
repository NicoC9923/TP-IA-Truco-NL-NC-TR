# Phase 2 live progress file and what the evolution page reads

Status: accepted (ticket #34, spec #15).

The new evolution page (ADR 0009 replaced by a three-tab page) watches a phase 2 run without any extra work in the trainer beyond one small file and one terminal line per round.

## Progress file

After every round the trainer rewrites `progress.json` in the run output directory: write `progress.json.tmp`, `fsync`, then `os.replace`, so a reader never sees a half-written file (on Windows, a refused replace is retried briefly and then skipped; losing one update never stops training). Fields: `elapsed`, `time_cap`, `generation` (completed generations), `round`, `matches`, `samples`, `updates`, `matches_per_s` and `samples_per_s` (over the last 5 rounds, wall clock), `recent_loss` (mean of the last 20 updates), `league_mix` (share of matches per category in the current generation), `temperature`, `champion_generation`, `promotions`, `panel_version`, `status` (`training` or `ended`) and `stop_reason` (`patience`, `time_cap`, or `completed` when the planned rounds finish; a run stopped early by a test hook stays `training`).

The file is observational. It never enters the semantic log or the checkpoint, uses its own wall clock (not the injected training clock) and touches no random stream, so a run is bit-identical with progress on or off, and resume stays identical (tested). `--no-progress` turns it and the terminal output off. The terminal shows one round line overwritten with `\r` and one permanent line per generation, e.g. `gen 24 · mejora +1.8 pts (±1.1) · asciende · 3h12m`; plain stdout writes only.

## Page feeds

- `/evolucion/phase2/{generations,reference,progress}` read the live run in `training_output/phase2` when it has a log or progress file, else the committed copies in `champions/phase2` (ADR 0018). Empty states are `[]` and `{"status": "empty"}`, never a 404.
- `/evolucion/phase2/example-hand`: the reigning champion plays one fixed-seed (7) hand, in a full match, against the committed phase 1 champion and returns each decision with its top three percentages. The sampling temperature is the one recorded in the run's `progress.json` (live, then committed), 0.1 if none. Torch is optional: without it, or without a champion, the route returns `{"available": false, "reason": ...}`. Phase 1's example hand reuses the one already logged in `generations.jsonl`; no new route.
- Polling: the page re-reads the phase 2 files every 5 seconds only while `status` is `training`, and stops when it is `ended`. Phase 1 data is read once.

## Considered Options

- **Trainer serves its own status over HTTP:** an extra process and port; rejected, the file is enough.
- **Page tails `generations.jsonl` only:** no mid-generation rate, bar or end estimate; rejected.
