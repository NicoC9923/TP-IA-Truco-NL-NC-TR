# Phase 2 live progress file and what the evolution page reads

Status: accepted (ticket #34, spec #15).

The new evolution page (ADR 0009 replaced by a three-tab page) watches a phase 2 run without any extra work in the trainer beyond one small file and one terminal line per round.

## Progress file

After every round the trainer rewrites `progress.json` in the run output directory: write `progress.json.tmp`, `fsync`, then `os.replace`, so a reader never sees a half-written file (on Windows, a refused replace is retried briefly and then skipped; losing one update never stops training). Fields: `elapsed`, `time_cap`, `generation` (completed generations), `round`, `matches`, `samples`, `updates`, `matches_per_s` and `samples_per_s` (over the last 5 rounds, wall clock), `recent_loss` (mean of the last 20 updates), `league_mix` (share of matches per category in the current generation), `temperature`, `champion_generation`, `promotions`, `panel_version`, `status` (`training` or `ended`) and `stop_reason` (`patience`, `time_cap`, or `completed` when the planned rounds finish; a run stopped early by a test hook stays `training`).

The file is observational. It never enters the semantic log or the checkpoint, uses its own wall clock (not the injected training clock) and touches no random stream, so a run is bit-identical with progress on or off, and resume stays identical (tested). Writing is best-effort: any `OSError` on create, write, `fsync` or replace is swallowed, one warning goes to stderr, and training continues without the file (same for a closed terminal). Measured cost on a tiny run (4 workers, 4 generations of 16 matches, 3 alternating runs): median 7.30 s with `--no-progress` (7.37 / 7.30 / 7.22) and 7.26 s with progress (7.46 / 7.19 / 7.26), no measurable slowdown. `--no-progress` turns it and the terminal output off. The terminal shows one round line overwritten with `\r` and one permanent line per generation, e.g. `gen 24 · mejora +1.8 pts (±1.1) · asciende · 3h12m`; plain stdout writes only.

## Page feeds

- `/evolucion/phase2/{generations,reference,progress}` read the live run in `training_output/phase2` when it has a log or progress file, else the committed copies in `champions/phase2` (ADR 0018). Empty states are `[]` and `{"status": "empty"}`, never a 404.
- `/evolucion/phase2/example-hand`: the reigning champion plays one fixed-seed (7) hand, in a full match, against the committed phase 1 champion and returns each decision with its top three percentages. The champion and its sampling temperature come from the same run directory, taken as a unit (live `training_output/phase2` first, else committed `champions/phase2`): the temperature is read from `reigning_champion.json`, a sidecar the trainer writes next to every exported champion (`{generation, temperature}`), else from that directory's `progress.json`, else 0.1. Committed copies are therefore self-describing. The result is cached by (champion hash, opponent hash, temperature, seed), so repeated expansions and simultaneous clients do not replay the match; an unreadable champion file (for example mid-write) returns `unreadable_champion`. Torch is optional: without it, or without a champion, the route returns `{"available": false, "reason": ...}`. Phase 1's example hand reuses the one already logged in `generations.jsonl`, taken only from the reigning champion's own generation entry (an empty state if that entry has none). The phase 1 log records decisions, legal actions and bluff flags but not action probabilities, so the phase 1 hand is shown without percentages; none are invented. `/evolucion/phase1/source` says whether the phase 1 log is `live`, `committed` or `empty`.
- Polling: every 5 seconds, only while something is training: phase 2 `status` is `training`, or the phase 1 source is `live` and its last generation has not ended the phase (status is derived from data, never hard-coded). A failed fetch does not by itself keep polling alive. A phase 2 run on the measuring-only panel is labelled "Panel de medición (provisorio)" with an adjusted rule note.

## Considered Options

- **Trainer serves its own status over HTTP:** an extra process and port; rejected, the file is enough.
- **Page tails `generations.jsonl` only:** no mid-generation rate, bar or end estimate; rejected.
