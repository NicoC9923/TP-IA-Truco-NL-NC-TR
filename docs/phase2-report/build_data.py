"""Extract every number used by reporte-fase2.html from its source file -> data.json.

Each entry: {"v": value, "src": relative source file(s), "how": how it was read}.
Run: py build_data.py
"""
import json, re, os
from pathlib import Path

W = Path("E:/Tools/GitHub-Repositories/TP-IA-Truco-NL-NC-TR/.orca/worktrees/TP-IA-Truco-NL-NC-TR/p2-run2")
OUT = Path(__file__).parent
D = {}


def put(key, v, src, how):
    D[key] = {"v": v, "src": src, "how": how}


def J(p):
    return json.loads((W / p).read_text(encoding="utf-8"))


def JL(p):
    return [json.loads(l) for l in (W / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def T(p):
    return (W / p).read_text(encoding="utf-8")


def const(key, v, src, needle):
    """A number quoted from prose: kept with the exact sentence fragment that must exist in the file."""
    put(key, v, src, "text:" + needle)


# ---------------------------------------------------------------- runs
RUNS = {
    "r1": dict(log="champions/phase2/generations.jsonl", prog="champions/phase2/progress.json"),
    "r2": dict(log="champions/phase2-run2/generations.jsonl", prog="champions/phase2-run2/progress.json"),
    "r3": dict(log="training_output/phase2-run3/generations.jsonl", prog="training_output/phase2-run3/progress.json"),
}
for r, c in RUNS.items():
    pg = J(c["prog"]); L = JL(c["log"])
    for k in ("matches", "samples", "updates", "elapsed", "champion_generation", "promotions", "temperature"):
        put(f"{r}.{k}", pg[k], c["prog"], f"progress.json[{k}]")
    put(f"{r}.hours_avg_rate", pg["matches"] / pg["elapsed"], c["prog"] + " (matches/elapsed)", "matches / elapsed seconds")
    full = [g for g in L if not g.get("phase_ended")]
    chk = [g for g in full if g.get("check")]
    put(f"{r}.n_gens_full", len(full), c["log"], "records without phase_ended")
    put(f"{r}.n_checks", len(chk), c["log"], "full records with a check")
    put(f"{r}.matches_per_gen", L[1]["matches"], c["log"], "matches of generation 2")
    prom = [g["generation"] for g in L if g.get("check") and g["check"].get("promoted")]
    put(f"{r}.promoted", prom, c["log"], "check.promoted true")
    put(f"{r}.n_promoted", len(prom), c["log"], "len(promoted)")
    last_prom = prom[-1]
    put(f"{r}.checks_after_last_promo", len([g for g in chk if g["generation"] > last_prom]), c["log"], "checks after last promotion")
    put(f"{r}.last_gen_full", full[-1]["generation"], c["log"], "last full generation")
    ser = []
    for g in full:
        ck = g.get("check")
        row = dict(gen=g["generation"], loss=g["mean_loss"], rate=g["matches"] / g["elapsed_seconds"],
                   secs=g["elapsed_seconds"])
        if ck:
            row.update(gain=ck["gain"], lo=ck["lower_bound"], hi=ck["upper_bound"], h2h=ck["head_to_head"],
                       prom=bool(ck["promoted"]), panel=ck.get("panel_version"))
        if "greedy_disagreement" in g:
            row["gd"] = g["greedy_disagreement"]
        ser.append(row)
    put(f"{r}.series", ser, c["log"], "per-generation: mean_loss, matches/elapsed_seconds, check.gain/lower_bound/upper_bound/head_to_head/promoted, greedy_disagreement")
    sec = [s for s in ser if 2 <= s["gen"] <= 6]
    late = ser[-10:]
    put(f"{r}.rate_early", sum(s["rate"] for s in sec) / len(sec), c["log"], "mean of matches/elapsed, generations 2-6")
    put(f"{r}.rate_late", sum(s["rate"] for s in late) / len(late), c["log"], "mean of matches/elapsed, last 10 full generations")
    put(f"{r}.secs_early", sum(s["secs"] for s in sec) / len(sec), c["log"], "mean elapsed seconds per generation, generations 2-6")
    put(f"{r}.secs_late", sum(s["secs"] for s in late) / len(late), c["log"], "mean elapsed seconds per generation, last 10 full generations")
    put(f"{r}.rate_min", min(s["rate"] for s in ser[1:]), c["log"], "min matches/elapsed (gen>=2)")
    put(f"{r}.rate_max", max(s["rate"] for s in ser[1:]), c["log"], "max matches/elapsed (gen>=2)")
    put(f"{r}.loss_min", min(s["loss"] for s in ser), c["log"], "min mean_loss")
    put(f"{r}.loss_first", ser[0]["loss"], c["log"], "mean_loss generation 1")
    put(f"{r}.loss_last", ser[-1]["loss"], c["log"], "mean_loss last full generation")
    put(f"{r}.gain_at_promo", {str(s["gen"]): [s["gain"], s["lo"], s["hi"]] for s in ser if s.get("prom")}, c["log"], "check gain/lo/hi of promoted gens")
    lm = L[1]["league_mix"]; put(f"{r}.league_gen2", lm, c["log"], "league_mix of generation 2")

# run 3 extra: greedy disagreement
s3 = D["r3.series"]["v"]
gd = [s["gd"] for s in s3]
put("r3.gd_first", gd[0], "training_output/phase2-run3/generations.jsonl", "greedy_disagreement generation 1")
put("r3.gd_last", gd[-1], "training_output/phase2-run3/generations.jsonl", "greedy_disagreement last full generation")
put("r3.gd_min", min(gd), "training_output/phase2-run3/generations.jsonl", "min greedy_disagreement")
put("r3.gd_max", max(gd), "training_output/phase2-run3/generations.jsonl", "max greedy_disagreement")

# run 3 progress
pr = J("training_output/phase2-run3/progress.json")
put("r3.generation_cut", 101, "training_output/phase2-run3/generations.jsonl", "generation 101 is the partial record cut by the cap")
put("r1.elapsed_h", D["r1.elapsed"]["v"] / 3600, "champions/phase2/progress.json", "elapsed/3600")
put("r2.elapsed_h", D["r2.elapsed"]["v"] / 3600, "champions/phase2-run2/progress.json", "elapsed/3600")
put("r3.elapsed_h", D["r3.elapsed"]["v"] / 3600, "training_output/phase2-run3/progress.json", "elapsed/3600")
ext = J("champions/phase2/time_cap_extension.json")
put("r1.cap_first_h", ext["old_cap"] / 3600, "champions/phase2/time_cap_extension.json", "old_cap/3600")
put("r1.cap_ext_h", ext["new_cap"] / 3600, "champions/phase2/time_cap_extension.json", "new_cap/3600")

# ---------------------------------------------------------------- network
import torch
sd = torch.load(W / "champions/phase2-run3/reigning_champion.pt", map_location="cpu", weights_only=True)
shp = {k: list(v.shape) for k, v in sd.items()}
n3 = sum(v.numel() for v in sd.values())
put("net.in_r3", shp["layers.0.weight"][1], "champions/phase2-run3/reigning_champion.pt", "shape layers.0.weight[1]")
put("net.h1", shp["layers.0.weight"][0], "champions/phase2-run3/reigning_champion.pt", "shape layers.0.weight[0]")
put("net.h2", shp["layers.2.weight"][0], "champions/phase2-run3/reigning_champion.pt", "shape layers.2.weight[0]")
put("net.out", shp["layers.4.weight"][0], "champions/phase2-run3/reigning_champion.pt", "shape layers.4.weight[0]")
put("net.params_r3", n3, "champions/phase2-run3/reigning_champion.pt", "sum numel of the state dict (1,635 inputs)")
n_in12 = 1628
put("net.in_r12", n_in12, "experiments/all-vs-all/output/config.json", "models.R2G38.input_size")
put("net.params_r12", n_in12 * 256 + 256 + 256 * 128 + 128 + 128 * 12 + 12, "formula from layer sizes", "1628*256+256+256*128+128+128*12+12")
put("net.new_cols", shp["layers.0.weight"][1] - n_in12, "champions/phase2-run3/reigning_champion.pt", "1635 - 1628")
ava = J("experiments/all-vs-all/output/config.json")
assert ava["models"]["R2G38"]["input_size"] == 1628 and ava["models"]["R3G73"]["input_size"] == 1635

# ---------------------------------------------------------------- equity
eq = J("experiments/phase2-run3/equity-table.json")
put("eq.matches", eq["metadata"]["matches"], "experiments/phase2-run3/equity-table.json", "metadata.matches")
put("eq.obs", eq["metadata"]["observations"], "experiments/phase2-run3/equity-table.json", "metadata.observations")
put("eq.mano15", eq["tables"]["15"][1][0][0], "experiments/phase2-run3/equity-table.json", "tables[15][mano=1][0][0]")
put("eq.pie15", eq["tables"]["15"][0][0][0], "experiments/phase2-run3/equity-table.json", "tables[15][mano=0][0][0]")
put("eq.mano30", eq["tables"]["30"][1][0][0], "experiments/phase2-run3/equity-table.json", "tables[30][mano=1][0][0]")
put("eq.pie30", eq["tables"]["30"][0][0][0], "experiments/phase2-run3/equity-table.json", "tables[30][mano=0][0][0]")
const("eq.brier_table", 0.1795, "experiments/phase2-run3/RULE.txt", "Brier 0.1795")
const("eq.brier_diff", 0.2013, "experiments/phase2-run3/RULE.txt", "0.2013 for the score difference")

# ---------------------------------------------------------------- pilots
for t, f in (("t005", "pilot-t005.json"), ("t002", "pilot-t002.json")):
    p = J("experiments/phase2-run3/pilots/" + f); src = "experiments/phase2-run3/pilots/" + f
    put(f"pilot.{t}.greedy", p["evaluation"]["pilot"]["mean"], src, "evaluation.pilot.mean")
    put(f"pilot.{t}.warm", p["evaluation"]["warm_start"]["mean"], src, "evaluation.warm_start.mean")
    put(f"pilot.{t}.gd", p["training"]["greedy_disagreement"], src, "training.greedy_disagreement")
    put(f"pilot.{t}.mps", p["throughput"]["matches_per_second"], src, "throughput.matches_per_second")
    put(f"pilot.{t}.pairs", p["evaluation"]["pairs"], src, "evaluation.pairs")
put("pilot.threshold", J("experiments/phase2-run3/pilots/pilot-t002.json")["threshold"], "experiments/phase2-run3/pilots/pilot-t002.json", "threshold")

# ---------------------------------------------------------------- speed (docs/perf)
SP = "docs/perf/phase2-speed.md"
const("speed.base_mps", 53.6, SP, "| Training matches/s (actor rounds) | 53.6 | 392.0 | 7.3× |")
const("speed.final_mps", 392.0, SP, "| Training matches/s (actor rounds) | 53.6 | 392.0 | 7.3× |")
const("speed.actor_x", 7.3, SP, "| Training matches/s (actor rounds) | 53.6 | 392.0 | 7.3× |")
const("speed.check_base", 42.7, SP, "| Check, 6 members × 2 × 100 + 100 direct pairs | 42.7 s | 7.7 s | 5.6× |")
const("speed.check_final", 7.7, SP, "| Check, 6 members × 2 × 100 + 100 direct pairs | 42.7 s | 7.7 s | 5.6× |")
const("speed.check_x", 5.6, SP, "| Check, 6 members × 2 × 100 + 100 direct pairs | 42.7 s | 7.7 s | 5.6× |")
const("speed.e2e_base", 110.0, SP, "**110.0 → 450.9 matches/s (4.1×)**")
const("speed.e2e_final", 450.9, SP, "**110.0 → 450.9 matches/s (4.1×)**")
const("speed.e2e_x", 4.1, SP, "**110.0 → 450.9 matches/s (4.1×)**")
const("speed.golden_gens", 3, SP, "three generations of 96 matches")
const("speed.golden_matches", 96, SP, "three generations of 96 matches")
const("speed.golden_samples", 5579, SP, "5,579 learner samples and 25 updates")
const("speed.golden_updates", 25, SP, "5,579 learner samples and 25 updates")
const("speed.tests", 499, SP, "499 tests OK")
const("ship.tests", 633, "docs/adr/0063-ship-run3-champion-by-user-decision.md", "633 tests OK")
const("speed.opt_count", 6, SP, "## Kept optimizations (one commit each, golden equal at workers 1 and 12)")
const("speed.actor_profile_base", 6.23, SP, "| Whole chunk | 6.23 s | 0.93 s |")
const("speed.actor_profile_final", 0.93, SP, "| Whole chunk | 6.23 s | 0.93 s |")
const("speed.workers", 12, SP, "12 stays")
const("speed.run2_gen_min", 15, "docs/adr/0062-phase2-run3-design.md", "about 15 minutes each")
const("speed.workers_8", 329.5, SP, "8 → 329.5, 12 → 422.1")
const("speed.workers_12", 422.1, SP, "8 → 329.5, 12 → 422.1")

# ---------------------------------------------------------------- measuring run & misc ADR constants
const("m.mps", 24.73, "experiments/phase2-measuring-run/README.md", "**24.73 matches/s**")
const("m.matches_gen", 57120, "experiments/phase2-measuring-run/README.md", "**57,120 matches per generation**")
const("m.T", 0.05, "experiments/phase2-measuring-run/README.md", "selected **T = 0.05**")
const("abort.mins", 80, "experiments/phase2-run1/README.md", "after about 80 minutes and 6 generations")
const("abort.gens", 6, "experiments/phase2-run1/README.md", "after about 80 minutes and 6 generations")
const("abort.pct", 0.26, "experiments/phase2-run1/README.md", "This happened in about 0.26% of decisions")
const("abort.per", 14, "docs/adr/0055-accept-truco-stake-on-raise.md", "approximately once per 14 matches")
const("abort.pct2", 0.7, "docs/adr/0055-accept-truco-stake-on-raise.md", "the illegal envido was offered in 0.7% of decisions")
const("abort.extra", 6, "experiments/phase2-run1/README.md", "found and fixed six more problems")
const("r1.panel_rot_gen", 14, "experiments/phase2-run1/README.md", "At generation 14 the panel rotated")
const("r1.net_share", 90, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "About 90% of its training opponents were closely related networks")
const("r1.vs_au", 35, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "it scored about 35%")
const("tests.suite", 237, "docs/perf/phase2-speed.md", "499 tests OK (237 s)")
const("r1.eval.g4.au", 23.5, "experiments/phase2-run1/README.md", "| gen4 | 46.9% ± 3.1 | 23.5% ± 2.6 |")
const("r1.eval.g10.au", 37.2, "experiments/phase2-run1/README.md", "| gen10 | 68.4% ± 2.8 | 37.2% ± 2.8 |")
const("r1.eval.g28.au", 34.6, "experiments/phase2-run1/README.md", "| gen28 | 68.8% ± 2.9 | 34.6% ± 3.0 |")
const("r1.eval.g28.re", 68.8, "experiments/phase2-run1/README.md", "| gen28 | 68.8% ± 2.9 | 34.6% ± 3.0 |")
const("fm.au_mean", 75.300, "docs/research/phase2-comparison-improvement.md", "**75.300%**")
const("fm.rl_mean", 61.233, "docs/research/phase2-comparison-improvement.md", "**61.233%**")
const("fm.re_mean", 34.383, "docs/research/phase2-comparison-improvement.md", "**34.383%**")
const("fm.ge_mean", 29.083, "docs/research/phase2-comparison-improvement.md", "**29.083%**")
const("fm.rl_vs_au", 36.95, "docs/research/phase2-comparison-improvement.md", "scored 36.95% against Adaptive Utility")
const("r2.share_self", 30, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| Current network (self-play) | 30% |")
const("r2.share_snap", 20, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| Snapshots of the run | 20% |")
const("r2.share_p1", 10, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| Phase 1 networks (same four as run 1) | 10% |")
const("r2.share_style", 5, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| 5% each, 20% in total |")
const("r2.share_re", 10, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| Reactive | 10% |")
const("r2.share_au", 10, "docs/adr/0056-phase2-run2-references-train-with-the-model.md", "| Adaptive Utility | 10% |")
const("r1.share_self", 50, "docs/adr/0043-phase2-league.md", "**50% the current network**")
const("r1.share_snap", 30, "docs/adr/0043-phase2-league.md", "**30% frozen snapshots:**")
const("r1.share_style", 20, "docs/adr/0043-phase2-league.md", "**20% styled snapshots,**")
const("r1.share_p1", 10, "docs/adr/0043-phase2-league.md", "about 10 points of the total from phase 1 networks")
const("r1.pairs_member", 100, "docs/adr/0047-phase2-champion-selection.md", "100 pairs against each panel member")
const("r1.pairs_direct", 200, "docs/adr/0047-phase2-champion-selection.md", "plus 200 pairs challenger vs champion")
const("r3.pairs_member", 300, "docs/adr/0062-phase2-run3-design.md", "300 mirror pairs per panel member and 300 direct pairs")
const("r2.panel_n", 6, "docs/adr/0057-phase2-run2-fixed-six-member-panel.md", "against six equal-weight members")
const("r1.panel_n", 4, "docs/adr/0047-phase2-champion-selection.md", "averaged over the four members")
const("T.run3_train", 0.02, "docs/adr/0062-phase2-run3-design.md", "use **T = 0.02**")
const("gain_min", 1, "docs/adr/0047-phase2-champion-selection.md", "is at least 1 percentage point")
const("direct_min", 47.5, "docs/adr/0047-phase2-champion-selection.md", "at least 47.5% against the champion directly")
const("patience", 40, "docs/adr/0044-phase2-training-budget.md", "**40 consecutive checks without a promotion**")
const("lr", 0.0003, "experiments/phase2-run3/RULE.txt", "--lr 0.0003")
const("minibatch", 512, "experiments/phase2-run3/RULE.txt", "--minibatch 512")
const("run3.workers", 12, "experiments/phase2-run3/RULE.txt", "--workers 12")
const("orca.gen", 15, "experiments/phase2-run3/RULE.txt", "generation 15, round 7271")
const("orca.down", 20, "experiments/phase2-run3/RULE.txt", "Downtime (~20 s)")
const("run3.seed_gen2", 124560, "experiments/phase2-run3/RULE.txt", "--matches-per-generation 124560")
const("loss_stuck", 0.73, "docs/adr/0062-phase2-run3-design.md", "stopped improving at around 0.73 from generation 13")
const("loss_stuck_gen", 13, "docs/adr/0062-phase2-run3-design.md", "stopped improving at around 0.73 from generation 13")
const("styles_n", 4, "docs/adr/0043-phase2-league.md", "5% for each of the four styles")
const("actions_n", 12, "docs/adr/0040-phase2-network-and-inputs.md", "Legal actions:** 12 bits")
const("events_max", 24, "docs/adr/0040-phase2-network-and-inputs.md", "up to 24 events")
const("e.mix_pct", 50, "docs/adr/0062-phase2-run3-design.md", "0.5 × match result (+1/−1) + 0.5 × match equity")

# ---------------------------------------------------------------- Part T (run 2 exam)
pt = J("experiments/phase2-final-exam/part-t/results.json")["analysis"]
PT = "experiments/phase2-final-exam/part-t/results.json"
for t in ("0.05", "0.02", "greedy"):
    put(f"pt.au.{t}", pt["win_rates"][t][0], PT, f"analysis.win_rates[{t}][0] (Utilidad adaptativa)")
    put(f"pt.p1.{t}", pt["win_rates"][t][1], PT, f"analysis.win_rates[{t}][1] (Genético publicado)")
    put(f"pt.m.{t}", pt["m"][t], PT, f"analysis.m[{t}]")
put("pt.diff_greedy", pt["paired_vs_0.05"]["greedy"]["m_diff"], PT, "analysis.paired_vs_0.05.greedy.m_diff")
put("pt.pairs", 300, "experiments/phase2-final-exam/README.md", "300 mirror pairs against each opponent")

# ---------------------------------------------------------------- Part E exams
def exam(prefix, path, names, opp_names):
    res = J(path + "/results.json")["analysis"]
    raw = JL(path + "/partial.jsonl")
    n = {}
    for rec in raw:
        b = rec["block"];
        for who, (w0, w1) in rec["winners"].items():
            n.setdefault((b, who), [0, 0])
            n[(b, who)][0] += (w0 == 0) + (w1 == 1)
            n[(b, who)][1] += 2
    for who in names:
        for b, o in enumerate(opp_names):
            w, tot = n[(b, who)]
            put(f"{prefix}.wr.{who}.{o}", w / tot, path + "/partial.jsonl", "evaluated network wins / games (game0 seat0, game1 seat1)")
    for c in [x for x in names if x != "B"]:
        cand = res["candidates"][c]
        for b, o in enumerate(opp_names):
            g = cand["gains"][b]
            put(f"{prefix}.gain.{c}.{o}", [g["gain"], g["lower"], g["upper"]], path + "/results.json", f"analysis.candidates.{c}.gains[{b}]")
        put(f"{prefix}.balance.{c}", cand["balance"]["mean"], path + "/results.json", f"analysis.candidates.{c}.balance.mean")
        put(f"{prefix}.passes.{c}", [cand["passes"], cand["guardrails_pass"], cand["primary"]["holm_rejected"], cand["balance"]["pass"]], path + "/results.json", "passes, guardrails_pass, primary.holm_rejected, balance.pass")
    put(f"{prefix}.pairs", len([r for r in raw if r["block"] == 0]), path + "/partial.jsonl", "records in block 0")


OPP = ["AU", "R", "P1", "H1", "H2"]
exam("e2", "experiments/phase2-final-exam/part-e", ["B", "C1", "C2"], OPP)
exam("e3", "experiments/phase2-run3-exam/part-e", ["B", "C1", "C2"], OPP)

# ---------------------------------------------------------------- run 2 acceptance
ac = J("experiments/phase2-run2/acceptance/results.json"); AC = "experiments/phase2-run2/acceptance/results.json"
put("acc.direct", [ac["direct"]["mean"], ac["direct"]["lower"], ac["direct"]["upper"]], AC, "direct mean/lower/upper")
put("acc.held", [ac["held_out"]["mean"], ac["held_out"]["lower"], ac["held_out"]["upper"]], AC, "held_out mean/lower/upper")
put("acc.decision", ac["decision"], AC, "decision")
put("acc.gap", 0.5 - ac["direct"]["lower"], AC, "0.5 - direct lower bound")

# ---------------------------------------------------------------- all vs all
AV = "experiments/all-vs-all/output/results.json"
av = J(AV)["analysis"]
AG = ["reactive", "adaptive_utility", "phase1_g48", "phase2_run1_g28", "phase2_run2_g38", "phase2_run2_g49", "phase2_run3_g73", "phase2_run3_g100"]
put("avs.agents", AG, AV, "config.agents order")
mat = {}
for p in av["pairs"]:
    mat[f"{p['row']}|{p['col']}"] = [p["mean"], p["lower"], p["upper"]]
put("avs.pairs", mat, AV, "analysis.pairs row/col mean/lower/upper")
put("avs.rank", {a["id"]: [a["equal_weight_mean"], a["elo"], a["bradley_terry_strength"], a["lower"], a["upper"]] for a in av["agents"]}, AV,
    "analysis.agents equal_weight_mean, elo, bradley_terry_strength, lower, upper")
put("avs.pair_count", J(AV)["pair_count"], AV, "pair_count")
const("avs.pairs_each", 1000, "docs/adr/0063-ship-run3-champion-by-user-decision.md", "1,000 mirror pairs per pairing")

# ---------------------------------------------------------------- run evolution
RE = "experiments/run-evolution/output/results.json"
rev = J(RE)["analysis"]
evo = {}
for run in rev["runs"]:
    evo[run["name"]] = dict(agents=run["agents"], opponents=run["opponents"],
                            table={a: {o: [v["mean"], v["lower"], v["upper"]] for o, v in t.items() if v} for a, t in run["table"].items()},
                            within={a: {b: [v["mean"], v["lower"], v["upper"]] for b, v in t.items() if v} for a, t in run["within"].items()})
put("evo", evo, RE, "analysis.runs[*].table/within mean/lower/upper")
put("evo.pairs", J(RE)["pair_count"], RE, "pair_count")

# ---------------------------------------------------------------- four-mode (context)
const("fm.pairs", 1000, "experiments/four-mode-round-robin/README.md", "across 1,000 mirror pairs (2,000 games)")

# ---------------------------------------------------------------- probe
const("probe.champ_acc", 11.208, "docs/research/phase2-comparison-improvement.md", "11.208% versus 10.708%")
const("probe.ctrl_acc", 10.708, "docs/research/phase2-comparison-improvement.md", "11.208% versus 10.708%")

json.dump(D, open(OUT / "data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("keys", len(D))

# ---- appended: derived totals
D2 = json.load(open(OUT / "data.json", encoding="utf-8"))
D2["tot.matches"] = {"v": sum(D2[f"r{i}.matches"]["v"] for i in (1, 2, 3)), "src": "the three progress.json files", "how": "sum of r1,r2,r3 matches"}
D2["tot.hours"] = {"v": sum(D2[f"r{i}.elapsed_h"]["v"] for i in (1, 2, 3)), "src": "the three progress.json files", "how": "sum of elapsed/3600"}
D2["tot.samples"] = {"v": sum(D2[f"r{i}.samples"]["v"] for i in (1, 2, 3)), "src": "the three progress.json files", "how": "sum of samples"}
D2["acc.gap_pp"] = {"v": (0.5 - D2["acc.direct"]["v"][1]) * 100, "src": D2["acc.direct"]["src"], "how": "(0.5 - lower bound)*100 percentage points"}
D2["r3.rate_x_r1"] = {"v": D2["r3.hours_avg_rate"]["v"] / D2["r1.hours_avg_rate"]["v"], "src": "progress.json r1,r3", "how": "ratio of average matches/s run 3 / run 1"}
json.dump(D2, open(OUT / "data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("keys with derived", len(D2))
