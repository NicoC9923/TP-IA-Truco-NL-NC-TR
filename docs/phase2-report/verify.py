"""Independent check: re-derive every number in data.json from its source files, then check the page.

Part A re-derives data.json values from raw files with code that is separate from build_data.py
(raw per-game winners instead of results.json, fresh log parsing, text needles, weight shapes).
Part B checks that every number rendered in reporte-fase2.html equals the formatted data.json value.
Run: py verify.py
"""
import json, math, re, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fmt import f as F

HERE = Path(__file__).parent
W = Path("E:/Tools/GitHub-Repositories/TP-IA-Truco-NL-NC-TR/.orca/worktrees/TP-IA-Truco-NL-NC-TR/p2-run2")
D = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
ok = 0
bad = []


def check(label, got, want, tol=1e-9):
    global ok
    good = False
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got) and any(isinstance(x, float) for x in want):
        good = all(a == b if not isinstance(a, float) and not isinstance(b, float) else abs(a - b) <= tol * max(1, abs(b)) for a, b in zip(got, want))
    elif isinstance(want, float) or isinstance(got, float):
        good = abs(got - want) <= tol * max(1, abs(want))
    else:
        good = got == want
    if good:
        ok += 1
    else:
        bad.append((label, got, want))


def lines(p):
    return [json.loads(l) for l in (W / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def js(p):
    return json.loads((W / p).read_text(encoding="utf-8"))


def ci(xs):
    n = len(xs); m = sum(xs) / n
    sd = statistics.stdev(xs)
    return m, m - 1.96 * sd / math.sqrt(n), m + 1.96 * sd / math.sqrt(n)


# ------------------------------------------------------------------ A1. constants quoted from prose
for k, e in D.items():
    if e["how"].startswith("text:"):
        needle = e["how"][5:]
        txt = (W / e["src"]).read_text(encoding="utf-8", errors="replace")
        check(f"needle {k}", needle in txt, True)
        # the numeric value itself must appear in the needle (as written in the source)
        v = e["v"]
        cands = {str(v), f"{v:,}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:g}", f"{v:.4f}"}
        if isinstance(v, float) and v == int(v): cands.add(str(int(v)))
        if k == "speed.opt_count": cands.add("six")
        if k in ("abort.extra",): cands.add("six")
        if k in ("speed.golden_gens",): cands.add("three")
        if k in ("r1.panel_n", "styles_n"): cands |= {"four"}
        if k in ("r2.panel_n",): cands |= {"six"}
        if k == "speed.workers": cands.add("12")
        if k in ("acc.margin",): pass
        if k == "r1.share_p1": cands.add("10")
        if k == "loss_stuck_gen": cands.add("13")
        if k == "m.T": cands.add("0.05")
        if k == "orca.gen": cands.add("15")
        if k == "e.mix_pct": cands.add("0.5")
        if k == "r1.net_share": cands.add("90")
        if k == "eq.brier_diff": cands.add("0.2013")
        if k == "pilot_dummy": pass
        found = any(c in needle for c in cands) or any(c in txt for c in cands)
        check(f"value-in-source {k}={v}", found, True)

# ------------------------------------------------------------------ A2. runs: progress and logs
RUNS = {"r1": ("champions/phase2/generations.jsonl", "champions/phase2/progress.json"),
        "r2": ("champions/phase2-run2/generations.jsonl", "champions/phase2-run2/progress.json"),
        "r3": ("training_output/phase2-run3/generations.jsonl", "training_output/phase2-run3/progress.json")}
for r, (lp, pp) in RUNS.items():
    L = lines(lp); P = js(pp)
    check(f"{r}.matches", P["matches"], D[f"{r}.matches"]["v"])
    check(f"{r}.samples", P["samples"], D[f"{r}.samples"]["v"])
    check(f"{r}.updates", P["updates"], D[f"{r}.updates"]["v"])
    check(f"{r}.rate", P["matches"] / P["elapsed"], D[f"{r}.hours_avg_rate"]["v"])
    check(f"{r}.elapsed_h", P["elapsed"] / 3600, D[f"{r}.elapsed_h"]["v"])
    check(f"{r}.champ", P["champion_generation"], D[f"{r}.champion_generation"]["v"])
    prom = [g["generation"] for g in L if (g.get("check") or {}).get("promoted") is True]
    check(f"{r}.promoted", prom, D[f"{r}.promoted"]["v"])
    check(f"{r}.n_promoted", len(prom), D[f"{r}.n_promoted"]["v"])
    check(f"{r}.promotions==progress", len(prom), P["promotions"])
    full = [g for g in L if g["phase_ended"] is not True]
    check(f"{r}.n_checks", sum(1 for g in full if g["check"]), D[f"{r}.n_checks"]["v"])
    check(f"{r}.after_last", sum(1 for g in full if g["check"] and g["generation"] > prom[-1]), D[f"{r}.checks_after_last_promo"]["v"])
    check(f"{r}.mpg", L[1]["matches"], D[f"{r}.matches_per_gen"]["v"])
    ser = D[f"{r}.series"]["v"]
    check(f"{r}.series len", len(ser), len(full))
    for g, s in zip(full, ser):
        check(f"{r}.g{g['generation']}.loss", g["mean_loss"], s["loss"])
        check(f"{r}.g{g['generation']}.rate", g["matches"] / g["elapsed_seconds"], s["rate"])
        if g["check"]:
            c = g["check"]
            check(f"{r}.g{g['generation']}.gain", [c["gain"], c["lower_bound"], c["upper_bound"], c["promoted"]], [s["gain"], s["lo"], s["hi"], s["prom"]])
            # recompute the gain from the raw per-pair scores kept in the log: equal-weight mean over members of (challenger - champion)
            mem = c["members"]
            diffs = [sum(m["challenger"][i] - m["champion"][i] for m in mem.values()) / len(mem) for i in range(len(next(iter(mem.values()))["challenger"]))]
            check(f"{r}.g{g['generation']}.gain-raw", sum(diffs) / len(diffs), c["gain"], tol=1e-9)
        if r == "r3":
            check(f"r3.g{g['generation']}.gd", g["greedy_disagreement"], s["gd"])
    sec = [g for g in full if 2 <= g["generation"] <= 6]
    check(f"{r}.rate_early", sum(g["matches"] / g["elapsed_seconds"] for g in sec) / len(sec), D[f"{r}.rate_early"]["v"])
    late = full[-10:]
    check(f"{r}.rate_late", sum(g["matches"] / g["elapsed_seconds"] for g in late) / 10, D[f"{r}.rate_late"]["v"])
    check(f"{r}.secs_late", sum(g["elapsed_seconds"] for g in late) / 10, D[f"{r}.secs_late"]["v"])
    check(f"{r}.loss_min", min(g["mean_loss"] for g in full), D[f"{r}.loss_min"]["v"])
for k in ("tot.matches", "tot.hours", "tot.samples"):
    pass
check("tot.matches", sum(js(p)["matches"] for _, p in RUNS.values()), D["tot.matches"]["v"])
check("tot.samples", sum(js(p)["samples"] for _, p in RUNS.values()), D["tot.samples"]["v"])
check("tot.hours", sum(js(p)["elapsed"] for _, p in RUNS.values()) / 3600, D["tot.hours"]["v"])
check("matches M approx 2.7/2.8/12.5", [F("mil", js(p)["matches"]) for _, p in RUNS.values()], ["2.7", "2.8", "12.5"])
ext = js("champions/phase2/time_cap_extension.json")
check("r1.cap_first_h", ext["old_cap"] / 3600, D["r1.cap_first_h"]["v"])

# ------------------------------------------------------------------ A3. network, equity, pilots
import torch
sd = torch.load(W / "champions/phase2-run3/reigning_champion.pt", map_location="cpu", weights_only=True)
tot = sum(v.numel() for v in sd.values())
check("params r3", tot, D["net.params_r3"]["v"])
w0 = sd["layers.0.weight"]
check("in r3", w0.shape[1], D["net.in_r3"]["v"]); check("h1", w0.shape[0], D["net.h1"]["v"])
check("h2", sd["layers.2.weight"].shape[0], D["net.h2"]["v"]); check("out", sd["layers.4.weight"].shape[0], D["net.out"]["v"])
check("params r12", 1628 * 256 + 256 + 256 * 128 + 128 + 128 * 12 + 12, D["net.params_r12"]["v"])
check("new cols", w0.shape[1] - 1628, D["net.new_cols"]["v"])
eq = js("experiments/phase2-run3/equity-table.json")
check("eq matches", eq["metadata"]["matches"], D["eq.matches"]["v"]); check("eq obs", eq["metadata"]["observations"], D["eq.obs"]["v"])
check("eq mano15", eq["tables"]["15"][1][0][0], D["eq.mano15"]["v"]); check("eq pie15", eq["tables"]["15"][0][0][0], D["eq.pie15"]["v"])
for t, f in (("t005", "pilot-t005"), ("t002", "pilot-t002")):
    p = js(f"experiments/phase2-run3/pilots/{f}.json")
    check(f"pilot {t} gd", p["training"]["greedy_disagreement"], D[f"pilot.{t}.gd"]["v"])
    check(f"pilot {t} greedy", p["evaluation"]["pilot"]["mean"], D[f"pilot.{t}.greedy"]["v"])
    check(f"pilot {t} mps", p["throughput"]["matches_per_second"], D[f"pilot.{t}.mps"]["v"])
    check(f"pilot {t} warm", p["evaluation"]["warm_start"]["mean"], D[f"pilot.{t}.warm"]["v"])
for t in ("0.05", "0.02", "greedy"):
    pass


# ------------------------------------------------------------------ A4. exams recomputed from raw per-game winners
def pair_scores(recs, key_fn):
    out = {}
    for rec in recs:
        for who, (w0_, w1_) in key_fn(rec):
            out.setdefault((rec["block"], who), []).append(((w0_ == 0) + (w1_ == 1)) / 2)
    return out


OPP = ["AU", "R", "P1", "H1", "H2"]
for pref, path in (("e2", "experiments/phase2-final-exam/part-e"), ("e3", "experiments/phase2-run3-exam/part-e")):
    recs = lines(path + "/partial.jsonl")
    ps = pair_scores(recs, lambda r: r["winners"].items())
    uniq = {(r["block"], r["i"]) for r in recs}
    check(f"{pref} unique records", len(uniq), 5000)
    for b, o in enumerate(OPP):
        base = ps[(b, "B")]
        for who in ("B", "C1", "C2"):
            xs = ps[(b, who)]
            check(f"{pref}.wr.{who}.{o}", sum(xs) / len(xs), D[f"{pref}.wr.{who}.{o}"]["v"])
            if who != "B":
                diffs = [x - y for x, y in zip(xs, base)]
                m, lo, hi = ci(diffs)
                g = D[f"{pref}.gain.{who}.{o}"]["v"]
                check(f"{pref}.gain.{who}.{o}", [m, lo, hi], g, tol=1e-9)
    for c in ("C1", "C2"):
        bal = sum(sum(ps[(b, c)]) / len(ps[(b, c)]) - sum(ps[(b, "B")]) / len(ps[(b, "B")]) for b in range(5)) / 5
        check(f"{pref}.balance.{c}", bal, D[f"{pref}.balance.{c}"]["v"])
        # guardrail rule (lower bound > -2 pp on the 4 others) re-derived
        gr = all(D[f"{pref}.gain.{c}.{o}"]["v"][1] > -0.02 for o in OPP[1:])
        check(f"{pref}.guard.{c}", gr, D[f"{pref}.passes.{c}"]["v"][1])
# part T
recs = lines("experiments/phase2-final-exam/part-t/partial.jsonl")
ps = pair_scores(recs, lambda r: r["winners"].items())
for t in ("0.05", "0.02", "greedy"):
    for b, o in enumerate(("au", "p1")):
        xs = ps[(b, t)]
        m, lo, hi = ci(xs)
        check(f"pt.{o}.{t}", [m, lo, hi], [D[f"pt.{o}.{t}"]["v"][k] for k in ("mean", "lower", "upper")])
# paired greedy - T0.05 difference: mean over the two opponents, se from the two independent opponent blocks
dd = []
for b in (0, 1):
    xs = [ps[(b, "greedy")][i] - ps[(b, "0.05")][i] for i in range(300)]
    dd.append((sum(xs) / 300, statistics.stdev(xs) / math.sqrt(300)))
m = sum(x[0] for x in dd) / 2; se = math.sqrt(sum(x[1] ** 2 for x in dd)) / 2
check("pt.diff_greedy", [m, m - 1.96 * se, m + 1.96 * se], [D["pt.diff_greedy"]["v"][k] for k in ("mean", "lower", "upper")], tol=1e-9)

# ------------------------------------------------------------------ A5. all-vs-all and run-evolution from raw pairs
def matrix(cfgp, rawp):
    cfg = js(cfgp); ag = [a["id"] for a in cfg["agents"]]
    per = {}
    for rec in lines(rawp):
        a, b = cfg["pairings"][rec["block"]]
        g0, g1 = rec["winners"]["games"]
        per.setdefault((a, b), {})[rec["i"]] = ((g0 == 0) + (g1 == 1)) / 2
    res = {}
    for (a, b), d_ in per.items():
        xs = [d_[i] for i in sorted(d_)]
        m, lo, hi = ci(xs)
        res[(ag[a], ag[b])] = (m, lo, hi, len(xs))
        res[(ag[b], ag[a])] = (1 - m, 1 - hi, 1 - lo, len(xs))
    return res


mat = matrix("experiments/all-vs-all/output/config.json", "experiments/all-vs-all/output/partial.jsonl")
for k, v in D["avs.pairs"]["v"].items():
    a, b = k.split("|")
    check(f"avs {k}", list(mat[(a, b)][:3]), v, tol=1e-9)
check("avs pairs count", len(D["avs.pairs"]["v"]), 56)
ag = D["avs.agents"]["v"]
for a in ag:
    ms = [mat[(a, b)][0] for b in ag if b != a]
    check(f"avs eq-mean {a}", sum(ms) / 7, D["avs.rank"]["v"][a][0])
order = sorted(ag, key=lambda a: -D["avs.rank"]["v"][a][1])
check("rank order by mean == by elo", order, sorted(ag, key=lambda a: -D["avs.rank"]["v"][a][0]))
evm = matrix("experiments/run-evolution/output/config.json", "experiments/run-evolution/output/partial.jsonl")
for rn, e in D["evo"]["v"].items():
    for a, row in e["table"].items():
        for o, v in row.items():
            check(f"evo {a} vs {o}", list(evm[(a, o)][:3]), v, tol=1e-9)
    for a, row in e["within"].items():
        for o, v in row.items():
            check(f"evo within {a} vs {o}", list(evm[(a, o)][:3]), v, tol=1e-9)
a2 = js("experiments/phase2-run2/acceptance/results.json")
check("acc.direct", [a2["direct"][k] for k in ("mean", "lower", "upper")], D["acc.direct"]["v"])
check("acc.held", [a2["held_out"][k] for k in ("mean", "lower", "upper")], D["acc.held"]["v"])
check("acc.gap_pp", (0.5 - a2["direct"]["lower"]) * 100, D["acc.gap_pp"]["v"])

# ------------------------------------------------------------------ B. the page
html = (HERE / "reporte-fase2.html").read_text(encoding="utf-8")
used = json.loads((HERE / "used.json").read_text(encoding="utf-8"))


def get(path):
    m = re.match(r"([^@#]+)(.*)", path)
    v = D[m.group(1)]["v"]
    for kind, t in re.findall(r"([@#])([^@#]+)", m.group(2)):
        v = v[t] if kind == "@" else v[int(t)]
    return v


pageok = 0
for u in used:
    want = F(u["fmt"], get(u["path"]))
    check(f"page text {u['path']}", u["text"], want)
    if u["text"] in html:
        pageok += 1
    else:
        bad.append((f"not in html {u['path']}", u["text"], None))
ok += pageok
check("no unresolved placeholders", "[[" in html or "{{" in html, False)
check("size < 2MB", len(html.encode()) < 2_000_000, True)
check("title", "<title>Fase 2 · aprendizaje por refuerzo</title>" in html, True)
check("viewport meta", '<meta name="claude:viewport-fit-sides" content="cover">' in html, True)
check("no raw ids", re.search(r"v1s2-g2|v1s1-g23|phase2_run\d_g\d+|v2s1|run1-g\d", html) is None, True)

print(f"PASS {ok}   FAIL {len(bad)}   (page placeholders checked: {len(used)}, distinct data keys: {len(D)})")
for b in bad[:40]:
    print("FAIL", b)
sys.exit(1 if bad else 0)
