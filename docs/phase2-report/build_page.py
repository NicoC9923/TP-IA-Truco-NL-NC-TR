"""Build reporte-fase2.html from template.html + style.css + data.json. Records every placeholder in used.json."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fmt import f as F
import charts as C

HERE = Path(__file__).parent
D = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
used = []

NAMES = {
    "reactive": "Reactivo", "adaptive_utility": "Utilidad adaptativa", "phase1_g48": "Genético publicado (gen 48)",
    "R": "Reactivo", "AU": "Utilidad adaptativa", "P1": "Genético publicado (gen 48)",
    "H1": "Genético semilla 1 (gen 23)", "H2": "Genético semilla 2 (gen 2)",
}
OPP = ["AU", "R", "P1", "H1", "H2"]


def name(i):
    if i in NAMES:
        return NAMES[i]
    m = re.match(r"phase2_run(\d)_g(\d+)$", i)
    return f"Refuerzo corrida {m.group(1)} · gen {m.group(2)}"


def short(i):
    m = re.match(r"phase2_run(\d)_g(\d+)$", i)
    if m:
        return f"C{m.group(1)} · {m.group(2)}"
    return {"reactive": "Reactivo", "adaptive_utility": "Util. adapt.", "phase1_g48": "Genét. 48"}[i]


def get(path):
    toks = re.match(r"([^@#]+)(.*)", path)
    key, rest = toks.group(1), toks.group(2)
    v = D[key]["v"]
    for kind, t in re.findall(r"([@#])([^@#]+)", rest):
        v = v[t] if kind == "@" else v[int(t)]
    return v


def V(fmt, path):
    s = F(fmt, get(path))
    used.append({"fmt": fmt, "path": path, "text": s})
    return s


def ph(m):
    return V(m.group(1), m.group(2))


# ----------------------------------------------------------------- chart builders
def series(r):
    return D[f"{r}.series"]["v"]


def gain_dom():
    lo = min(s["lo"] for r in ("r1", "r2", "r3") for s in series(r) if "lo" in s)
    hi = max(s["hi"] for r in ("r1", "r2", "r3") for s in series(r) if "hi" in s)
    return (-(int(-lo * 20) + 1) / 20, (int(hi * 20) + 1) / 20)


def lines_for(r, key, scale=1.0):
    return [(s["gen"], s[key] * scale) for s in series(r) if key in s]


def tbl_exam(prefix, l1, l2):
    h = (f'<div class="tw"><table><thead><tr><th>Rival</th><th class="n">Línea base</th><th class="n">{l1}</th>'
         f'<th class="n">Ganancia</th><th class="n">{l2}</th><th class="n">Ganancia</th></tr></thead><tbody>')
    for o in OPP:
        h += f'<tr><td>{C.esc(NAMES[o])}</td><td class="n">{V("pc1", f"{prefix}.wr.B.{o}")}</td>'
        for c in ("C1", "C2"):
            g = f"{prefix}.gain.{c}.{o}"
            h += (f'<td class="n">{V("pc1", f"{prefix}.wr.{c}.{o}")}</td>'
                  f'<td class="n">{V("pp1", g + "#0")} <span style="color:var(--muted)">[{V("pp1", g + "#1")}, {V("pp1", g + "#2")}]</span></td>')
        h += "</tr>"
    h += "</tbody></table></div><p class=\"sub\">Porcentaje de partidas ganadas (2,000 por celda) y ganancia sobre la línea base en puntos porcentuales, con intervalo de 95%.</p>"
    return h


def forest_for(prefix, cls, labels):
    rows = [(NAMES[o], [tuple(D[f"{prefix}.gain.{c}.{o}"]["v"]) for c in ("C1", "C2")]) for o in OPP]
    allv = [x for _, cs in rows for c in cs for x in c]
    lo, hi = min(allv), max(allv)
    dom = (min(lo - 0.02, -0.04), hi + 0.03)
    step = 10 if hi - lo > 0.3 else 5
    dom = (-(int(-dom[0] * 100 / step) + 1) * step / 100, (int(dom[1] * 100 / step) + 1) * step / 100)
    return C.forest(rows, cls, dom, step, "Ganancia de cada candidata frente a la línea base, por rival", labels)


def heat(rows, cols, cell, selfcell="—"):
    """cell(row, col) -> (value|None, text, title)"""
    h = '<div class="tw"><table class="hm"><thead><tr><th></th>' + "".join(f'<th>{C.esc(c[1])}</th>' for c in cols) + "</tr></thead><tbody>"
    for r in rows:
        h += f'<tr><th class="rl">{C.esc(r[1])}</th>'
        for c in cols:
            v, text, title = cell(r[0], c[0])
            if v is None:
                h += f'<td class="self">{selfcell}</td>'
            else:
                hi, lo = max(0, (v - .5) / .5), max(0, (.5 - v) / .5)
                h += f'<td style="--hi:{hi:.3f};--lo:{lo:.3f}" title="{C.esc(title)}">{text}</td>'
        h += "</tr>"
    return h + "</tbody></table></div>"


def chart(name_):
    if name_ == "gain1": return C.gain_chart(series("r1"), "c1", gain_dom(), "Ganancia por generación, corrida 1")
    if name_ == "gain2": return C.gain_chart(series("r2"), "c2", gain_dom(), "Ganancia por generación, corrida 2")
    if name_ == "gain3": return C.gain_chart(series("r3"), "c3", gain_dom(), "Ganancia por generación, corrida 3")
    if name_ == "loss12":
        a = lines_for("r1", "loss"); b = lines_for("r2", "loss")
        return C.line_chart([("c1", "Corrida 1", a), ("c2", "Corrida 2", b)], (0.70, 0.88), 0.02, lambda v: f"{v:.2f}",
                            "Pérdida por generación, corridas 1 y 2", xmax=50, ylab="pérdida")
    if name_ == "loss3":
        return C.line_chart([("c3", "Corrida 3", lines_for("r3", "loss"))], (0.18, 0.22), 0.01, lambda v: f"{v:.2f}",
                            "Pérdida por generación, corrida 3", xmax=100, ylab="pérdida")
    if name_ == "gd":
        return C.line_chart([("c3", "Corrida 3", lines_for("r3", "gd"))], (0, 0.6), 0.1, lambda v: f"{v*100:.0f}%",
                            "Desacuerdo con la acción sin azar, corrida 3", xmax=100, ylab="% decisiones",
                            refs=[(D["pilot.t005.gd"]["v"], "piloto con temperatura 0.05")])
    if name_ == "speed":
        return C.line_chart([(c, l, lines_for(r, "rate")) for c, l, r in (("c1", "Corrida 1", "r1"), ("c2", "Corrida 2", "r2"), ("c3", "Corrida 3", "r3"))],
                            (0, 350), 50, lambda v: f"{v:.0f}", "Partidas por segundo por generación", xmax=100, ylab="partidas/s")
    if name_ == "evo":
        evo = D["evo"]["v"]; h = ""
        for ri, cls in (("run1", "c1"), ("run2", "c2"), ("run3", "c3")):
            e = evo[ri]; n = ri[-1]
            gens = [re.search(r"g(\d+)$", a).group(1) for a in e["agents"]]
            h += f'<h4 style="margin:14px 0 6px" class="t{n}">Corrida {n} · generaciones {", ".join(gens)}</h4><div class="evo">'
            for opp in e["opponents"]:
                pts = [e["table"][a].get(opp) for a in e["agents"]]
                nm = name(opp)
                h += f'<figure><figcaption>vs {C.esc(nm)}</figcaption>' + C.evo_panel(cls, f"{nm}, corrida {n}", gens, pts, lambda v: f"{v*100:.0f}") + "</figure>"
            h += "</div>"
        return h
    if name_ == "within":
        evo = D["evo"]["v"]; h = '<div class="cols">'
        for ri in ("run1", "run2", "run3"):
            e = evo[ri]; n = ri[-1]
            rows = [(a, f"gen {re.search(r'g(\d+)$', a).group(1)}") for a in e["agents"]]
            def cell(r, c, e=e):
                if r == c or c not in e["within"][r]:
                    return (None, "", "")
                v = e["within"][r][c]
                return (v[0], f"{v[0]*100:.1f}", f"{name(r)} vs {name(c)}: {v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}]")
            h += f'<div><h4 class="t{n}" style="margin-bottom:6px">Corrida {n}</h4>' + heat(rows, rows, cell) + "</div>"
        return h + "</div><div class=\"scale\"><span>gana la columna</span><i></i><span>gana la fila</span></div>"
    if name_ == "partT":
        g = [(lab, [(D[f"pt.au.{t}"]["v"]["mean"], D[f"pt.au.{t}"]["v"]["lower"], D[f"pt.au.{t}"]["v"]["upper"]),
                    (D[f"pt.p1.{t}"]["v"]["mean"], D[f"pt.p1.{t}"]["v"]["lower"], D[f"pt.p1.{t}"]["v"]["upper"])])
             for lab, t in (("Temperatura 0.05", "0.05"), ("Temperatura 0.02", "0.02"), ("Sin azar", "greedy"))]
        return C.grouped_bars(g, ["Utilidad adaptativa", "Genético publicado"], "c2", 1, "Gen 38 de la corrida 2 según la forma de elegir la acción")
    if name_ == "forest2": return forest_for("e2", "c2", ["gen 38", "gen 49"])
    if name_ == "forest3": return forest_for("e3", "c3", ["gen 73", "gen 100"])
    if name_ == "tbl2": return tbl_exam("e2","gen 38","gen 49")
    if name_ == "tbl3": return tbl_exam("e3","gen 73","gen 100")
    if name_ == "avs":
        ag = D["avs.agents"]["v"]; mat = D["avs.pairs"]["v"]
        rows = [(a, name(a)) for a in ag]
        def cell(r, c):
            if r == c: return (None, "", "")
            v = mat[f"{r}|{c}"]
            used.append({"fmt": "n1", "path": f"avs.pairs@{r}|{c}#0", "text": F("n1", v[0])})
            return (v[0], F("n1", v[0]), f"{name(r)} vs {name(c)}: {v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}]")
        return heat(rows, [(a, short(a)) for a in ag], cell) + '<div class="scale"><span>gana la columna</span><i></i><span>gana la fila</span></div>'
    if name_ == "rank":
        rk = D["avs.rank"]["v"]; order = sorted(rk, key=lambda a: -rk[a][0])
        h = '<div class="tw"><table><thead><tr><th class="n">#</th><th>Participante</th><th>Media contra los otros 7</th><th class="n">Puntaje tipo Elo</th></tr></thead><tbody>'
        for i, a in enumerate(order, 1):
            w = rk[a][0] * 180
            h += (f'<tr><td class="n">{i}</td><td>{C.esc(name(a))}</td><td class="barcell"><i class="bar" style="width:{w:.0f}px"></i>'
                  f'<span class="num">{V("pc1", f"avs.rank@{a}#0")}</span></td><td class="n">{V("sf0", f"avs.rank@{a}#1")}</td></tr>')
        return h + "</tbody></table></div>"
    raise KeyError(name_)


def pill(m):
    key = m.group(1)
    passes, guard, holm, bal = get(key.replace("e3.C1", "e3.passes.C1").replace("e3.C2", "e3.passes.C2"))
    ic_ok = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.5 2.5L10 3.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'
    ic_no = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'
    if passes:
        return f'<span class="pill ok">{ic_ok}Aprueba</span>'
    cand = key.split(".")[-1]
    fails = sum(1 for o in ("R", "P1", "H1", "H2") if D[f"e3.gain.{cand}.{o}"]["v"][1] <= -0.02)
    what = "falla una salvaguarda" if fails == 1 else f"falla {('dos', 'tres', 'cuatro')[fails - 2]} salvaguardas"
    return f'<span class="pill no">{ic_no}No aprueba: {what}</span>'


html = (HERE / "template.html").read_text(encoding="utf-8")
html = html.replace("/*CSS*/", (HERE / "style.css").read_text(encoding="utf-8"))
html = re.sub(r"\{\{chart:(\w+)\}\}", lambda m: chart(m.group(1)), html)
html = re.sub(r"\{\{pill:([\w.]+)\}\}", pill, html)
html = re.sub(r"\[\[(\w+):([^\]]+)\]\]", ph, html)
assert "[[" not in html and "{{" not in html, re.findall(r".{30}(?:\[\[|\{\{).{30}", html)[:3]
(HERE / "reporte-fase2.html").write_text(html, encoding="utf-8")
(HERE / "used.json").write_text(json.dumps(used, ensure_ascii=False, indent=0), encoding="utf-8")
print("html bytes", len(html.encode("utf-8")), "placeholders", len(used))
