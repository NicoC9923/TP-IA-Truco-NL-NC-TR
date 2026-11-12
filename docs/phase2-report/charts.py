"""Inline-SVG charts drawn to scale from data.json. Colours come from CSS classes (theme tokens)."""
import math


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo, hi, step):
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10)); t += step
    return out


class Scale:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def fmtpct(v):
    return f"{v*100:.0f}%".replace("-", "−")


def fmtpp(v):
    return f"{v*100:+.0f}".replace("-", "−")


def _axes(W, H, L, R, T, B, y, yticks, yfmt, xticks, x, xlab=None, zero=None):
    s = []
    for t in yticks:
        yy = y(t)
        s.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        s.append(f'<text class="tk" x="{L-7}" y="{yy+4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    s.append(f'<line class="ax" x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}"/>')
    for t in xticks:
        xx = x(t)
        s.append(f'<line class="ax" x1="{xx:.1f}" x2="{xx:.1f}" y1="{H-B}" y2="{H-B+4}"/>')
        s.append(f'<text class="tk" x="{xx:.1f}" y="{H-B+17}" text-anchor="middle">{t}</text>')
    if xlab:
        s.append(f'<text class="tk" x="{(L+W-R)/2:.1f}" y="{H-3}" text-anchor="middle">{esc(xlab)}</text>')
    if zero is not None:
        s.append(f'<line class="zero" x1="{L}" x2="{W-R}" y1="{y(zero):.1f}" y2="{y(zero):.1f}"/>')
    return "\n".join(s)


def gain_chart(series, cls, ydom, title, label_gens=True):
    """Bars = challenger-minus-champion gain at each check; whisker = 95% interval; promoted bars in run colour."""
    W, H, L, R, T, B = 640, 230, 46, 10, 20, 38
    N = max(s["gen"] for s in series)
    x = Scale(0.5, N + 0.5, L, W - R)
    y = Scale(ydom[1], ydom[0], T, H - B)
    step = 5
    yt = nice_ticks(ydom[0] * 100, ydom[1] * 100, step)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    out.append(_axes(W, H, L, R, T, B, lambda v: y(v / 100), yt, lambda v: f"{v:+.0f}".replace("-", "−") if v else "0",
                     [g for g in range(0, N + 1, 10) if g >= 1] if N > 20 else list(range(5, N + 1, 5)), x,
                     "generación", zero=0))
    bw = max(2.0, (W - L - R) / N * 0.62)
    for s in series:
        if "gain" not in s:
            continue
        cx = x(s["gen"])
        top, bot = y(max(0, s["gain"])), y(min(0, s["gain"]))
        klass = cls if s["prom"] else "off"
        out.append(f'<rect class="{klass}" x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(bot-top,0.8):.1f}">'
                   f'<title>gen {s["gen"]}: {s["gain"]*100:+.1f} pp [{s["lo"]*100:+.1f}, {s["hi"]*100:+.1f}]{" · promovida" if s["prom"] else ""}</title></rect>')
        out.append(f'<line class="wh" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(s["lo"]):.1f}" y2="{y(s["hi"]):.1f}"/>')
    if label_gens:
        for s in series:
            if s.get("prom"):
                cx = x(s["gen"])
                out.append(f'<text class="labb" x="{min(cx, W - R - 9):.1f}" y="{y(s["hi"])-4:.1f}" text-anchor="middle">{s["gen"]}</text>')
    out.append(f'<text class="tk" x="{L-34}" y="{T-8}">pp</text>')
    out.append("</svg>")
    return "\n".join(out)


def line_chart(series_list, ydom, ystep, yfmt, title, xmax=None, ylab="", labels=True, refs=(), W=640, H=230, R=70, xstep=None):
    """series_list: [(cls, label, [(x,y),...])]. Direct labels at the right end."""
    L, T, B = 50, 16, 38
    xs = [p[0] for _, _, pts in series_list for p in pts]
    xm = xmax or max(xs)
    x = Scale(1, xm, L, W - R)
    y = Scale(ydom[1], ydom[0], T, H - B)
    yt = nice_ticks(ydom[0], ydom[1], ystep)
    xt = list(range(10, xm + 1, xstep or (10 if xm > 30 else 5))) if xm > 12 else list(range(1, xm + 1))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    out.append(_axes(W, H, L, R, T, B, y, yt, yfmt, xt, x, "generación"))
    for val, lab in refs:
        out.append(f'<line class="ref" x1="{L}" x2="{W-R}" y1="{y(val):.1f}" y2="{y(val):.1f}"/>')
        out.append(f'<text class="tk" x="{W-R-4}" y="{y(val)-5:.1f}" text-anchor="end">{esc(lab)}</text>')
    # nudge end labels apart so they never overlap
    lyd = {}
    last = None
    for cls, label, pts in sorted(series_list, key=lambda t: y(t[2][-1][1])):
        yy = y(pts[-1][1]) + 4
        if last is not None and yy - last < 15:
            yy = last + 15
        lyd[label] = yy; last = yy
    for cls, label, pts in series_list:
        d = " ".join(("M" if i == 0 else "L") + f"{x(px):.1f} {y(py):.1f}" for i, (px, py) in enumerate(pts))
        out.append(f'<path class="ln {cls}" d="{d}"/>')
        if labels:
            lx, ly = pts[-1]
            out.append(f'<circle class="{cls}" cx="{x(lx):.1f}" cy="{y(ly):.1f}" r="3.5"/>')
            out.append(f'<text class="{cls} labb" x="{x(lx)+8:.1f}" y="{lyd[label]:.1f}">{esc(label)}</text>')
    if ylab:
        out.append(f'<text class="tk" x="{L-40}" y="{T-6}">{esc(ylab)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def evo_panel(cls, title, xs_labels, pts, nfmt):
    """Small multiple: win % (0-100) of five generations of one run against one opponent, with 95% interval."""
    W, H, L, R, T, B = 200, 150, 30, 8, 16, 26
    n = len(xs_labels)
    x = Scale(0, n - 1, L + 14, W - R - 14)
    y = Scale(1, 0, T, H - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in (0, 0.5, 1):
        out.append(f'<line class="{"ref" if t==0.5 else "grid"}" x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tk" x="{L-5}" y="{y(t)+4:.1f}" text-anchor="end">{int(t*100)}</text>')
    out.append(f'<line class="ax" x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}"/>')
    run = [(i, p) for i, p in enumerate(pts) if p]
    d = " ".join(("M" if k == 0 else "L") + f"{x(i):.1f} {y(p[0]):.1f}" for k, (i, p) in enumerate(run))
    out.append(f'<path class="ln {cls}" d="{d}" style="stroke-width:1.5"/>')
    for i, lab in enumerate(xs_labels):
        out.append(f'<text class="tk" x="{x(i):.1f}" y="{H-B+15}" text-anchor="middle">{lab}</text>')
    for i, p in run:
        m, lo, hi = p
        xx = x(i)
        out.append(f'<line class="wh" x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(min(hi,1)):.1f}" y2="{y(lo):.1f}"/>')
        out.append(f'<circle class="{cls}" cx="{xx:.1f}" cy="{y(m):.1f}" r="3.6" style="stroke:var(--panel);stroke-width:1.5">'
                   f'<title>{nfmt(m)}% [{nfmt(lo)}, {nfmt(hi)}]</title></circle>')
        ly = y(hi) - 4 if y(hi) > T + 8 else y(lo) + 12
        out.append(f'<text class="lab" style="font-size:10px" x="{xx:.1f}" y="{ly:.1f}" text-anchor="middle">{nfmt(m)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def forest(rows, cls, dom, step, title, cand_labels, guard=-0.02):
    """rows: [(label, [(gain, lo, hi) for each candidate])]; solid dot = first candidate, hollow = second."""
    W = 640
    rh = 46
    L, R, T, B = 190, 14, 22, 34
    H = T + rh * len(rows) + B
    x = Scale(dom[0], dom[1], L, W - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in nice_ticks(dom[0] * 100, dom[1] * 100, step):
        xx = x(t / 100)
        out.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{T-6}" y2="{H-B}"/>')
        out.append(f'<text class="tk" x="{xx:.1f}" y="{H-B+16}" text-anchor="middle">{f"{t:+.0f}".replace("-", chr(8722)) if t else "0"}</text>')
    out.append(f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T-6}" y2="{H-B}"/>')
    if guard is not None:
        out.append(f'<line class="ref" x1="{x(guard):.1f}" x2="{x(guard):.1f}" y1="{T-6}" y2="{H-B}"/>')
        out.append(f'<text class="tk" x="{x(guard)-4:.1f}" y="{T-9}" text-anchor="end">límite −2</text>')
    out.append(f'<text class="tk" x="{(L+W-R)/2:.1f}" y="{H-3}" text-anchor="middle">ganancia frente a la línea base (puntos porcentuales)</text>')
    for i, (lab, cands) in enumerate(rows):
        cy = T + rh * i + rh / 2
        out.append(f'<text class="lab" x="{L-10}" y="{cy+4:.1f}" text-anchor="end">{esc(lab)}</text>')
        for k, (g, lo, hi) in enumerate(cands):
            yy = cy + (-8 if k == 0 else 8)
            out.append(f'<line class="whp {cls}" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" style="fill:none"/>')
            if k == 0:
                out.append(f'<circle class="{cls}" cx="{x(g):.1f}" cy="{yy:.1f}" r="4.5"><title>{cand_labels[k]}: {g*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]</title></circle>')
            else:
                out.append(f'<circle class="{cls} hol" cx="{x(g):.1f}" cy="{yy:.1f}" r="4" style="fill:var(--panel);stroke-width:2"><title>{cand_labels[k]}: {g*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def grouped_bars(groups, series_names, cls, dom_max, title):
    """groups: [(label, [(mean, lo, hi), ...])]; series 0 solid, series 1 outlined. Vertical bars, 0..1."""
    W, H, L, R, T, B = 560, 260, 46, 10, 18, 52
    y = Scale(1, 0, T, H - B)
    n = len(groups)
    gw = (W - L - R) / n
    bw = gw * 0.30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tk" x="{L-7}" y="{y(t)+4:.1f}" text-anchor="end">{int(t*100)}%</text>')
    out.append(f'<line class="ax" x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}"/>')
    for gi, (lab, bars) in enumerate(groups):
        gx = L + gw * gi + gw / 2
        for si, (m, lo, hi) in enumerate(bars):
            bx = gx + (-bw - 3 if si == 0 else 3)
            if si == 0:
                out.append(f'<rect class="{cls}" x="{bx:.1f}" y="{y(m):.1f}" width="{bw:.1f}" height="{y(0)-y(m):.1f}" rx="0"><title>{series_names[si]}: {m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]</title></rect>')
            else:
                out.append(f'<rect class="{cls}" x="{bx+1:.1f}" y="{y(m)+1:.1f}" width="{bw-2:.1f}" height="{y(0)-y(m)-1:.1f}" style="fill:var(--panel);stroke-width:2"><title>{series_names[si]}: {m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]</title></rect>')
            out.append(f'<line class="wh" x1="{bx+bw/2:.1f}" x2="{bx+bw/2:.1f}" y1="{y(hi):.1f}" y2="{y(lo):.1f}"/>')
            out.append(f'<text class="labb" x="{bx+bw/2:.1f}" y="{y(hi)-4:.1f}" text-anchor="middle">{m*100:.1f}</text>')
        out.append(f'<text class="lab" x="{gx:.1f}" y="{H-B+18}" text-anchor="middle">{esc(lab)}</text>')
    out.append("</svg>")
    return "\n".join(out)
