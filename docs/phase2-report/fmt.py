"""Shared number formatting (page builder and verifier use the same one)."""
MINUS = "\u2212"

def _s(x, s):
    return (MINUS + s[1:]) if s.startswith("-") else s

def f(name, v):
    if name == "pc0": return _s(0, f"{v*100:.0f}%")
    if name == "pc1": return _s(0, f"{v*100:.1f}%")
    if name == "pc2": return _s(0, f"{v*100:.2f}%")
    if name == "n1": return _s(0, f"{v*100:.1f}")          # percent value without sign
    if name == "pp1": return _s(0, f"{v*100:+.1f}")        # signed percentage points
    if name == "pp2": return _s(0, f"{v*100:+.2f}")
    if name == "ppu3": return f"{v:.3f}"                   # already in points
    if name == "i": return f"{int(v):,}"
    if name == "f0": return f"{v:.0f}"
    if name == "f1": return f"{v:.1f}"
    if name == "f2": return f"{v:.2f}"
    if name == "f3": return f"{v:.3f}"
    if name == "f4": return f"{v:.4f}"
    if name == "sf0": return _s(0, f"{v:+.0f}")
    if name == "LIST": return ", ".join(str(x) for x in v)
    if name == "raw": return str(v)
    if name == "min1": return f"{v/60:.1f}"
    if name == "mil": return f"{v/1e6:.1f}"                # millions, 1 decimal
    if name == "mil2": return f"{v/1e6:.2f}"
    if name == "x1": return f"{v:.1f}"
    raise KeyError(name)
