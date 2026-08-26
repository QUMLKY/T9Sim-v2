# -*- coding: utf-8 -*-
"""Two SVG figures for dissertation section 3b, the censoring map.

WHY THIS EXISTS. The 3b formula D_full = (x, z, f, b, w, m, N, y, v*) with
D_obs_k = C_k(D_full) reads as nine scalars. It is nine BLOCKS of columns, and
the two masks inside C_k act on two of them. That block-and-mask reading tripped
the author once, so it will trip an examiner. These figures say it in one look.

TWO LAYOUTS, ONE RULES LOADER.
  censoring_matrix.svg  a reference grid, rows Oracle and C1-C4, columns the
                        tuple blocks, cells visible / won rows only / hidden,
                        with two labelled brackets naming the masks.
  censoring_flow.svg    the mechanism, master record -> the operator C_k with
                        its two gates -> four view cards of surviving blocks.

GROUND TRUTH. Block states are DERIVED from t9v2/config/graph.yaml observability
entries, then asserted against the schedule in specification 1.5 (all / won /
none per view). If the config and this figure ever disagree, the script dies
rather than drawing a lie. user_id (master lineage) and non-column auxiliaries
are excluded, matching the deposit.

STYLE. House rules: Segoe UI, raw SVG assembly, register-role colours as in
make_price_dag.py, hidden cells hatched with explicit 45 degree segments (an
SVG <pattern> rasterises badly in print-to-PDF), edges orthogonal, legend strip
filtered to states actually used. The market blocks (f b w, m N) wear the v1
auction-family orange #FFE0B2/#EF6C00, a fifth role make_price_dag.py does not
carry; declared here so the palettes are known to differ on H nodes.
"""
from __future__ import annotations

import pathlib
import sys
from xml.sax.saxutils import escape

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPH = ROOT / "config" / "graph.yaml"
OUTDIR = ROOT / "docs" / "diagrams"

FONT = "Segoe UI, Arial, sans-serif"
VIEWS = ["C1", "C2", "C3", "C4"]
VIEW_CAPTION = {"C1": "DSP alone", "C2": "DSP + MMP", "C3": "DSP + SSP", "C4": "all three"}

# register-role palette, per make_price_dag.py
OBSF, OBSS = "#FFFFFF", "#607D8B"
LATF, LATS = "#BBDEFB", "#1976D2"
TRUF, TRUS = "#E1BEE7", "#8E24AA"
OUTF, OUTS = "#FFCDD2", "#D32F2F"
AUCF, AUCS = "#FFE0B2", "#EF6C00"
GREY = "#78909C"
INK = "#212121"
SUB = "#546E7A"

# the six tuple blocks, in document order, with their register mapping
BLOCKS = [
    ("x", "observables", "A B C D", OBSF, OBSS),
    ("f b w", "floor, bid, won", "H1 H2 H3", AUCF, AUCS),
    ("y", "funnel outcomes", "E F G", OUTF, OUTS),
    ("m N", "price, rivals", "H4 H9", AUCF, AUCS),
    ("z", "latents", "LU LA LC LR", LATF, LATS),
    ("v*", "truth", "heads, EV", TRUF, TRUS),
]

# specification 1.5 schedule the derived states must reproduce
EXPECTED = {
    "x":     {v: "all" for v in VIEWS},
    "f b w": {v: "all" for v in VIEWS},
    "y":     {"C1": "won", "C2": "all", "C3": "won", "C4": "all"},
    "m N":   {"C1": "none", "C2": "none", "C3": "all", "C4": "all"},
    "z":     {v: "none" for v in VIEWS},
    "v*":    {v: "none" for v in VIEWS},
}


def block_of(node) -> str | None:
    """Map a graph.yaml node to its tuple block, or None if it ships no view."""
    nid, role = node["id"], node["role"]
    if node.get("kind") != "column":
        return None
    if "user_id" in node.get("columns", []):
        return None                      # master lineage only, never in a view
    if nid in ("H1", "H2", "H3"):
        return "f b w"
    if nid in ("H4", "H9"):
        return "m N"
    if role == "latent":
        return "z"
    if role in ("truth", "estimand"):
        return "v*"
    if role in ("out", "outcome"):
        return "y"
    if role == "observable":
        return "x"
    return None                          # auxiliaries with columns, if any


def load_states():
    g = yaml.safe_load(GRAPH.read_text(encoding="utf-8"))
    # the caption counts the master record, so it counts the same list the
    # generator emits rather than a number typed beside the drawing
    NCOL = len(g["column_order"])
    per_block: dict[str, dict[str, set]] = {b[0]: {v: set() for v in VIEWS} for b in BLOCKS}
    counts: dict[str, int] = {b[0]: 0 for b in BLOCKS}
    for node in g["nodes"]:
        blk = block_of(node)
        if blk is None:
            continue
        counts[blk] += 1
        obs = node["observability"]
        for v in VIEWS:
            state = "none" if obs == "latent" else obs[v]
            per_block[blk][v].add(state)
    states: dict[str, dict[str, str]] = {}
    for blk, per_view in per_block.items():
        states[blk] = {}
        for v, seen in per_view.items():
            assert len(seen) == 1, f"block {blk} view {v} mixes states {seen}"
            states[blk][v] = seen.pop()
    assert states == EXPECTED, f"graph.yaml disagrees with spec 1.5:\n{states}"
    return states, counts


# ---------------------------------------------------------------- primitives
def hatch(x, y, w, h, step=7, colour="#B0BEC5"):
    """Explicit 45 degree hatch segments clipped to the rect by hand."""
    out = []
    d = -h + step
    while d < w:
        # segment from (x+d, y+h) to (x+d+h, y); clip both ends to the rect
        ax, ay = x + d, y + h
        bx, by = x + d + h, y
        if ax < x:
            ay -= (x - ax); ax = x
        if bx > x + w:
            by += (bx - (x + w)); bx = x + w
        if ax < bx:
            out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                       f'stroke="{colour}" stroke-width="1.1"/>')
        d += step
    return "\n".join(out)


def text(x, y, s, size=12, weight=400, fill=INK, anchor="start", style=""):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    st = f' {style}' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{a}{st}>{escape(s)}</text>')


def lighten(hexcol, f=0.55):
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(c + (255 - c) * f)
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


# ------------------------------------------------------------------- matrix
def draw_matrix(states):
    M, LABW, COLW, ROWH = 34, 200, 128, 46
    HEADH, BRACKH, TITLEH, FOOTH, LEGH = 56, 44, 60, 46, 40
    W = M * 2 + LABW + COLW * len(BLOCKS)
    H = TITLEH + BRACKH + HEADH + ROWH * 5 + FOOTH + LEGH
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W // 2}" height="{H // 2}" '
         f'viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']
    s.append(text(W / 2, 34, "One master record, four views", 21, 700, INK, "middle"))
    s.append(text(W / 2, 52, "which blocks of D_full survive each censoring operator C_k",
                  12, 400, SUB, "middle"))

    gx = M + LABW
    gy = TITLEH + BRACKH + HEADH
    cx = {b[0]: gx + i * COLW for i, b in enumerate(BLOCKS)}

    # mask brackets over y and m N
    for blk, label in (("y", "outcome mask (MMP)"), ("m N", "price mask (SSP)")):
        x0, x1 = cx[blk] + 8, cx[blk] + COLW - 8
        yb = TITLEH + 14
        s.append(f'<polyline points="{x0},{yb + 12} {x0},{yb} {x1},{yb} {x1},{yb + 12}" '
                 f'fill="none" stroke="{INK}" stroke-width="1.6"/>')
        s.append(text((x0 + x1) / 2, yb - 5, label, 10.5, 700, INK, "middle"))

    # column headers
    hy = TITLEH + BRACKH
    for sym, name, fam, _, stroke in BLOCKS:
        x = cx[sym]
        s.append(text(x + COLW / 2, hy + 20, sym, 14, 700, stroke, "middle"))
        s.append(text(x + COLW / 2, hy + 35, name, 10.5, 400, SUB, "middle"))
        s.append(text(x + COLW / 2, hy + 48, fam, 9.5, 400, "#90A4AE", "middle"))

    rows = [("Oracle", "the evaluator", None)] + [(v, VIEW_CAPTION[v], v) for v in VIEWS]
    used = set()
    for r, (name, caption, view) in enumerate(rows):
        y = gy + r * ROWH
        if r % 2 == 0:
            s.append(f'<rect x="{M}" y="{y}" width="{W - 2 * M}" height="{ROWH}" fill="#F7F9FA"/>')
        s.append(text(M + 8, y + ROWH / 2 + 1, name, 13, 700,
                      GREY if view is None else INK))
        s.append(text(M + 8, y + ROWH / 2 + 15, caption, 10, 400, SUB))
        for sym, _, _, fill, stroke in BLOCKS:
            x = cx[sym]
            cw, ch = COLW - 16, ROWH - 12
            rx0, ry0 = x + 8, y + 6
            state = "all" if view is None else states[sym][view]
            used.add(state)
            if state == "none":
                s.append(f'<rect x="{rx0}" y="{ry0}" width="{cw}" height="{ch}" rx="6" '
                         f'fill="#FFFFFF" stroke="#CFD8DC" stroke-width="1.2"/>')
                s.append(hatch(rx0 + 2, ry0 + 2, cw - 4, ch - 4))
            else:
                f = fill if state == "all" else lighten(fill, 0.55) if fill != OBSF else OBSF
                s.append(f'<rect x="{rx0}" y="{ry0}" width="{cw}" height="{ch}" rx="6" '
                         f'fill="{f}" stroke="{stroke}" stroke-width="1.6"'
                         + (' stroke-dasharray="5,3"' if state == "won" else "") + '/>')
                word = "all rows" if state == "all" else "won rows only"
                s.append(text(rx0 + cw / 2, ry0 + ch / 2 + 4, word, 10, 700, stroke, "middle"))

    # frame and gridlines
    s.append(f'<rect x="{M}" y="{gy}" width="{W - 2 * M}" height="{ROWH * 5}" '
             f'fill="none" stroke="#B0BEC5" stroke-width="1.4"/>')
    s.append(f'<line x1="{M}" y1="{gy + ROWH}" x2="{W - M}" y2="{gy + ROWH}" '
             f'stroke="#78909C" stroke-width="1.6"/>')  # oracle divider

    fy = gy + ROWH * 5 + 20
    s.append(text(M, fy, "m is the DSP's own bid on won rows and does not exist on unsold rows.",
                  10.5, 400, SUB))
    s.append(text(M, fy + 15, "z and v* reach no view. They score models, they never train them.",
                  10.5, 400, SUB))

    ly = H - 18
    lx = M
    legend = [("all rows", "visible", False), ("won rows only", "dashed border, lighter fill", True),
              ("hidden", "hatched", None)]
    s.append(text(lx, ly, "cell states:", 10, 700, "#37474F")); lx += 78
    for word, gloss, dash in legend:
        s.append(text(lx, ly, f"{word} = {gloss}", 10, 400, "#37474F"))
        lx += 9 + len(f"{word} = {gloss}") * 5.4 + 20
    s.append("</svg>")
    return "\n".join(s), W, H


# --------------------------------------------------------------------- flow
def draw_flow(states):
    W, H = 1240, 560
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W // 2}" height="{H // 2}" '
         f'viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
         f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{GREY}"/></marker></defs>']
    s.append(text(W / 2, 34, "The censoring operator, block by block", 21, 700, INK, "middle"))
    s.append(text(W / 2, 52, "one master record, two masks, four views", 12, 400, SUB, "middle"))

    # left, the master stack
    SX, SY, SW, RH = 60, 100, 240, 54
    s.append(text(SX + SW / 2, SY - 14, "master record D_full, %d columns" % NCOL, 12.5, 700, INK, "middle"))
    for i, (sym, name, fam, fill, stroke) in enumerate(BLOCKS):
        y = SY + i * (RH + 8)
        s.append(f'<rect x="{SX}" y="{y}" width="{SW}" height="{RH}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        s.append(text(SX + 14, y + 23, f"{sym}  {name}", 12.5, 700, INK))
        s.append(text(SX + 14, y + 40, fam, 10, 400, SUB))
    s.append(text(SX, SY + 6 * (RH + 8) + 16,
                  "the oracle evaluator reads all of it", 10.5, 400, SUB))

    # middle, the operator with two gates
    OX, OY, OW, OH = 420, 130, 300, 300
    s.append(f'<rect x="{OX}" y="{OY}" width="{OW}" height="{OH}" rx="10" '
             f'fill="#FAFAFA" stroke="{INK}" stroke-width="1.8"/>')
    s.append(text(OX + OW / 2, OY + 26, "censoring operator C_k", 13.5, 700, INK, "middle"))
    gates = [("outcome mask", "hides y on lost rows", "lifted by the MMP layer (C2, C4)", OUTS),
             ("price mask", "hides m and N", "lifted by the SSP layer (C3, C4)", AUCS)]
    for i, (gname, does, lifts, col) in enumerate(gates):
        gy = OY + 46 + i * 86
        s.append(f'<rect x="{OX + 18}" y="{gy}" width="{OW - 36}" height="{72}" rx="8" '
                 f'fill="#FFFFFF" stroke="{col}" stroke-width="1.8"/>')
        s.append(text(OX + 32, gy + 22, gname, 12.5, 700, col))
        s.append(text(OX + 32, gy + 40, does, 11, 400, INK))
        s.append(text(OX + 32, gy + 57, lifts, 11, 400, SUB))
    s.append(text(OX + OW / 2, OY + OH - 42, "z and v* are deleted in every view,",
                  11, 700, TRUS, "middle"))
    s.append(text(OX + OW / 2, OY + OH - 26, "x, f, b and w pass through untouched",
                  11, 400, SUB, "middle"))

    # edges master -> operator -> cards
    midy = OY + OH / 2
    s.append(f'<polyline points="{SX + SW},{midy} {OX},{midy}" fill="none" '
             f'stroke="{GREY}" stroke-width="2.2" marker-end="url(#ar)"/>')

    # right, four view cards
    CX, CW, CH, GAP = 800, 380, 96, 14
    for i, v in enumerate(VIEWS):
        y = 80 + i * (CH + GAP)
        s.append(f'<rect x="{CX}" y="{y}" width="{CW}" height="{CH}" rx="10" '
                 f'fill="#FFFFFF" stroke="#90A4AE" stroke-width="1.8"/>')
        s.append(text(CX + 16, y + 24, v, 14, 700, INK))
        s.append(text(CX + 48, y + 24, VIEW_CAPTION[v], 11, 400, SUB))
        # surviving blocks as pills
        px = CX + 16
        for sym, name, fam, fill, stroke in BLOCKS:
            st = states[sym][v]
            if st == "none":
                continue
            label = sym if st == "all" else f"{sym} won rows"
            wpx = 16 + len(label) * 6.6
            dash = ' stroke-dasharray="5,3"' if st == "won" else ""
            s.append(f'<rect x="{px}" y="{y + 38}" width="{wpx:.0f}" height="24" rx="12" '
                     f'fill="{fill if st == "all" else lighten(fill)}" stroke="{stroke}" '
                     f'stroke-width="1.5"{dash}/>')
            s.append(text(px + wpx / 2, y + 54, label, 10.5, 700, stroke, "middle"))
            px += wpx + 10
        lacks = [b[0] for b in BLOCKS if states[b[0]][v] == "none"]
        s.append(text(CX + 16, y + 82, "never sees  " + ", ".join(lacks), 10.5, 400, SUB))
        ey = y + CH / 2
        sy = min(max(ey, OY + 16), OY + OH - 16)   # leave the operator box, not thin air
        bus = (OX + OW + CX) / 2
        s.append(f'<polyline points="{OX + OW},{sy} {bus},{sy} {bus},{ey} {CX},{ey}" '
                 f'fill="none" stroke="{GREY}" stroke-width="2.0" marker-end="url(#ar)"/>')

    # legend, mirroring the matrix
    ly, lx = H - 18, 60
    s.append(text(lx, ly, "pill states:", 10, 700, "#37474F")); lx += 78
    for entry in ("solid = all rows", "dashed and lighter = won rows only",
                  "absent = never in the view"):
        s.append(text(lx, ly, entry, 10, 400, "#37474F"))
        lx += 9 + len(entry) * 5.4 + 22
    s.append("</svg>")
    return "\n".join(s), W, H


if __name__ == "__main__":
    states, counts = load_states()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    audit = [f"block column-node counts: {counts}"]
    for name, fn in (("censoring_matrix.svg", draw_matrix), ("censoring_flow.svg", draw_flow)):
        svg, w, h = fn(states)
        p = OUTDIR / name
        p.write_text(svg, encoding="utf-8")
        audit.append(f"wrote {p}  {w}x{h}  {len(svg)} bytes")
    audit.append("derived states match specification 1.5 schedule: PASS")
    print("\n".join(audit))
    print("PROBLEMS: NONE")
