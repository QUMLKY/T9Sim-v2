# -*- coding: utf-8 -*-
"""The price mechanism as a DAG, in NODE SCHEMA CODES.

    python tools/make_price_dag.py

What decides `floor_price` (H1) and `lu7_competing_bid` (LU7): every input a node,
labelled with its register code rather than its formal symbol, and coloured by the
ROLE the register gives it. Written for v2 rather than reusing v1's generator,
because the rebuild plan says v2 inherits no code from v1.

Layering is proper: every edge spans exactly one column, so each vertical bend
lands in an empty gutter and no edge crosses an unrelated box.
"""
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"

# The 4 roles the register gives these nodes, plus the outputs. Coloured by ROLE
# and not by family, because the question this diagram answers is "who can see
# what", and role is the answer to that.
TYPE = {
    "observable": ("#FFFFFF", "#607D8B"),
    "latent":     ("#BBDEFB", "#1976D2"),
    "auxiliary":  ("#FFF9C4", "#FBC02D"),
    "truth":      ("#E1BEE7", "#8E24AA"),
    "out":        ("#FFCDD2", "#D32F2F"),
}
LEGEND = [
    ("observable", "observable - a column in every view"),
    ("latent", "latent - hidden in all 4 views"),
    ("auxiliary", "auxiliary - generator-internal, no column"),
    ("truth", "truth / estimand"),
    ("out", "the two price outputs"),
]

CHARW, FS, LH = 6.7, 12.5, 16
PAD_X, PAD_Y, HGAP, VGAP, M = 14, 12, 92, 24, 28
TITLE_H, LEGEND_H, BUS = 62, 72, 26


def _wh(label):
    lines = label.split("\n")
    return (max(max(len(l) for l in lines) * CHARW + 2 * PAD_X, 96),
            len(lines) * LH + 2 * PAD_Y)


def render(title, subtitle, nodes, edges, fname):
    layers = {}
    for k, (lab, typ, lyr) in nodes.items():
        layers.setdefault(lyr, []).append(k)
    order = sorted(layers)

    for a, b in edges:                       # the one rule that keeps it clean
        if nodes[b][2] != nodes[a][2] + 1:
            raise SystemExit("edge %s->%s spans %d layers; layering must be proper"
                             % (a, b, nodes[b][2] - nodes[a][2]))

    size = {k: _wh(v[0]) for k, v in nodes.items()}
    colw = {l: max(size[k][0] for k in layers[l]) for l in order}
    xc, x = {}, M
    for l in order:
        xc[l] = x + colw[l] / 2
        x += colw[l] + HGAP
    W = x - HGAP + M

    colh = {l: sum(size[k][1] for k in layers[l]) + VGAP * (len(layers[l]) - 1)
            for l in order}
    body = max(colh.values())
    pos = {}
    for l in order:
        y = TITLE_H + (body - colh[l]) / 2
        for k in layers[l]:
            w, h = size[k]
            pos[k] = (xc[l], y + h / 2, w, h)
            y += h + VGAP
    H = TITLE_H + body + LEGEND_H
    used = {t for (_, t, _) in nodes.values()}
    legend = [(t, d) for t, d in LEGEND if t in used]
    W = max(W, M + sum(22 + len(d) * 6.0 + 16 for _, d in legend) + M)

    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
         'viewBox="0 0 %.0f %.0f" font-family="Segoe UI, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H),
         '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
         'orient="auto" markerUnits="strokeWidth">'
         '<path d="M0,0 L8,3 L0,6 Z" fill="#455A64"/></marker></defs>',
         '<rect width="%.0f" height="%.0f" fill="#FAFAFA"/>' % (W, H),
         '<text x="%.0f" y="30" text-anchor="middle" font-size="20" '
         'font-weight="700" fill="#263238">%s</text>' % (W / 2, escape(title)),
         '<text x="%.0f" y="49" text-anchor="middle" font-size="12" '
         'fill="#546E7A">%s</text>' % (W / 2, escape(subtitle))]

    for a, b in edges:                       # edges first, so boxes paint over them
        ax, ay, aw, _ = pos[a]
        bx, by, bw, _ = pos[b]
        x1, x2 = ax + aw / 2, bx - bw / 2
        bus = x2 - BUS if x2 - BUS > x1 + 6 else (x1 + x2) / 2
        s.append('<polyline points="%.0f,%.0f %.0f,%.0f %.0f,%.0f %.0f,%.0f" '
                 'fill="none" stroke="#455A64" stroke-width="1.5" '
                 'marker-end="url(#a)"/>' % (x1, ay, bus, ay, bus, by, x2, by))

    for k, (lab, typ, _) in nodes.items():
        cx, cy, w, h = pos[k]
        fill, stroke = TYPE[typ]
        s.append('<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" rx="9" '
                 'fill="%s" stroke="%s" stroke-width="%.1f"/>'
                 % (cx - w / 2, cy - h / 2, w, h, fill, stroke,
                    2.6 if typ == "out" else 1.4))
        lines = lab.split("\n")
        y0 = cy - (len(lines) - 1) * LH / 2
        for i, ln in enumerate(lines):
            s.append('<text x="%.0f" y="%.0f" text-anchor="middle" font-size="%.1f" '
                     'font-weight="%d" fill="#263238">%s</text>'
                     % (cx, y0 + i * LH + 4, FS if i == 0 else FS - 1.5,
                        (700 if typ == "out" else 600) if i == 0 else 400, escape(ln)))

    ly = H - LEGEND_H + 26
    s.append('<text x="%d" y="%.0f" font-size="11" font-weight="700" '
             'fill="#546E7A">REGISTER ROLE</text>' % (M, ly - 4))
    cx = M
    for typ, desc in legend:
        fill, stroke = TYPE[typ]
        s.append('<rect x="%.0f" y="%.0f" width="13" height="13" rx="3" fill="%s" '
                 'stroke="%s"/>' % (cx, ly + 6, fill, stroke))
        s.append('<text x="%.0f" y="%.0f" font-size="10.5" fill="#37474F">%s</text>'
                 % (cx + 18, ly + 16, escape(desc)))
        cx += 22 + len(desc) * 6.0 + 16
    s.append("</svg>")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / fname).write_text("\n".join(s), encoding="utf-8")
    return OUT / fname


# key -> (label, register role, layer).  Labels lead with the SCHEMA CODE.
NODES = {
    "B6":    ("B6\nslot_format", "observable", 0),
    "A3":    ("A3\nos", "observable", 0),
    "A5":    ("A5\ndevice_type", "observable", 0),
    "D1":    ("D1\ntimestamp", "observable", 0),

    "T":     ("eCPM target\n1.00 / 6.50 / 18.00", "auxiliary", 1),
    "seg":   ("segment\nos x device", "auxiliary", 1),
    "day":   ("day index\n0..27", "auxiliary", 1),
    "fsh":   ("floor_shape\niPinYou resample", "auxiliary", 1),
    "psh":   ("pay_shape\niPinYou resample", "auxiliary", 1),
    "evt":   ("ev_truth\nthe oracle value", "truth", 1),
    "B3":    ("B3\nad_exchange", "observable", 1),

    "H1":    ("H1  floor_price\n= floor_shape x target", "out", 2),
    "base":  ("base_e\n= pay_shape x target", "auxiliary", 2),
    "z":     ("z\nstandardised log ev", "auxiliary", 2),
    "LR1":   ("LR1  w_k\nvalue loading", "latent", 2),
    "LR2":   ("LR2  R[s,k]\nretargeting", "latent", 2),
    "LR3":   ("LR3  pace_k(d)\nbudget pacing", "latent", 2),
    "LR4":   ("LR4  pi_ke\nexchange taste", "latent", 2),
    "LR5":   ("LR5  sigma_k\ndispersion", "latent", 2),
    "LR6":   ("LR6  F_k(d)\nflight mask", "latent", 2),

    "bk":    ("b_k\nlog b_k = log base_e\n+ rho(w_k z + b_R R + pace)\n+ sigma_k eps",
              "auxiliary", 3),
    "part":  ("participate_k\nBernoulli, k=0 forced", "auxiliary", 3),

    "LU7":   ("LU7  lu7_competing_bid\n= max b_k over participants", "out", 4),
    "H9":    ("H9  bid_density\n= participant count", "out", 4),
}

EDGES = [
    ("B6", "T"), ("A3", "seg"), ("A5", "seg"), ("D1", "day"),
    ("T", "H1"), ("fsh", "H1"),
    ("T", "base"), ("psh", "base"),
    ("evt", "z"),
    ("seg", "LR2"), ("B3", "LR4"), ("day", "LR3"), ("day", "LR6"),
    ("base", "bk"), ("z", "bk"), ("LR1", "bk"), ("LR2", "bk"),
    ("LR3", "bk"), ("LR5", "bk"),
    ("LR3", "part"), ("LR4", "part"), ("LR6", "part"),
    ("bk", "LU7"), ("part", "LU7"), ("part", "H9"),
]

if __name__ == "__main__":
    p = render(
        "T9Sim v2 — the price mechanism",
        "what decides floor_price (H1) and the competing bid (LU7). "
        "Node schema codes, coloured by register role.",
        NODES, EDGES, "price_mechanism.svg")
    print("wrote %s" % p)
    print("  %d nodes, %d edges, %d layers"
          % (len(NODES), len(EDGES), max(v[2] for v in NODES.values()) + 1))
    from collections import Counter
    for r, n in Counter(v[1] for v in NODES.values()).most_common():
        print("    %-11s %d" % (r, n))
