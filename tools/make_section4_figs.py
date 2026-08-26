# -*- coding: utf-8 -*-
"""Two SVG figures for dissertation section 4.

generation_flow.svg   comment on 4a, the five generation stages from the four
                      frozen pools to the master record, numbered to
                      match the 1) to 4) list in the 4a lead.
two_tier_training.svg comment on 4c, Tier 1 and Tier 2 training and how the
                      profit maximising bid rule is assembled from them.

House style follows make_censoring_map.py, raw SVG assembly, Segoe UI,
register-role colours, orthogonal edges, market blocks in the v1 auction
orange, and the recommended-bid pill wears a sixth decision green #E8F5E9/#2E7D32
declared here. Natural size is emitted at half the viewBox so a docx embed lands
near 6.5 inches wide. Self-audit asserts every label fits its box at 0.58 em width.
"""
from __future__ import annotations

import pathlib
from xml.sax.saxutils import escape

import yaml

# The master record's width, read from the document that defines it. Typed
# into three captions until 23 August 2026, where it went stale at 55 the
# moment v2.2 added `min_winning_price` and the parquets became 56.
NCOL = len(yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[1] / "config" / "graph.yaml")
    .read_text(encoding="utf-8"))["column_order"])

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "diagrams"

FONT = "Segoe UI, Arial, sans-serif"
OBSF, OBSS = "#FFFFFF", "#607D8B"
LATF, LATS = "#BBDEFB", "#1976D2"
TRUF, TRUS = "#E1BEE7", "#8E24AA"
OUTF, OUTS = "#FFCDD2", "#D32F2F"
AUCF, AUCS = "#FFE0B2", "#EF6C00"
GREY, INK, SUB = "#78909C", "#212121", "#546E7A"

_audit: list[str] = []


def text(x, y, s, size=12, weight=400, fill=INK, anchor="start"):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{a}>{escape(s)}</text>')


def fits(label, width, size, where):
    est = len(label) * size * 0.58
    assert est <= width, f"text overflow in {where}: '{label}' {est:.0f}px > {width}px"
    _audit.append(f"fit ok [{where}] '{label[:30]}' {est:.0f}/{width}px")


def box(s, x, y, w, h, fill, stroke, title, lines, size=11.5):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    fits(title, w - 24, 12.5, title)
    s.append(text(x + 12, y + 24, title, 12.5, 700, INK))
    for i, ln in enumerate(lines):
        fits(ln, w - 24, size, title)
        s.append(text(x + 12, y + 44 + i * 17, ln, size, 400, SUB))


def arrow(s, pts, colour=GREY, width=2.2):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        assert x1 == x2 or y1 == y2, f"non-orthogonal segment {pts}"
    p = " ".join(f"{x},{y}" for x, y in pts)
    s.append(f'<polyline points="{p}" fill="none" stroke="{colour}" '
             f'stroke-width="{width}" marker-end="url(#ar)"/>')


def defs():
    return (f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
            f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{GREY}"/></marker></defs>')


# ------------------------------------------------------------ generation flow
def draw_generation():
    W, H = 1240, 320
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W // 2}" height="{H // 2}" '
         f'viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]
    s.append(text(W / 2, 34, "How one auction row is generated", 21, 700, INK, "middle"))
    s.append(text(W / 2, 52, "five stages, in generation order, ending in the %d column master record" % NCOL,
                  12, 400, SUB, "middle"))

    stages = [
        ("1)  four frozen pools", ["LU users, LA apps,", "LC campaigns, LR rivals"], LATF, LATS),
        ("2)  observable features", ["A user and device,", "B app, C campaign, D time"], OBSF, OBSS),
        ("3)  true probabilities", ["p click, p install, p pay,", "spend head, EV truth"], TRUF, TRUS),
        ("4)  the market settles", ["floor, rival bids, own bid,", "win and winning price (H)"], AUCF, AUCS),
        ("5)  funnel outcomes", ["E click, F install, G payer", "and spend, on every row"], OUTF, OUTS),
    ]
    BW, BH, GAP, Y = 212, 96, 28, 90
    x0 = (W - (BW * 5 + GAP * 4)) / 2
    for i, (title, lines, fill, stroke) in enumerate(stages):
        x = x0 + i * (BW + GAP)
        box(s, x, Y, BW, BH, fill, stroke, title, lines, 10.5)
        if i:
            arrow(s, [(x - GAP, Y + BH / 2), (x, Y + BH / 2)])

    # the master record, below, fed by stages 2 to 5
    MX, MY, MW, MH = (W - 560) / 2, 232, 560, 58
    s.append(f'<rect x="{MX}" y="{MY}" width="{MW}" height="{MH}" rx="10" '
             f'fill="#FAFAFA" stroke="{INK}" stroke-width="2.2"/>')
    s.append(text(MX + MW / 2, MY + 25, "the master record, %d columns" % NCOL, 13, 700, INK, "middle"))
    s.append(text(MX + MW / 2, MY + 43,
                  "one parquet per seed and scale, latents and truth retained for scoring only",
                  10.5, 400, SUB, "middle"))
    lastx = x0 + 4 * (BW + GAP) + BW / 2
    arrow(s, [(lastx, Y + BH), (lastx, MY - 24), (W / 2 + MW / 2 - 40, MY - 24),
              (W / 2 + MW / 2 - 40, MY)])
    s.append("</svg>")
    return "\n".join(s), W, H


# --------------------------------------------------------- two tier training
def draw_two_tier():
    W, H = 1240, 430
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W // 2}" height="{H // 2}" '
         f'viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', defs()]
    s.append(text(W / 2, 34, "Two tiers, one bidding algorithm", 21, 700, INK, "middle"))
    s.append(text(W / 2, 52, "the same training runs under each view, only the visible data differs",
                  12, 400, SUB, "middle"))

    # left, the training data
    box(s, 50, 150, 240, 120, "#FAFAFA", GREY, "training split, view k",
        ["the rows and columns this", "view is allowed to see,", "censoring applied"], 10.5)

    # middle, the two tiers
    box(s, 380, 90, 340, 128, OUTF, OUTS, "Tier 1, the value of an impression",
        ["four heads, E1) p(click), F1) p(install | click),", "G1) p(payer | install), G2) spend of a payer,",
         "multiplied into the value estimate EV"], 10.5)
    box(s, 380, 250, 340, 110, AUCF, AUCS, "Tier 2, the cost of winning",
        ["one head, p(win at bid b), label H3) won,", "inputs every view feature, H2) the bid",
         "swept monotone, H1) the floor among them"], 10.5)

    # right, the assembly
    box(s, 810, 128, 380, 168, "#FAFAFA", INK, "profit_max, the bid rule",
        ["sweep the ladder of 48 candidate prices,", "at each rung b score  (EV - b) x p(win at b),",
         "recommend the rung with the highest score,", "when no rung profits the chosen rung rarely wins"], 10.5)
    s.append(f'<rect x="900" y="330" width="200" height="40" rx="20" '
             f'fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.8"/>')
    fits("recommended bid", 176, 12, "bid pill")
    s.append(text(1000, 355, "recommended bid", 12, 700, "#2E7D32", "middle"))

    arrow(s, [(290, 180), (335, 180), (335, 154), (380, 154)])
    arrow(s, [(290, 240), (335, 240), (335, 305), (380, 305)])
    arrow(s, [(720, 154), (765, 154), (765, 190), (810, 190)])
    arrow(s, [(720, 305), (765, 305), (765, 250), (810, 250)])
    arrow(s, [(1000, 296), (1000, 330)])

    s.append(text(50, H - 20, "the composition is arithmetic, nothing after the two tiers is learned",
                  10.5, 400, SUB))
    s.append("</svg>")
    return "\n".join(s), W, H


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("generation_flow.svg", draw_generation),
                     ("two_tier_training.svg", draw_two_tier)):
        svg, w, h = fn()
        p = OUTDIR / name
        p.write_text(svg, encoding="utf-8")
        print(f"wrote {p}  {w}x{h}  {len(svg)} bytes")
    print(f"text fit checks passed: {len(_audit)}")
    print("PROBLEMS: NONE")
