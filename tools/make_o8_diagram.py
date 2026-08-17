# -*- coding: utf-8 -*-
"""Draw the open item O8 explainer: why an unconstrained win curve picks the wrong bid.

    python tools/make_o8_diagram.py

Writes "Schema diagrams/O8_monotone_win_curve.svg".

The curves are COMPUTED, not sketched. The constrained curve is the win curve the
project actually uses, Phi((log b - log m) / sigma), which is the form at
pipeline.py:747. The unconstrained curve is the same curve with a step-shaped dent
of the kind a gradient-boosted tree fits when a region of the bid grid is thin, and
it is a DEMONSTRATION of the mechanism rather than a measurement of v1.

The point of the picture is the right-hand panel: the dent is small and looks
harmless on the win curve, but the bidder maximises (ev - b) x p(win|b), and that
product turns a small dent into a different chosen bid.
"""
import io
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parents[2] / "Schema diagrams" / "O8_monotone_win_curve.svg"

FONT = "Segoe UI, Verdana, sans-serif"
INK, MUTE, GRID = "#263238", "#607D8B", "#ECEFF1"
GOOD, BAD, EV = "#2E7D32", "#C62828", "#1565C0"

EV_TRUE = 0.060          # the row's expected value, USD per impression
M, SIGMA = 0.022, 0.75   # win-curve location and scale
LO, HI, N = 0.004, 0.100, 400

# a tree's dent: a sag over one stretch of the grid, flat-bottomed like a split
DENT_LO, DENT_HI, DENT_DEPTH = 0.019, 0.037, 0.105


def pwin(b):
    return 0.5 * (1 + math.erf((math.log(b) - math.log(M)) / (SIGMA * math.sqrt(2))))


def pwin_unconstrained(b):
    if not (DENT_LO <= b <= DENT_HI):
        return pwin(b)
    t = (b - DENT_LO) / (DENT_HI - DENT_LO)          # 0..1 across the dent
    shape = math.sin(math.pi * t) ** 0.55            # flat-bottomed, tree-like
    return max(0.0, pwin(b) - DENT_DEPTH * shape)


def profit(b, p):
    return (EV_TRUE - b) * p


def series():
    step = (HI - LO) / (N - 1)
    bs = [LO + i * step for i in range(N)]
    good = [pwin(b) for b in bs]
    bad = [pwin_unconstrained(b) for b in bs]
    return bs, good, bad, [profit(b, p) for b, p in zip(bs, good)], \
        [profit(b, p) for b, p in zip(bs, bad)]


def upto_value(bs, ys):
    """Profit is negative once the bid passes the row's value, and no bidder goes
    there, so the profit panel stops at b = ev rather than running off the canvas."""
    keep = [(b, y) for b, y in zip(bs, ys) if b <= EV_TRUE]
    return [b for b, _ in keep], [y for _, y in keep]


def argmax(bs, ys):
    i = max(range(len(ys)), key=lambda k: ys[k])
    return bs[i], ys[i]


# ---------- drawing helpers -------------------------------------------------
W, H = 1180, 640
PW, PH = 470, 340
AX, AY, BX = 90, 150, 640


class Panel:
    def __init__(self, x, y, w, h, xlo, xhi, ylo, yhi):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yhi

    def px(self, v):
        return self.x + (v - self.xlo) / (self.xhi - self.xlo) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.ylo) / (self.yhi - self.ylo) * self.h

    def path(self, xs, ys):
        return " ".join(("M" if i == 0 else "L") + "%.1f %.1f" % (self.px(x), self.py(y))
                        for i, (x, y) in enumerate(zip(xs, ys)))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12, fill=INK, weight="400", anchor="start", style=""):
    return ('<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
            'text-anchor="%s"%s>%s</text>'
            % (x, y, size, weight, fill, anchor,
               ' font-style="italic"' if style else "", esc(s)))


def main():
    bs, good, bad, pgood, pbad = series()
    b_good, v_good = argmax(bs, pgood)
    b_bad, v_bad = argmax(bs, pbad)
    loss = (v_good - v_bad) / v_good * 100

    A = Panel(AX, AY, PW, PH, LO, HI, 0, 1)
    B = Panel(BX, AY, PW, PH, LO, HI, 0, max(pgood) * 1.18)

    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
         'font-family="%s">' % (W, H, W, H, FONT),
         '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H),
         txt(W / 2, 44, "O8  Why the win classifier needs a monotone constraint",
             21, INK, "700", "middle"),
         txt(W / 2, 70,
             "The same small dent, seen on the win curve and then on the thing the bidder "
             "actually maximises", 13, MUTE, "400", "middle"),
         txt(W / 2, 92,
             "Illustrative curves, computed from the project's own win-curve form. "
             "Not a measurement of v1.", 11.5, MUTE, "400", "middle", style=True)]

    for P, title, sub in ((A, "1.  The win curve  p(win | bid)",
                              "the dent looks minor here"),
                          (B, "2.  Expected profit  (ev − bid) × p(win | bid)",
                              "the dent decides the bid here")):
        s.append(txt(P.x, P.y - 26, title, 14.5, INK, "700"))
        s.append(txt(P.x, P.y - 9, sub, 11.5, MUTE))
        for k in range(6):
            gy = P.y + P.h * k / 5
            s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                     % (P.x, gy, P.x + P.w, gy, GRID))
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.6"/>'
                 % (P.x, P.y + P.h, P.x + P.w, P.y + P.h, MUTE))
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.6"/>'
                 % (P.x, P.y, P.x, P.y + P.h, MUTE))
        for v in (0.02, 0.04, 0.06, 0.08, 0.10):
            s.append(txt(P.px(v), P.y + P.h + 18, "$%.2f" % v, 10.5, MUTE, "400", "middle"))
        s.append(txt(P.x + P.w / 2, P.y + P.h + 38, "bid", 12, INK, "600", "middle"))

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(txt(A.x - 10, A.py(v) + 4, "%.2f" % v, 10.5, MUTE, "400", "end"))
    for k in range(5):
        v = B.yhi * k / 4
        s.append(txt(B.x - 10, B.py(v) + 4, "%.3f" % v, 10.5, MUTE, "400", "end"))
    s.append(txt(B.x - 78, B.y + B.h / 2, "profit", 11.5, MUTE, "600", "middle"))

    # panel 1, the two win curves
    s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.6"/>' % (A.path(bs, good), GOOD))
    s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.6" stroke-dasharray="7 4"/>'
             % (A.path(bs, bad), BAD))
    s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="0.10"/>'
             % (A.px(DENT_LO), A.y, A.px(DENT_HI) - A.px(DENT_LO), A.h, BAD))
    s.append(txt(A.px((DENT_LO + DENT_HI) / 2), A.y + 26,
                 "p(win) dips as the bid rises", 11, BAD, "600", "middle"))
    s.append(txt(A.px((DENT_LO + DENT_HI) / 2), A.y + 42,
                 "which cannot be true", 11, BAD, "400", "middle"))

    # panel 2, the two profit curves, with the chosen bids. Both stop at b = ev.
    gx, gy_ = upto_value(bs, pgood)
    bx, by_ = upto_value(bs, pbad)
    s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.6"/>' % (B.path(gx, gy_), GOOD))
    s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.6" stroke-dasharray="7 4"/>'
             % (B.path(bx, by_), BAD))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.2" '
             'stroke-dasharray="2 4"/>' % (B.px(EV_TRUE), B.y + 40, B.px(EV_TRUE), B.y + B.h, MUTE))
    s.append(txt(B.px(EV_TRUE), B.y + 34, "bid = ev, profit 0", 10, MUTE, "400", "middle"))
    # labels above and below the curve so they cannot collide
    for b, v, col, lab, dy, anch in (
            (b_good, v_good, GOOD, "constrained picks $%.3f" % b_good, -16, "middle"),
            (b_bad, v_bad, BAD, "unconstrained picks $%.3f" % b_bad, 34, "start")):
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4" '
                 'stroke-dasharray="3 3"/>' % (B.px(b), B.py(v), B.px(b), B.y + B.h, col))
        s.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (B.px(b), B.py(v), col))
        s.append(txt(B.px(b) + (6 if anch == "start" else 0), B.py(v) + dy, lab, 11, col, "600", anch))
    s.append(txt(B.x + B.w - 6, B.y + 18,
                 "%.0f%% of the profit forgone" % loss, 12.5, BAD, "700", "end"))

    # legend and closing line
    ly = H - 62
    for i, (col, dash, lab) in enumerate((
            (GOOD, "", "monotone constraint on  —  p(win) can only rise with the bid"),
            (BAD, ' stroke-dasharray="7 4"', "no constraint  —  the fitted curve may sag"))):
        y = ly + i * 21
        s.append('<line x1="60" y1="%.1f" x2="96" y2="%.1f" stroke="%s" stroke-width="2.8"%s/>'
                 % (y, y, col, dash))
        s.append(txt(106, y + 4, lab, 12, INK))
    s.append(txt(W - 60, ly + 4,
                 "Fix: one key, monotone_constraints, on the Tier-2 win classifier.",
                 12, INK, "600", "end"))
    s.append(txt(W - 60, ly + 25,
                 "The bid grid is 48 log-spaced prices, so a sag can sit between two of them.",
                 11.5, MUTE, "400", "end"))
    s.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(s), encoding="utf-8")
    print("written: %s" % OUT.name)
    print("  constrained   picks $%.4f, profit %.5f" % (b_good, v_good))
    print("  unconstrained picks $%.4f, profit %.5f" % (b_bad, v_bad))
    print("  profit forgone %.1f%%" % loss)


if __name__ == "__main__":
    main()
