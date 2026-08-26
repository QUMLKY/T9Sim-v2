# -*- coding: utf-8 -*-
"""Is the bid ladder wide enough? Measured as profit at risk, not as occupancy.

    python tools/ladder_boundary.py --scale 100K --seed 20250
    python tools/ladder_boundary.py --parquet output/x.parquet --compare

A ladder can fail at either end. If the profit-maximising bid on a row lies above
the top rung, the bidder is barred from reaching it and the argmax is clipped;
the same at the bottom. Both are one-directional biases, unlike spacing, which is
noise that mostly cancels because every policy shares the grid.

THE MEASURE IS PROFIT ON THE EXTREME RUNG, NOT ROWS ON IT, and the difference
decides the answer. Only about 0.04 percent of ROWS land on the old top rung,
which reads as negligible. Those rows carry over one percent of the profit,
because whale traffic is rare and large. A row-count criterion passes a ladder
that is losing real money, which is how the defect survived v1 and v2.

Scored on the ORACLE policy alone: true value, true win rule, no model error. The
question is whether the GRID can express the right bid, so a learned curve would
only add noise to the answer.

PASS: oracle profit on the top rung, and on the bottom rung, each below 0.1
percent of total oracle profit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t9v2.core.config import load                                    # noqa: E402
from t9v2.train import bidders as B                                  # noqa: E402
from t9v2.train import metrics as M                                  # noqa: E402

THRESHOLD = 0.001          # 0.1 percent of total oracle profit, per end


def oracle_on(prices, m_win, ev):
    """Best rung per row under the true win rule, and the profit it earns.

    Profit is in dollars: every price is a CPM and a won row is one impression,
    so the sum divides by PER_MILLE exactly as `metrics.economics` does.
    """
    profit = (ev[:, None] - prices[None, :]) * (prices[None, :] >= m_win[:, None])
    j = profit.argmax(axis=1)
    best = np.maximum(profit[np.arange(len(j)), j], 0.0)
    return j, best / M.PER_MILLE


def measure(master, prices):
    m_win = np.maximum(master["lu7_competing_bid"].to_numpy(dtype=float),
                       master["floor_price"].to_numpy(dtype=float))
    ev = B.to_price_unit(master["ev_truth"].to_numpy(dtype=float))
    j, best = oracle_on(prices, m_win, ev)
    bid = best > 0
    total = float(best.sum())
    top = float(best[(j == len(prices) - 1) & bid].sum())
    bot = float(best[(j == 0) & bid].sum())
    return {
        "low": float(prices[0]), "high": float(prices[-1]), "n": int(len(prices)),
        "step_pct": float(100 * (prices[1] / prices[0] - 1)),
        "oracle_profit": total,
        "profit_on_top_rung": top, "profit_on_bottom_rung": bot,
        "share_on_top_rung": top / total if total else float("nan"),
        "share_on_bottom_rung": bot / total if total else float("nan"),
        "rows_on_top_rung": float(((j == len(prices) - 1) & bid).mean()),
        "rows_on_bottom_rung": float(((j == 0) & bid).mean()),
        "bids_on": float(bid.mean()),
    }


def verdict(r):
    ok = (r["share_on_top_rung"] < THRESHOLD
          and r["share_on_bottom_rung"] < THRESHOLD)
    return "PASS" if ok else "FAIL"


def line(tag, r):
    return ("%-22s %6.3f-%-8.1f n=%-3d step %4.1f%%  top %7.3f%%  bottom %7.3f%%  %s"
            % (tag, r["low"], r["high"], r["n"], r["step_pct"],
               100 * r["share_on_top_rung"], 100 * r["share_on_bottom_rung"],
               verdict(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet")
    ap.add_argument("--scale", default="100K")
    ap.add_argument("--seed", type=int, default=20250)
    ap.add_argument("--compare", action="store_true",
                    help="also score the shipped v2 ladder and some alternatives")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s = load("default")
    p = a.parquet or ("output/t9v2_%s_seed%d.parquet" % (a.scale, a.seed))
    master = pd.read_parquet(Path(p), columns=["lu7_competing_bid", "floor_price",
                                               "ev_truth"])
    live = measure(master, B.ladder(s))
    print("profit at risk on each boundary, oracle policy, %s\n" % p)
    print(line("settings", live))

    others = {}
    if a.compare:
        for tag, pr in [("v2 0.1-120 x48", np.geomspace(0.1, 120, 48)),
                        ("0.1-1200 x60", np.geomspace(0.1, 1200, 60)),
                        ("0.3-1200 x60", np.geomspace(0.3, 1200, 60)),
                        ("0.1-1200 x96", np.geomspace(0.1, 1200, 96))]:
            others[tag] = measure(master, pr)
            print(line(tag, others[tag]))

    print("\nPASS needs BOTH ends under %.1f%% of total oracle profit."
          % (100 * THRESHOLD))
    print("Note rows vs money: the settings ladder puts %.3f%% of ROWS on its top "
          "rung and %.3f%% of PROFIT." % (100 * live["rows_on_top_rung"],
                                          100 * live["share_on_top_rung"]))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"source": p, "threshold": THRESHOLD, "settings": live,
             "verdict": verdict(live), "compared": others},
            indent=1, default=float), encoding="utf-8")
        print("wrote %s" % a.out)
    return 0 if verdict(live) == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
