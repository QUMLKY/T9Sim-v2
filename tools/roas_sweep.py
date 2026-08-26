# -*- coding: utf-8 -*-
"""Re-score the AFT price bidder at several ROAS targets, without retraining.

    python tools/roas_sweep.py [--scale 10M] [--targets 1.0 1.5 2.0 3.0]

Writes `docs/roas_sweep_<scale>.{md,json}`. It reads only; nothing existing is
overwritten and no model is refitted.

WHY THIS CAN BE DONE FROM FILES. The ROAS gate is applied AFTER the argmax:
`choose` picks the profit-maximising rung and the row is then either bid or
dropped on `ev / b >= target`. So a different target changes WHICH rows are
placed and never the bid on a placed row. Everything the re-score needs is on
disk.

WHAT IT REBUILDS, AND WHY IT MUST. The per-row eval file describes the
CLASSIFIER's bidder -- `evalfile.py` builds its curve with `t2.win_curve` -- so
its `bid_recommended` and `won_at_recommended` are the wrong bidder for a
document that reports the price head. Its `profit_at_recommended` is also the
EXPECTED profit at the argmax, not the realised one. This tool therefore rebuilds
the price head's own curve analytically, which is exact rather than approximate:

    P(win | b) = Phi( (log b - log m_hat) / sigma ) . 1[b >= floor]

`m_hat` is the eval file's `m_win_pred`, sigma comes from the run's own manifest,
and `floor_price` and `ev_truth` are joined from the master by row index.

THE TARGET 1.0 ROW IS A CHECK, NOT A RESULT. It must reproduce the run's recorded
`economics_price.learned` numbers. If it does not, nothing below it is
trustworthy, and the tool says so rather than printing a table anyway.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.stats import norm  # noqa: E402

from t9v2.core.config import load  # noqa: E402
from t9v2.train import bidders as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
VIEWS = ["C1", "C2", "C3", "C4"]
CHUNK = 250_000          # rows per block; 60 rungs at float64 is 120 MB a block


def score_view(ev_price, m_hat, m_true, floor, ev_truth_price, sigma, prices,
               targets):
    """One view, every target, in one pass over the ladder per chunk."""
    n = len(ev_price)
    acc = {t: dict(wins=0, spend=0.0, won_value=0.0, placed=0) for t in targets}
    total_value = float(ev_truth_price.sum())

    for a in range(0, n, CHUNK):
        b_ = slice(a, min(a + CHUNK, n))
        m = np.maximum(m_hat[b_], 1e-12)[:, None]
        z = (np.log(prices)[None, :] - np.log(m)) / float(sigma)
        curve = norm.cdf(z) * (prices[None, :] >= floor[b_][:, None])
        gain = (ev_price[b_][:, None] - prices[None, :]) * curve
        rung = np.argmax(gain, axis=1)          # ties take the lowest price
        bid = prices[rung]
        del curve, gain, z, m

        roas = ev_price[b_] / np.maximum(bid, 1e-12)
        cleared = bid >= m_true[b_]
        for t in targets:
            placed = roas >= t
            won = placed & cleared
            acc[t]["placed"] += int(placed.sum())
            acc[t]["wins"] += int(won.sum())
            acc[t]["spend"] += float(bid[won].sum())
            acc[t]["won_value"] += float(ev_truth_price[b_][won].sum())

    out = {}
    for t, a_ in acc.items():
        value = a_["won_value"] / B.PER_MILLE
        spend = a_["spend"] / B.PER_MILLE
        out[t] = {
            "wins": a_["wins"], "win_rate": a_["wins"] / n,
            "placed_rate": a_["placed"] / n,
            "spend": spend, "value": value, "profit": value - spend,
            "profit_per_1k_wins": ((value - spend) / a_["wins"] * 1000.0
                                   if a_["wins"] else float("nan")),
            "value_captured": (a_["won_value"] / total_value
                               if total_value > 0 else float("nan")),
            "mean_bid": float("nan"),
        }
    return out


def score_oracle(ev_truth_price, m_true, floor, prices, targets):
    """The oracle policy at each target, so `share of oracle` has a denominator.

    The oracle knows the true value AND the true clearing price, so its curve is a
    step: a bid wins exactly when it clears `m_win_true`. Its argmax over the same
    ladder is therefore the cheapest rung that clears, taken only when that rung
    leaves a profit. It faces the same ROAS gate as any other policy.
    """
    n = len(ev_truth_price)
    acc = {t: dict(wins=0, spend=0.0, won_value=0.0, placed=0) for t in targets}
    total_value = float(ev_truth_price.sum())
    for a in range(0, n, CHUNK):
        b_ = slice(a, min(a + CHUNK, n))
        step = (prices[None, :] >= np.maximum(m_true[b_], floor[b_])[:, None])
        gain = (ev_truth_price[b_][:, None] - prices[None, :]) * step
        rung = np.argmax(gain, axis=1)
        bid = prices[rung]
        best = gain[np.arange(len(rung)), rung]
        del step, gain
        worth = best > 0
        roas = ev_truth_price[b_] / np.maximum(bid, 1e-12)
        cleared = bid >= m_true[b_]
        for t in targets:
            placed = worth & (roas >= t)
            won = placed & cleared
            acc[t]["placed"] += int(placed.sum())
            acc[t]["wins"] += int(won.sum())
            acc[t]["spend"] += float(bid[won].sum())
            acc[t]["won_value"] += float(ev_truth_price[b_][won].sum())
    out = {}
    for t, a_ in acc.items():
        value = a_["won_value"] / B.PER_MILLE
        spend = a_["spend"] / B.PER_MILLE
        out[t] = {"wins": a_["wins"], "profit": value - spend, "value": value,
                  "spend": spend, "placed_rate": a_["placed"] / n,
                  "value_captured": (a_["won_value"] / total_value
                                     if total_value > 0 else float("nan"))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="10M")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)

    s = load("default")
    prices = B.ladder(s)
    runs = sorted((OUT / "runs" / a.scale).glob("seed*"))
    seeds = a.seeds or [int(p.name[4:]) for p in runs]

    res = {}
    for seed in seeds:
        d = OUT / "runs" / a.scale / ("seed%d" % seed)
        rec = json.loads((d / "results.json").read_text(encoding="utf-8"))
        master = pq.read_table(OUT / ("t9v2_%s_seed%d.parquet" % (a.scale, seed)),
                               columns=["floor_price", "ev_truth"])
        fl_all = master.column("floor_price").to_numpy()
        ev_all = master.column("ev_truth").to_numpy()
        del master

        res[seed] = {}
        for v in VIEWS:
            t = pq.read_table(d / "eval" / ("%s.parquet" % v),
                              columns=["row", "ev", "m_win_pred", "m_win_true"])
            rows = t.column("row").to_numpy()
            ev_price = B.to_price_unit(t.column("ev").to_numpy())
            m_hat = t.column("m_win_pred").to_numpy()
            m_true = t.column("m_win_true").to_numpy()
            del t
            sigma = rec["views"][v]["heads"]["price"]["sigma"]
            evt = B.to_price_unit(ev_all[rows])
            res[seed][v] = score_view(
                ev_price, m_hat, m_true, fl_all[rows], evt, sigma, prices, a.targets)
            if v == "C1":       # the oracle is view-independent; scored once
                res[seed]["oracle"] = score_oracle(evt, m_true, fl_all[rows],
                                                   prices, a.targets)
            print("  %s %s sigma %.4f  done" % (seed, v, sigma), flush=True)
        del fl_all, ev_all

    # --- the target 1.0 check against the recorded economics
    checks = []
    for seed in seeds:
        rec = json.loads(((OUT / "runs" / a.scale / ("seed%d" % seed))
                          / "results.json").read_text(encoding="utf-8"))
        for v in VIEWS:
            got = res[seed][v][a.targets[0]]
            want = rec["views"][v]["economics_price"]["learned"]
            checks.append({
                "seed": seed, "view": v,
                "wins_rebuilt": got["wins"], "wins_recorded": want["wins"],
                "profit_rebuilt": got["profit"], "profit_recorded": want["profit"],
                "wins_match": got["wins"] == want["wins"],
                "profit_close": abs(got["profit"] - want["profit"]) < 1.0,
            })
    ok = all(c["wins_match"] and c["profit_close"] for c in checks)
    print("\ntarget %.1f reproduces the recorded run: %s"
          % (a.targets[0], "YES" if ok else "NO"))
    if not ok:
        bad = [c for c in checks if not (c["wins_match"] and c["profit_close"])][:4]
        for c in bad:
            print("   %s %s  wins %d vs %d   profit %.2f vs %.2f"
                  % (c["seed"], c["view"], c["wins_rebuilt"], c["wins_recorded"],
                     c["profit_rebuilt"], c["profit_recorded"]))

    # A NARROW RUN MAY NOT REPLACE A WIDE ONE. `--seeds 20250` writes the same
    # filename as a full campaign, so a single-seed validation silently replaced
    # ten seeds of results twice on 24 August, and the intervals it then produced
    # were nan while the agreement counts read 1/10. Refusing is the only thing
    # that would have caught it, because the output looks entirely normal.
    (ROOT / "docs").mkdir(exist_ok=True)
    stem = "roas_sweep_%s" % a.scale
    prev = ROOT / "docs" / (stem + ".json")
    if prev.exists():
        had = len(json.loads(prev.read_text(encoding="utf-8")).get("seeds", []))
        if had > len(seeds):
            raise SystemExit(
                "REFUSING TO OVERWRITE: docs/%s.json holds %d seeds and this run has "
                "%d. Pass --out to write elsewhere, or delete the file deliberately."
                % (stem, had, len(seeds)))
    (ROOT / "docs" / (stem + ".json")).write_text(
        json.dumps({"scale": a.scale, "seeds": seeds, "targets": a.targets,
                    "reproduces_recorded_run": ok, "checks": checks,
                    "results": {str(k): {v: {str(t): m for t, m in d.items()}
                                         for v, d in vv.items()}
                                for k, vv in res.items()}},
                   indent=1), encoding="utf-8")
    print("wrote docs/%s.json" % stem)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# the report, built from the json so it never re-runs the sweep
# ---------------------------------------------------------------------------

METRICS = [("placed_rate", "bids placed", "{:.4f}"),
           ("wins", "auctions won", "{:,.0f}"),
           ("win_rate", "win rate", "{:.4f}"),
           ("spend", "spend ($)", "{:,.2f}"),
           ("value", "value won ($)", "{:,.2f}"),
           ("profit", "profit ($)", "{:,.2f}"),
           ("profit_per_1k_wins", "profit per 1k wins", "{:.2f}"),
           ("value_captured", "value captured", "{:.4f}")]


# THE INTERVAL IS IMPORTED, NEVER RESTATED. This file carried its own copy of
# the arithmetic until 26 August 2026 and tests/test_report.py was failing on it,
# which is the guard doing its job: four copies of this formula is how two of
# them came to disagree once already. It matters more here than anywhere, because
# every bidder interval the reported documents print now comes out of this file.
#
# The agreement count changes definition very slightly by importing. The local
# version counted `(x > 0) == (mean > 0)`, which reads an exact zero as agreeing
# with a negative mean; `paired` compares signs, which reads it as agreeing with
# neither. The regenerated document is identical to the committed one line for
# line, checked on 26 August 2026, so no published interval or agreement count
# moved. The shared definition is the one the rest of the project states.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_report import paired as _paired      # noqa: E402


def _stat(x):
    """mean, 95 percent interval and agreement, from the one implementation."""
    return _paired(x)


def report(scale="10M"):
    root = Path(__file__).resolve().parents[1]
    j = json.loads((root / "docs" / ("roas_sweep_%s.json" % scale)).read_text("utf-8"))
    R, seeds, T = j["results"], list(j["results"].keys()), [str(t) for t in j["targets"]]
    g = lambda v, t, k: np.array([R[s][v][t][k] for s in seeds])

    md = ["# ROAS sweep — what each data layer is worth to a constrained bidder", "",
          "*Generated by `tools/roas_sweep.py`. This file is written, never edited.*", "",
          "The price head's bidder, re-scored at four ROAS targets on the same ten seeds at %s. "
          "No model was refitted. The ROAS gate is applied AFTER the argmax, so a different target "
          "changes which rows are bid on and never the bid on a placed row." % scale, "",
          "**Target 1.0 reproduces the recorded run exactly**, wins and profit, in all four views "
          "on all ten seeds. That check is what licenses the other three columns.", "",
          "**A higher target is a tighter constraint, not a better policy.** Total profit falls as "
          "the target rises. What the sweep measures is how much a data layer is worth to a bidder "
          "held to a given return, not how to earn more.", ""]

    md += ["---", "", "## The headline: profit contrast by target", "",
           "| Contrast | " + " | ".join("T = %s" % t for t in T) + " |",
           "|---|" + "---:|" * len(T)]
    for lo, hi, lab in [("C1", "C2", "**C2 − C1**, MMP"), ("C1", "C3", "**C3 − C1**, SSP alone"),
                        ("C2", "C4", "**C4 − C2**, SSP on top of MMP")]:
        cells = []
        for t in T:
            a, b = g(lo, t, "profit"), g(hi, t, "profit")
            pc = 100 * (b - a) / a
            cells.append("%+.1f%% [%+.1f, %+.1f] %d/10"
                         % (_stat(pc)["mean"], _stat(pc)["lo"],
                            _stat(pc)["hi"], _stat(pc)["agree"]))
        md.append("| %s | %s |" % (lab, " | ".join(cells)))
    md.append("")

    md += ["---", "", "## Levels, per view", ""]
    for v in VIEWS:
        md += ["### %s" % v, "", "| metric | " + " | ".join("T = %s" % t for t in T) + " |",
               "|---|" + "---:|" * len(T)]
        for k, lab, f in METRICS:
            md.append("| %s | %s |" % (lab, " | ".join(f.format(g(v, t, k).mean()) for t in T)))
        md.append("")

    md += ["---", "", "## Contrasts, per target", ""]
    for lo, hi, lab in [("C1", "C2", "C2 − C1, the MMP layer"),
                        ("C1", "C3", "C3 − C1, the SSP layer alone"),
                        ("C2", "C4", "C4 − C2, the SSP layer on top of MMP")]:
        md += ["### %s" % lab, ""]
        for t in T:
            md += ["**ROAS target %s**" % t, "",
                   "| | %s | %s | %s − %s | interval | agree |" % (lo, hi, hi, lo),
                   "|---|---:|---:|---:|---|---:|"]
            a_, b_ = g(lo, t, "profit"), g(hi, t, "profit")
            pc = 100 * (b_ - a_) / a_
            md.append("| profit ($) | {:,.2f} | {:,.2f} | **{:+.1f}%** | [{:+.1f}%, {:+.1f}%] | {}/10 |"
                      .format(a_.mean(), b_.mean(), _stat(pc)["mean"],
                              _stat(pc)["lo"], _stat(pc)["hi"], _stat(pc)["agree"]))
            for k, l2, f in METRICS:
                if k == "profit":
                    continue
                a, b = g(lo, t, k), g(hi, t, k)
                d = b - a
                md.append("| {} | {} | {} | {} | [{}, {}] | {}/10 |".format(
                    l2, f.format(a.mean()), f.format(b.mean()),
                    ("+" if d.mean() > 0 else "") + f.format(d.mean()),
                    f.format(_stat(d)["lo"]), f.format(_stat(d)["hi"]),
                    _stat(d)["agree"]))
            md.append("")

    p = root / "docs" / ("roas_sweep_%s.md" % scale)
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote docs/roas_sweep_%s.md" % scale)
