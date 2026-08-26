# -*- coding: utf-8 -*-
"""Score the Tier-1 heads on EVERY test row, not only the rows their view can see.

    python tools/tier1_allrows.py                 # 10M, all ten seeds
    python tools/tier1_allrows.py --scale 1M
    python tools/tier1_allrows.py --seeds 20250 20251

Writes `docs/gates/tier1_allrows.{json,md}`. `results_report.py` reads the JSON
and adds three rows to the results table; it never recomputes this itself,
because the work is a disk read of ten 10M masters and a report regeneration has
to stay cheap.

WHAT IS WRONG WITH THE FIGURE THIS REPLACES. `runner.py` scores every Tier-1 head
under one mask, `shown = d_te["click"].notna()`. In C2 and C4 that is every test
row. In C1 and C3 the funnel is masked on lost rows, so it is the rows the DSP
won -- about 27 percent of them for click, 19 for install and 16 for payer, and
C1's own count swings from 194,404 to 508,804 across the ten 10M seeds depending
on how much that seed's bidder happened to win. So `C2 click AUC 0.706` and
`C1 click AUC 0.641` are not computed on the same sample, and the gap between
them mixes the MMP layer with the difference in exam.

TRAINING ON WON ROWS IS THE STUDY. SCORING ON THEM IS NOT. One mask, two stages.
C1 and C3 must train on won rows only, because a DSP without MMP integration is
never told what happened on an auction it lost -- that deprivation is the thing
the ablation exists to measure. But the model then predicts every row, and the
question the dissertation asks of it is how well it ranks the market. Throwing
away the predictions it made on 73 percent of the market answers a different
question.

NOTHING IS RE-CENSORED AND NOTHING IS RETRAINED, and both matter enough to say
plainly. Censoring is a MASK over one intact master, not a deletion: on 100K seed
20250 the C1 view has `click` NaN on all 69,942 lost rows and the master has it
NaN on none of them. And every view already predicted every test row -- C1's
`eval/C1.parquet` carries 1,428,601 rows with zero NaN among the 1,038,636 lost
ones. This tool reads predictions that are already frozen and labels that were
never removed.

THE STUDY ALREADY SCORES THIS WAY EVERYWHERE ELSE, which is what makes the Tier-1
mask the inconsistency rather than the fix. `runner.py` computes profit,
`value_captured`, `wins` and the oracle ceiling "against the uncensored master on
the same test rows" -- its own comment -- for all four views, using `ev_truth` and
`lu7_competing_bid`, which no view can see. If reading truth the model cannot
observe were illegitimate, the headline economics would go with it.

BOTH POPULATIONS ARE REPORTED, never one instead of the other. They answer
different questions and both are worth having:

    own rows   what this DSP could measure about itself. A real C1 buyer cannot
               compute the all-rows figure, because it does not hold the labels.
    all rows   how good the model actually is. Available only because the data is
               simulated and the evaluator holds the truth.

WHAT IT DOES TO THE HEADLINE. The MMP click contrast is +0.0642 as reported and
+0.0220 scored like for like, both 10/10 at 10M, and the whole of the difference
is C1 RISING rather than C2 falling: C1 reads 0.6414 on its own rows and 0.6835
on all of them. Won rows are the slice the bidder bid up, which correlates with
what it predicted, so the predictor has roughly half the spread to discriminate
across -- sd 0.0145 against 0.0262 on seed 20250. Range restriction, not a worse
model. Install moves +0.0458 to +0.0377 and payer barely moves at all, +0.0953 to
+0.0948, because C1's payer head is near-uninformative on either population and
there is little discrimination for the restriction to attenuate.

NO ECONOMIC NUMBER IS TOUCHED by anything here.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from t9v2.core.config import load as load_settings      # noqa: E402
from t9v2.train import features as F                    # noqa: E402
from t9v2.train import metrics as M                      # noqa: E402

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

ROOT = HERE.parent
OUTPUT = ROOT / "output"
RUNS = OUTPUT / "runs"
GATES = ROOT / "docs" / "gates"
VIEWS = ["C1", "C2", "C3", "C4"]

# The views whose funnel is masked on lost rows. Listed rather than derived from
# the frame so that a view gaining or losing MMP is a decision made in the
# censoring map and read here, not inferred from whatever columns turn up.
MASKED = ("C1", "C3")

# head -> (master label column, eval prediction column, parent condition column)
HEADS = [("click",   "click",    "p_click",   None),
         ("install", "install",  "p_install", "click"),
         ("payer",   "is_payer", "p_payer",   "install")]

# head -> the generator's OWN probability for it, which is the oracle predictor
#
# THE ORACLE FOR A FUNNEL HEAD IS THE BAYES CEILING, not 1.0. `p_click` is the
# probability the generator drew the click from, so scoring it against the
# realised outcome measures how well the best possible predictor could rank this
# market. What is left below 1.0 is the coin flip itself and no model can have
# it. Without the row, a reader comparing C2's 0.706 to a blank has no idea
# whether 0.706 is most of what is available or a third of it. It is most of it.
ORACLE_COL = {"click": "p_click", "install": "p_install", "payer": "p_payer"}

MASTER_COLS = (["click", "install", "is_payer", "won", "timestamp"]
               + sorted(ORACLE_COL.values()))


def seeds_for(scale):
    """The seeds that have both a master and a full set of eval files."""
    out = []
    for d in sorted((RUNS / scale).glob("seed*")):
        seed = int(d.name[4:])
        if not (OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))).exists():
            continue
        if all((d / "eval" / ("%s.parquet" % v)).exists() for v in VIEWS):
            out.append(seed)
    return out


def one_seed(scale, seed, s):
    """Both populations for one seed, all four views.

    The master is read at five columns and dropped before the next seed, because
    ten 10M masters held at once do not fit on this machine.
    """
    m = pd.read_parquet(OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed)),
                        columns=MASTER_COLS)
    m = m[F.split(m, s)["test"]].reset_index(drop=True)
    out = {}
    for view in VIEWS:
        ev = pd.read_parquet(RUNS / scale / ("seed%d" % seed) / "eval" /
                             ("%s.parquet" % view),
                             columns=[p for _, _, p, _ in HEADS])
        if len(ev) != len(m):
            raise RuntimeError(
                "%s seed %d view %s: %d eval rows against %d test rows. The eval "
                "file is not the frozen per-row output for this master."
                % (scale, seed, view, len(ev), len(m)))
        shown = ((m["won"].to_numpy() == 1) if view in MASKED
                 else np.ones(len(m), bool))
        out[view] = {}
        for name, label, pred, parent in HEADS:
            sub = (np.ones(len(m), bool) if parent is None
                   else (m[parent].to_numpy() == 1))
            y = m[label].to_numpy(dtype=float)
            p = ev[pred].to_numpy(dtype=float)
            own, allr = shown & sub, sub
            out[view][name] = {
                "n_own": int(own.sum()), "auc_own": M.auc(y[own], p[own]),
                "n_all": int(allr.sum()), "auc_all": M.auc(y[allr], p[allr])}
        # the spend head's two populations, as counts only. Its metric is CRPS
        # and recomputing that needs the fitted lognormal scale, which is not in
        # the eval file. The counts alone are worth reporting: C1 scores its
        # E(spend | payer) CRPS on about 92 payers at 10M against C2's 660.
        sub = m["is_payer"].to_numpy() == 1
        out[view]["spend"] = {"n_own": int((shown & sub).sum()),
                              "n_all": int(sub.sum())}

    # THE ORACLE, once per seed rather than once per view. It does not depend on
    # a view at all: it is the generator's own probability against the realised
    # outcome, on the same test rows and the same parent condition the views are
    # scored on, so the ceiling is comparable with the figures beneath it.
    out["oracle"] = {}
    for name, label, _pred, parent in HEADS:
        sub = (np.ones(len(m), bool) if parent is None
               else (m[parent].to_numpy() == 1))
        y = m[label].to_numpy(dtype=float)[sub]
        p = m[ORACLE_COL[name]].to_numpy(dtype=float)[sub]
        out["oracle"][name] = {"auc": M.auc(y, p), "n": int(sub.sum())}

    del m
    return out


def paired(vals):
    """`results_report.paired`, imported so there is one interval in the project."""
    sys.path.insert(0, str(HERE))
    from results_report import paired as _p
    return _p(vals)


def contrast(data, hi, lo, head, key):
    return paired([data[s][hi][head][key] - data[s][lo][head][key]
                   for s in sorted(data)])


def markdown(scale, data):
    seeds = sorted(data)
    L = ["# Tier-1 heads on both populations, %s" % scale, "",
         "*Generated by `tools/tier1_allrows.py`. This file is written, never "
         "edited.*", "",
         "%d seeds. **own rows** is what each view can see, which is the figure "
         "`results.json` records and the one a real DSP could compute about "
         "itself. **all rows** scores the same frozen predictions against the "
         "uncensored master on every test row. C2 and C4 see every row, so their "
         "two columns are identical by construction." % len(seeds), "",
         "Nothing is re-censored and nothing is retrained. The predictions are "
         "the frozen per-row output already on disk and the labels were never "
         "removed from the master.", "",
         "## Levels, mean over %d seeds" % len(seeds), "",
         "| Head | View | own rows | n | all rows | n |",
         "|---|---|---:|---:|---:|---:|"]
    for name, _, _, _ in HEADS:
        for v in VIEWS:
            L.append("| %s | %s | %.4f | %.0f | %.4f | %.0f |"
                     % (name, v,
                        np.mean([data[s][v][name]["auc_own"] for s in seeds]),
                        np.mean([data[s][v][name]["n_own"] for s in seeds]),
                        np.mean([data[s][v][name]["auc_all"] for s in seeds]),
                        np.mean([data[s][v][name]["n_all"] for s in seeds])))
    L += ["", "## The oracle, and what is left below it", "",
          "The generator OWN probability scored against the realised outcome, on "
          "the same rows. This is the Bayes ceiling: what is left below 1.0 is the "
          "coin flip itself and no model can have it.", "",
          "| Head | oracle | best view | gap left |", "|---|---:|---:|---:|"]
    for name, _, _, _ in HEADS:
        orc = np.mean([data[s]["oracle"][name]["auc"] for s in seeds])
        best = max(np.mean([data[s][v][name]["auc_all"] for s in seeds])
                   for v in VIEWS)
        L.append("| %s | %.4f | %.4f | %.4f |" % (name, orc, best, orc - best))
    L += ["", "## Contrasts, paired within seed", "",
          "| Contrast | Head | as reported (own rows) | like for like (all rows) |",
          "|---|---|---|---|"]
    for hi, lo in [("C2", "C1"), ("C3", "C1"), ("C4", "C2")]:
        for name, _, _, _ in HEADS:
            a = contrast(data, hi, lo, name, "auc_own")
            b = contrast(data, hi, lo, name, "auc_all")
            L.append("| %s-%s | %s | %+.4f [%+.4f, %+.4f] %d/%d | "
                     "%+.4f [%+.4f, %+.4f] %d/%d |"
                     % (hi, lo, name, a["mean"], a["lo"], a["hi"], a["agree"],
                        a["n"], b["mean"], b["lo"], b["hi"], b["agree"], b["n"]))
    L += ["", "## The spend head's two populations, counts only", "",
          "Its metric is CRPS and recomputing that needs the fitted lognormal "
          "scale, which the eval file does not carry. The counts are reported "
          "because they are the more alarming number.", "",
          "| View | own rows | all rows |", "|---|---:|---:|"]
    for v in VIEWS:
        L.append("| %s | %.0f | %.0f |"
                 % (v, np.mean([data[s][v]["spend"]["n_own"] for s in seeds]),
                    np.mean([data[s][v]["spend"]["n_all"] for s in seeds])))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="10M")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)

    s = load_settings("default")
    seeds = a.seeds if a.seeds is not None else seeds_for(a.scale)
    if not seeds:
        sys.exit("no complete seeds found for %s" % a.scale)

    data = {}
    for seed in seeds:
        data[seed] = one_seed(a.scale, seed, s)
        print("  seed %d" % seed, flush=True)

    GATES.mkdir(parents=True, exist_ok=True)
    jp = GATES / "tier1_allrows.json"
    payload = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
    payload[a.scale] = {str(k): v for k, v in data.items()}
    jp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    mp = GATES / "tier1_allrows.md"
    mp.write_text("\n".join(markdown(a.scale, data)) + "\n", encoding="utf-8")

    print("wrote %s and %s, %d seeds at %s"
          % (jp.relative_to(ROOT), mp.relative_to(ROOT), len(seeds), a.scale))
    for name, _, _, _ in HEADS:
        r = contrast(data, "C2", "C1", name, "auc_own")
        l_ = contrast(data, "C2", "C1", name, "auc_all")
        print("  MMP %-8s own %+.4f %2d/%d    all %+.4f %2d/%d"
              % (name, r["mean"], r["agree"], r["n"],
                 l_["mean"], l_["agree"], l_["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
