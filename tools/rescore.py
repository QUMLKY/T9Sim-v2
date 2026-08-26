# -*- coding: utf-8 -*-
"""Re-score finished seeds at the current ROAS target, WITHOUT refitting.

    python tools/rescore.py --scale 10M                 # every seed
    python tools/rescore.py --scale 10M --seeds 20250   # one
    python tools/rescore.py --scale 100K --check        # prove it, write nothing

WHY THIS IS SOUND. The ROAS gate is applied AFTER training and AFTER the argmax,
at `train/runner.py:156`. It changes which rows the bidder places and never the
bid on a placed row, and it cannot reach any fitted model. So a target change
needs new ECONOMICS and nothing else, and a full re-run spends about 35 minutes a
seed at 10M refitting models that are already frozen on disk.

WHY IT NEEDS NO REFIT. Every view's fitted pieces are in `bundle/<view>/` and
`bundle.load_bundle` returns them. The classifier's 60-rung win curve is not
persisted, but it is RECOMPUTED from the loaded Tier-2 model rather than refitted,
which is the same thing `evalfile.score_view` already does. The price head's curve
comes from its own prediction and its frozen sigma.

WHAT IT REWRITES, AND NOTHING ELSE. Per view: `economics`, `economics_price`,
`roas_target`, and the four `*_at_recommended` calibration figures under
`heads.win` and `heads.win_price`, which are scored on placed rows only and so
genuinely move. Every other key in results.json is left byte-identical, because
the file is loaded, those fields are assigned, and it is written back. Invariance
is a property of the code path rather than a hope.

THE CHECK IS THE POINT. `--check` re-scores at target 1.0 and compares against
`docs/baseline_roas1/`, which holds the thirty results.json as they were before
any of this. Wins and profit must match exactly, for both bidders, in all four
views. If they do not, the tool writes nothing and says so.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from t9v2 import campaign as C                                  # noqa: E402
from t9v2 import censor as CEN                                  # noqa: E402
from t9v2.core.config import load                               # noqa: E402
from t9v2.train import bidders as B                             # noqa: E402
from t9v2 import bundle as BU                                    # noqa: E402
from t9v2.train import encoder as E                             # noqa: E402
from t9v2.train import features as F                            # noqa: E402
from t9v2.train import metrics as M                             # noqa: E402
from t9v2.train import price as PR                              # noqa: E402
from t9v2.train.runner import calibration_or_nan                # noqa: E402

ROOT = HERE.parent
VIEWS = ["C1", "C2", "C3", "C4"]

# The pre-change snapshot, taken before the target was touched and deliberately
# left UNTRACKED so it cannot reach the published tree. It is the rollback for
# the only files this tool writes, and the reference `--check` measures against.
BASELINE = ROOT.parent / ".roas3_checkpoint" / "results_roas1"


def rescore_view(master, bundle_path, target, settings):
    """Everything a target change moves, for one view. Mirrors `runner.train_view`."""
    s = settings
    pieces = BU.load_bundle(bundle_path)
    view = pieces["manifest"]["meta"].get("view")

    d = F.prepare(CEN.censor(master, view, s), s)
    E.apply(pieces["encoders"], d)
    te = F.split(d, s)["test"]
    d_te = d[te]
    m_te = master[te]

    t1, t2, ph = pieces["tier1"], pieces["tier2"], pieces.get("price")
    p1 = t1.predict(d_te)
    prices = B.ladder(s)
    bins = s.raw["calibration"]["bins"]["value"]

    lu7 = m_te["lu7_competing_bid"].to_numpy(dtype=float)
    floor = m_te["floor_price"].to_numpy(dtype=float)
    hurdle = np.maximum(lu7, floor)
    y_win = d_te["won"].to_numpy(dtype=float)
    ev_price = B.to_price_unit(p1["ev"])
    truth_curve = B.true_win_curve(lu7, floor, prices)

    out = {}

    # --- the classifier's bidder
    curve = t2.win_curve(d_te, prices)
    out["economics"] = B.run_policies(m_te, p1["ev"], curve, prices, target,
                                      truth_curve=truth_curve)
    _, _, rung, placed = B.choose(ev_price, curve, prices, target)
    p_reco = curve[np.arange(len(rung)), rung]
    bid = prices[rung]
    y_reco = (bid >= hurdle).astype(float)[placed]
    ece_r, mce_r = calibration_or_nan(y_reco, p_reco[placed], bins)
    out["win_at_recommended"] = {"ece_at_recommended": ece_r,
                                 "mce_at_recommended": mce_r,
                                 "n_at_recommended": int(len(y_reco))}
    del curve, p_reco
    gc.collect()                       # the classifier is finished with it

    # --- the price head's bidder, the reported one
    if ph is not None:
        p_price = ph.predict(d_te)
        curve_p = PR.win_curve(p_price, prices, ph.sigma, floor)
        out["economics_price"] = B.run_policies(m_te, p1["ev"], curve_p, prices,
                                                target, truth_curve=truth_curve)
        bid_pr, _, rung_pr, placed_pr = B.choose(ev_price, curve_p, prices, target)
        p_reco_pr = curve_p[np.arange(len(rung_pr)), rung_pr]
        y_reco_pr = (bid_pr >= hurdle).astype(float)[placed_pr]
        ece_pr, mce_pr = calibration_or_nan(y_reco_pr, p_reco_pr[placed_pr], bins)
        out["win_price_at_recommended"] = {"ece_at_recommended": ece_pr,
                                           "mce_at_recommended": mce_pr,
                                           "n_at_recommended": int(len(y_reco_pr))}
        del curve_p, p_reco_pr
    del truth_curve, d, d_te, m_te
    gc.collect()
    return out


def apply_to(res, view, new, target):
    """Assign only what moved. Everything else in the file is untouched."""
    v = res["views"][view]
    v["economics"] = new["economics"]
    v["roas_target"] = target
    v["heads"]["win"].update(new["win_at_recommended"])
    if "economics_price" in new:
        v["economics_price"] = new["economics_price"]
        if "win_price" in v["heads"]:
            v["heads"]["win_price"].update(new["win_price_at_recommended"])


def one_seed(scale, seed, target, settings, check):
    d = C.seed_dir(scale, seed)
    rp = d / "results.json"
    if not rp.exists():
        return "no results.json"
    master = pd.read_parquet(C.master_path(scale, seed))
    res = json.loads(rp.read_text(encoding="utf-8"))

    fresh = {}
    for v in VIEWS:
        fresh[v] = rescore_view(master, d / "bundle" / v, target, settings)
    del master
    gc.collect()

    if check:
        base = json.loads((BASELINE / scale / ("seed%d.json" % seed))
                          .read_text(encoding="utf-8"))
        bad = []
        for v in VIEWS:
            for blk in ("economics", "economics_price"):
                if blk not in fresh[v]:
                    continue
                for pol in ("learned", "truth_ev", "oracle"):
                    a = base["views"][v][blk][pol]
                    b = fresh[v][blk][pol]
                    for k in ("wins", "profit"):
                        if abs(float(a[k]) - float(b[k])) > 1e-6:
                            bad.append("%s/%s/%s/%s %.6f != %.6f"
                                       % (v, blk, pol, k, b[k], a[k]))
        return bad or None

    for v in VIEWS:
        apply_to(res, v, fresh[v], target)
    rp.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="10M", choices=C.SCALES)
    ap.add_argument("--seeds", default=None, help="comma separated, default all ten")
    ap.add_argument("--check", action="store_true",
                    help="re-score at 1.0 and compare to docs/baseline_roas1; write nothing")
    a = ap.parse_args(argv)

    s = load("default")
    target = 1.0 if a.check else B.roas_target(s)
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else C.SEEDS)

    print("%s %s at ROAS %g, %d seeds"
          % ("CHECKING" if a.check else "re-scoring", a.scale, target, len(seeds)))
    t0, failed = time.time(), []
    for seed in seeds:
        t1 = time.time()
        bad = one_seed(a.scale, seed, target, s, a.check)
        if bad:
            failed.append((seed, bad))
            print("  seed %d  %.0fs  MISMATCH: %s"
                  % (seed, time.time() - t1,
                     bad if isinstance(bad, str) else "; ".join(bad[:3])))
        else:
            print("  seed %d  %.0fs  %s" % (seed, time.time() - t1,
                                            "reproduces baseline" if a.check else "written"))
    print("%s in %.1f min" % ("FAILED" if failed else "all clean",
                              (time.time() - t0) / 60))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
