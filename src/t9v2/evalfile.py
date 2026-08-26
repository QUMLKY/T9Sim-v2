# -*- coding: utf-8 -*-
"""The per-row eval file: what the frozen predictor said about each test row.

    from t9v2.evalfile import write_eval
    write_eval(master, bundle_dir, out_dir, settings)

WHY IT IS A SEPARATE PASS THAT LOADS A BUNDLE. It could have been written during
the fit, from the model still in memory, and the branch this reimplements did
exactly that. Loading instead puts the frozen artifact on the critical path: the
numbers reported come from the file that shipped, so a bundle that fails to
reload, or reloads differently, is caught here rather than never. The fit writes
the model; this reads it back and scores with it.

WHAT IT BUYS. v2 kept aggregates only, so a new metric meant refitting every
view of every seed, four hours at 10M, and the refit produced a different model.
With this file a new metric is a re-score of numbers already on disk, minutes.

THE ROW SET IS THE TEST SPLIT, not everything. Every metric in the study is
computed on test, and at 10M the full master is ten million rows against about
1.4 million in test, four times over for the four views.

THE TWO WIN PROBABILITIES ARE NOT THE SAME NUMBER, and this is the file's real
purpose. `p_win_at_logged` is the classifier at the bid the generator recorded on
the row, which is where v2 measured calibration. `p_win_at_recommended` is the
curve at the rung the BIDDER chose, `curve[i, j]` for the index the bidder handed
back. The second is where the bidder actually operates and the first is not, so
calibration measured at the logged bid never touched the model as used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import bundle as BU
from . import censor as CEN
from .core.config import load
from .train import bidders as B
from .train import encoder as E
from .train import features as F

SCHEMA = "t9v2-eval-1"


def score_view(master, bundle_path, settings=None):
    """Load one view's bundle and score the test rows with it. Returns a frame."""
    s = settings or load("default")
    pieces = BU.load_bundle(bundle_path)
    view = pieces["manifest"]["meta"].get("view")

    d = F.prepare(CEN.censor(master, view, s), s)
    E.apply(pieces["encoders"], d)                 # rebuild the encoder columns
    te = F.split(d, s)["test"]
    d_te = d[te]

    t1, t2 = pieces["tier1"], pieces["tier2"]
    p1 = t1.predict(d_te)
    p_logged = t2.predict(d_te)
    ph = pieces.get("price")
    p_price = ph.predict(d_te) if ph is not None else None

    prices = B.ladder(s)
    curve = t2.win_curve(d_te, prices)
    ev_price = B.to_price_unit(p1["ev"])
    target = B.roas_target(s)
    bid, profit, rung, placed = B.choose(ev_price, curve, prices, target)
    p_reco = curve[np.arange(len(rung)), rung]

    # the counterfactual outcome of the bid the bidder would have made, read from
    # the uncensored master: would b* have cleared the hurdle on this row
    m_te = master[te]
    hurdle = np.maximum(m_te["lu7_competing_bid"].to_numpy(dtype=float),
                        m_te["floor_price"].to_numpy(dtype=float))

    out = pd.DataFrame({
        "row": np.flatnonzero(np.asarray(te)),
        "p_click": p1["click"], "p_install": p1["install"],
        "p_payer": p1["payer"], "spend_hat": p1["spend"],
        "ev": p1["ev"],
        "bid_logged": d_te["bid_price"].to_numpy(dtype=float),
        "p_win_at_logged": p_logged,
        "bid_recommended": bid,
        "rung": rung.astype(np.int16),
        "p_win_at_recommended": p_reco,
        "profit_at_recommended": profit,
        # the ROAS gate's answer, kept beside the bid rather than folded into it.
        # `bid_recommended` stays the ungated argmax on every row, so the file
        # records both what the bidder would pay and whether it was willing to.
        "placed": placed.astype(np.int8),
        "won_logged": d_te["won"].to_numpy(dtype=np.int8),
        "won_at_recommended": (placed & (bid >= hurdle)).astype(np.int8),
        # the TRUE minimum winning price, from the uncensored master. Written
        # beside the prediction so the price head can be re-scored from this
        # file in every view, including the two that could never observe it.
        "m_win_true": hurdle,
    })
    if p_price is not None:
        out["m_win_pred"] = p_price
    return out, {"view": view, "rows": int(len(out)), "rungs": int(len(prices)),
                 "roas_target": target,
                 "placed_rate": float(placed.mean()),
                 "price_sigma": (float(ph.sigma) if ph is not None else None)}


def write_eval(master, bundle_dir, out_dir, settings=None, views=None, quiet=False):
    """Score every view's bundle and write `<out_dir>/<view>.parquet` plus a manifest."""
    s = settings or load("default")
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_view = {}
    for v in (views or CEN.VIEWS):
        root = bundle_dir / v
        if not (root / "manifest.json").exists():
            raise FileNotFoundError("no bundle for %s at %s" % (v, root))
        frame, info = score_view(master, root, s)
        frame.to_parquet(out_dir / ("%s.parquet" % v), index=False,
                         compression="snappy")
        per_view[v] = info
        if not quiet:
            print("  %-3s %7d rows  mean bid %8.3f  p(win) logged %.4f -> "
                  "recommended %.4f"
                  % (v, len(frame), frame["bid_recommended"].mean(),
                     frame["p_win_at_logged"].mean(),
                     frame["p_win_at_recommended"].mean()))

    (out_dir / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "views": per_view,
        "columns": list(frame.columns),
    }, indent=1, sort_keys=True), encoding="utf-8")
    return out_dir
