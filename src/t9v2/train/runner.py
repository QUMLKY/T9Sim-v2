# -*- coding: utf-8 -*-
"""Train and score one view, and one seed across all four.

The order here is the whole discipline of the stage. Split first, on days, before
anything is fitted. Fit the encoders on TRAIN only. Fit the heads on train with
early stopping on valid. Score on TEST, which nothing has seen.

Every view gets the same rows and the same days. The only thing that differs is
what each is allowed to look at, which is what makes the comparison an ablation.
"""
from __future__ import annotations

import numpy as np

from .. import censor as CEN
from ..core.config import load
from . import bidders as B
from . import encoder as E
from . import features as F
from . import metrics as M
from .tier1 import HEADS, Tier1
from .tier2 import Tier2


def train_view(master, view, settings, seed=0):
    """Fit and score one view. Returns the metrics and the fitted pieces.

    The censored frame is released as soon as `prepare` has copied it. At 10M
    rows a master frame is about 5 GB, and holding the master, its censored copy
    and the prepared copy at once needs more memory than this machine has. The
    campaign runs 10 seeds at that scale, so this is not a one-off precaution.
    """
    import gc
    censored = CEN.censor(master, view, settings)
    d = F.prepare(censored, settings)
    del censored
    gc.collect()
    masks = F.split(d, settings)

    enc = E.build(d[masks["train"]].copy(), view, settings)
    enc_cols = E.apply(enc, d)

    t1_cols = F.tier1_features(d, "click", extra=enc_cols)
    t2_cols = F.tier2_features(d, extra=enc_cols)

    t1 = Tier1().fit(d, masks, t1_cols, settings, seed)
    t2 = Tier2().fit(d, masks, t2_cols, settings, seed)

    te = masks["test"]
    d_te = d[te]
    p1 = t1.predict(d_te)
    p_win = t2.predict(d_te)

    bins = settings.raw["calibration"]["bins"]["value"]
    res = {"view": view, "columns": int(len(d.columns)), "head_mode": dict(t1.mode),
           "features_tier1": len(t1_cols), "features_tier2": len(t2_cols),
           "encoders": sorted(enc), "spend_scale": t1.scale,
           "n_train": dict(t1.n_train, win=t2.n_train),
           "heads": {}}

    # --- Tier 1, scored only where this view SHOWS the label
    shown = d_te["click"].notna().to_numpy()
    for h, label, sub in [("click", "click", np.ones(len(d_te), bool)),
                          ("install", "install", d_te["click"].to_numpy() == 1),
                          ("payer", "is_payer", d_te["install"].to_numpy() == 1)]:
        m = shown & np.asarray(sub, dtype=bool)
        y = d_te.loc[m, label].to_numpy(dtype=float)
        p = np.asarray(p1[h])[m]
        ece, mce = calibration_or_nan(y, p, bins)
        res["heads"][h] = {"n": int(m.sum()), "auc": M.auc(y, p) if m.sum() else float("nan"),
                           "ece": ece, "mce": mce}

    m = shown & (d_te["is_payer"].to_numpy() == 1)
    y = d_te.loc[m, "ltv_value"].to_numpy(dtype=float)
    mu = np.asarray(p1["spend"])[m]
    res["heads"]["spend"] = {
        "n": int(m.sum()),
        "crps": M.crps_lognormal(y, mu, t1.scale or 1.0),
        "rmse_log": M.rmse(np.log(np.where(y > 0, y, np.nan)), mu),
    }

    # --- Tier 2, scored on every row: `won` is always visible
    y_win = d_te["won"].to_numpy(dtype=float)
    ece, mce = calibration_or_nan(y_win, p_win, bins)
    res["heads"]["win"] = {"n": int(len(y_win)), "auc": M.auc(y_win, p_win),
                           "ece": ece, "mce": mce}

    # --- economics, against the uncensored master on the same test rows
    prices = B.ladder(settings)
    curve = t2.win_curve(d_te, prices)
    m_te = master[te]
    res["economics"] = B.run_policies(m_te, p1["ev"], curve, prices)
    # what this view's VALUATIONS are worth before any bidder touches them,
    # both per impression, which is v1's `ev_ratio` diagnostic
    res["ev"] = B.ev_bias(p1["ev"], m_te["ev_truth"].to_numpy(dtype=float))
    return res, {"tier1": t1, "tier2": t2, "encoders": enc,
                 "t1_cols": t1_cols, "t2_cols": t2_cols}


def calibration_or_nan(y, p, bins):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return M.calibration(y, p, bins)


def run_seed(master, settings=None, seed=0, views=None, quiet=False):
    """All 4 views on one master. The comparison the whole study is."""
    import gc
    s = settings or load("default")
    out = {}
    for v in (views or CEN.VIEWS):
        r, fitted = train_view(master, v, s, seed)
        out[v] = r
        del fitted                    # the models and their frames, per view
        gc.collect()
        if not quiet:
            h = r["heads"]
            e = r["economics"]["learned"]
            print("  %-3s %2d cols  click AUC %.4f  install %.4f  win %.4f  "
                  "profit %.1f  ev_ratio %.4f  of oracle %.4f  "
                  "ev level %.3f order %.3f"
                  % (v, r["columns"], h["click"]["auc"], h["install"]["auc"],
                     h["win"]["auc"], e["profit"], e["ev_ratio"],
                     e["value_vs_oracle"], r["ev"]["ratio"],
                     r["ev"]["spearman"]))
    return out
