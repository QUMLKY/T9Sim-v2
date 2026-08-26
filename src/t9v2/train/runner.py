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
from . import price as PR
from .tier1 import HEADS, Tier1
from .tier2 import Tier2


def train_view(master, view, settings, seed=0, sigma=None):
    """Fit and score one view. Returns the metrics and the fitted pieces.

    `sigma` is the price head's AFT scale and is REQUIRED for C1 and C2, where
    every label is an interval and the scale shapes the likelihood. C3 and C4
    ignore whatever is passed, because a normal AFT with exact labels has a
    location MLE independent of the scale, so they take the placeholder. There
    is no default: v1's scale was 1.0 copied from nowhere and it sat on the
    headline contrast, and a default here would let that happen again quietly.

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
    # LEAVE-ONE-OUT ON THE TRAINING ROWS, and only there. The encoders were fitted
    # on `masks["train"]`, so those rows sit inside their own cell means; valid
    # and test were never in the fit and take the plain estimate. Passing the mask
    # is what stops `_enc_ssp_minwin_price` handing a training row a shrunk copy
    # of its own label, `won` being an identity in `min_winning_price`.
    enc_cols = E.apply(enc, d, train_mask=masks["train"])

    t1_cols = F.tier1_features(d, "click", extra=enc_cols)
    t2_cols = F.tier2_features(d, extra=enc_cols)

    t1 = Tier1().fit(d, masks, t1_cols, settings, seed)
    t2 = Tier2().fit(d, masks, t2_cols, settings, seed)

    # --- the price head. Same estimand, same features, same likelihood class in
    # all four views; only the LABEL differs, exact in C3 and C4 and interval in
    # C1 and C2. That is the whole design of step 6.
    pr_cols = PR.features(d, enc_cols)
    exact = "min_winning_price" in d.columns
    if exact:
        sig = settings.raw["xgboost"]["sigma"]["placeholder"]["value"]
    elif sigma is None:
        raise RuntimeError(
            "%s trains the price head on INTERVALS, so the AFT scale shapes its "
            "likelihood and it must be supplied. `run_seed` reads it off C3's "
            "residuals; a caller training %s alone has to pass one explicitly. "
            "Refusing to default: an unjustified scale on the censored views is "
            "the defect this replaces." % (view, view))
    else:
        sig = sigma
    ph = PR.PriceHead().fit(d, masks, pr_cols, settings, sig, seed)

    te = masks["test"]
    d_te = d[te]
    n_cols = int(len(d.columns))
    # THE PREPARED FRAME GOES HERE, and at 10M it is about 4 GB. Everything
    # below reads `d_te`, which is already a copy; the fits are done and the
    # heads hold column NAMES rather than the frame. Holding `d` through the
    # scoring section alongside the master and four win curves is what took a
    # 10M seed to 14.5 GB and made it page.
    del d
    gc.collect()
    p1 = t1.predict(d_te)
    p_win = t2.predict(d_te)

    bins = settings.raw["calibration"]["bins"]["value"]
    res = {"view": view, "columns": n_cols, "head_mode": dict(t1.mode),
           "features_tier1": len(t1_cols), "features_tier2": len(t2_cols),
           # THE NAMES, not only the counts. Gate 4's anti-leak condition asks
           # what each view actually FITTED ON, and a count cannot answer it: a
           # view could carry `min_winning_price` and still report 24 features.
           # Recording the lists makes the check read the fit rather than the ban
           # list that is supposed to have prevented it.
           "t1_cols": list(t1_cols), "t2_cols": list(t2_cols),
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

    # --- economics, against the uncensored master on the same test rows
    prices = B.ladder(settings)
    curve = t2.win_curve(d_te, prices)
    m_te = master[te]

    # --- Tier 2, scored on every row: `won` is always visible
    #
    # AUC AT THE LOGGED BID, CALIBRATION AT THE RECOMMENDED ONE, and they are
    # different questions rather than a choice between two conventions.
    #
    # `won` is the realised outcome at the bid the generator recorded, so a
    # ranking scored anywhere else would pair each score with a label from a
    # different counterfactual. AUC therefore stays at the logged bid.
    #
    # Calibration asks whether a stated probability happens that often, and the
    # only probabilities the bidder acts on are the ones at the rung it chose.
    # Both sides of the logged-bid figure come from the data, so it never touched
    # the bid the algorithm recommends: measured here, the two differ on 85
    # percent of rows, 0.307 against 0.439. The logged figure is kept beside it
    # because v2's thirty committed results.json files mean it by `ece_win`.
    y_win = d_te["won"].to_numpy(dtype=float)
    ece_l, mce_l = calibration_or_nan(y_win, p_win, bins)

    # Calibration at the recommended bid is scored on the rows the bidder ACTS
    # ON. A row the ROAS gate declined has a stated probability the bidder never
    # bet on, so including it would measure a counterfactual the policy refused.
    # At the default target nothing is declined and the two row sets coincide,
    # which is why `n_at_recommended` still equals `n`; `placed_rate` in the
    # economics is what says whether that is still true.
    target = B.roas_target(settings)
    ev_price = B.to_price_unit(p1["ev"])
    bid_reco, _, rung, placed = B.choose(ev_price, curve, prices, target)
    p_reco = curve[np.arange(len(rung)), rung]
    hurdle = np.maximum(m_te["lu7_competing_bid"].to_numpy(dtype=float),
                        m_te["floor_price"].to_numpy(dtype=float))
    y_reco = (bid_reco >= hurdle).astype(float)[placed]
    ece_r, mce_r = calibration_or_nan(y_reco, p_reco[placed], bins)

    res["heads"]["win"] = {
        "n": int(len(y_win)), "auc": M.auc(y_win, p_win),
        "ece_at_recommended": ece_r, "mce_at_recommended": mce_r,
        "ece_at_logged": ece_l, "mce_at_logged": mce_l,
        "n_at_recommended": int(len(y_reco)),
    }
    # --- the price head, scored against the TRUTH in every view. C1 and C2 never
    # see m^win and are still measured against it, exactly as the economics are
    # scored counterfactually. That comparison IS the ablation's question.
    m_win_true = np.maximum(m_te["lu7_competing_bid"].to_numpy(dtype=float),
                            m_te["floor_price"].to_numpy(dtype=float))
    p_price = ph.predict(d_te)
    res["heads"]["price"] = dict(
        PR.score(m_win_true, p_price),
        sigma=ph.sigma, exact_labels=bool(ph.exact),
        features=len(pr_cols), n_train=int(ph.n_train))

    # SIGMA IS READ OFF HERE, where the residuals already are, and only on a view
    # whose labels were exact. `run_seed` then hands C3's value to C1 and C2
    # rather than recomputing anything. Recorded per seed so it is visible: v1's
    # scale was 1.0 from nowhere and nothing in its output said so.
    if ph.exact:
        used, raw = PR.sigma_from_residuals(m_win_true, p_price, settings)
        res["heads"]["price"]["sigma_hat"] = used
        res["heads"]["price"]["sigma_raw"] = raw
        res["heads"]["price"]["sigma_clamped"] = bool(abs(used - raw) > 1e-12)
    res["price_cols"] = list(pr_cols)
    res["roas_target"] = target
    # ONE truth curve for both bidders. It is the exact win rule, identical
    # every time, and 672 MB at 10M.
    truth_curve = B.true_win_curve(
        m_te["lu7_competing_bid"].to_numpy(dtype=float),
        m_te["floor_price"].to_numpy(dtype=float), prices)
    res["economics"] = B.run_policies(m_te, p1["ev"], curve, prices, target,
                                      truth_curve=truth_curve)

    # ------------------------------------------------------------------------
    # THE SECOND BIDDER, step 7. The price head's own lognormal IS a win curve,
    # so it bids over the same ladder by the same argmax.
    #
    # ITS LEVELS ARE NOT COMPARABLE WITH THE CLASSIFIER'S and must never be put
    # beside them. The two curves choose different bids, so they win different
    # impressions; only contrasts WITHIN a bidder mean anything. v1 learned this
    # the hard way, reporting an AFT bidder that showed higher profit in every
    # cell, which was evidence of nothing.
    #
    # WHY A SECOND HEAD AT ALL. `won` is visible in all four views, so the
    # classifier fits the same label on the same rows everywhere and only its
    # FEATURES can differ. Every SSP economic contrast in v2 was therefore null
    # by construction rather than by finding. This head's target is observed
    # exactly by C3 and C4 and only bounded by C1 and C2, so SSP value can now
    # enter through LABEL quality as well.
    floor_te = m_te["floor_price"].to_numpy(dtype=float)
    del curve                    # the classifier is finished with it
    gc.collect()
    curve_p = PR.win_curve(p_price, prices, ph.sigma, floor_te)
    res["economics_price"] = B.run_policies(m_te, p1["ev"], curve_p, prices,
                                            target, truth_curve=truth_curve)

    # scored ALIKE: AUC at the logged bid, calibration at the recommended one,
    # placed rows only. Same rows, same labels, same key names as `win`.
    p_win_pr = PR.win_at(p_price, d_te["bid_price"].to_numpy(dtype=float),
                         ph.sigma, floor_te)
    ece_pl, mce_pl = calibration_or_nan(y_win, p_win_pr, bins)
    bid_pr, _, rung_pr, placed_pr = B.choose(ev_price, curve_p, prices, target)
    p_reco_pr = curve_p[np.arange(len(rung_pr)), rung_pr]
    y_reco_pr = (bid_pr >= hurdle).astype(float)[placed_pr]
    ece_pr, mce_pr = calibration_or_nan(y_reco_pr, p_reco_pr[placed_pr], bins)
    res["heads"]["win_price"] = {
        "n": int(len(y_win)), "auc": M.auc(y_win, p_win_pr),
        "ece_at_recommended": ece_pr, "mce_at_recommended": mce_pr,
        "ece_at_logged": ece_pl, "mce_at_logged": mce_pl,
        "n_at_recommended": int(len(y_reco_pr)), "sigma": ph.sigma,
    }

    # THE SIGMA SENSITIVITY, 7d's targeted guard. The v1 rule that only claims
    # agreeing across both heads may be quoted is NOT adopted, because the two
    # heads measure different mechanisms and a disagreement is the informative
    # case. What replaces it is narrower: any claim resting on this head must be
    # shown insensitive to sigma. Costs no refit -- sigma enters the curve and
    # the scorer, never the fit -- so all three are scored here per seed and the
    # aggregator forms the contrast at each.
    #
    # ONLY THE LEARNED POLICY, so it calls `economics` rather than
    # `run_policies`. The oracle and truth_ev arms do not depend on sigma at
    # all -- the oracle never sees it and truth_ev would only repeat what the
    # headline already reports -- so running all three here built two more
    # truth curves and four more argmax arrays for numbers nothing reads. That
    # was 4 GB of allocation per view at 10M.
    lu7_te = m_te["lu7_competing_bid"].to_numpy(dtype=float)
    ev_true_te = B.to_price_unit(m_te["ev_truth"].to_numpy(dtype=float))
    bid_logged = d_te["bid_price"].to_numpy(dtype=float)
    res["sigma_sweep"] = {}
    for tag, mult in (("lo", 0.75), ("hi", 1.5)):
        sg = ph.sigma * mult
        cv = PR.win_curve(p_price, prices, sg, floor_te)
        b_sg, _, _, pl_sg = B.choose(ev_price, cv, prices, target)
        del cv
        ec = M.economics(b_sg, None, ev_true_te, lu7_te, floor_te, pl_sg)
        ec["mean_bid"] = float(np.mean(b_sg))
        # `value_vs_oracle` is DELIBERATELY ABSENT here, and its absence is the
        # only shape difference between this dict and the headline one. It
        # divides by the oracle's value, and the oracle does not depend on sigma
        # at all, so inside a sigma sweep it would be profit rescaled by a
        # constant -- a second copy of the row above it. The aggregator reads
        # `profit` and `auc` from this block and nothing else.
        res["sigma_sweep"][tag] = {
            "sigma": sg,
            "auc": M.auc(y_win, PR.win_at(p_price, bid_logged, sg, floor_te)),
            "economics": ec,
        }
    gc.collect()
    # what this view's VALUATIONS are worth before any bidder touches them,
    # both per impression, which is v1's `ev_ratio` diagnostic
    res["ev"] = B.ev_bias(p1["ev"], m_te["ev_truth"].to_numpy(dtype=float))
    return res, {"tier1": t1, "tier2": t2, "price": ph, "encoders": enc,
                 "t1_cols": t1_cols, "t2_cols": t2_cols, "price_cols": pr_cols}


def calibration_or_nan(y, p, bins):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return M.calibration(y, p, bins)


def sigma_order(views):
    """The view order the price head forces: C3 first, then everything else.

    THE ONE ORDERING RULE step 6 costs. C1 and C2 train on intervals, so their
    likelihood is shaped by the AFT scale and they cannot be fitted until it is
    known. It is read off C3's residuals, so C3 has to go first. C3 itself is
    indifferent to the value it is given, because a normal AFT with exact labels
    has a location MLE independent of the scale — which is exactly why reading it
    off afterwards is nearly free rather than needing a pilot run.

    The RESULT is still returned in C1..C4 order. Only the fitting is reordered,
    so nothing downstream can tell.
    """
    vs = list(views)
    return (["C3"] + [v for v in vs if v != "C3"]) if "C3" in vs else vs


def run_seed(master, settings=None, seed=0, views=None, quiet=False,
             bundle_dir=None, sigma=None):
    """All 4 views on one master. The comparison the whole study is.

    `bundle_dir` freezes each view's fitted pieces to `<bundle_dir>/<view>/`.
    Without it the models die with this function, which is what v2 did: a new
    metric then cost a full retrain, and a retrain does not reproduce the model.

    SAVED BEFORE THE RELEASE, NOT AFTER, and the `del` below is why. At 10M a
    master frame is about 5 GB and holding two views' worth at once exceeds this
    machine, so the model has to be written while it is still in hand. Anything
    wanting it later loads the bundle.

    `sigma` overrides the value read off C3 and exists for callers that train a
    censored view WITHOUT C3 — a single-view test, mostly. In the campaign it is
    always None and always derived, because deriving it is the point.
    """
    import gc
    from pathlib import Path

    s = settings or load("default")
    order = sigma_order(views or CEN.VIEWS)
    out = {}
    for v in order:
        r, fitted = train_view(master, v, s, seed, sigma=sigma)
        out[v] = r
        if v == "C3" and sigma is None:
            # SIGMA, TAKEN FROM C3 AND HANDED TO C1 AND C2. `train_view` already
            # computed it where the residuals were, so nothing is recomputed and
            # there is no second definition to drift from the first. This is the
            # replacement for v1's AFT_SCALE = 1.0, which was copied from nowhere
            # and sat directly on the study's headline contrast.
            sigma = r["heads"]["price"]["sigma_hat"]
            if not quiet:
                print("  sigma from C3: %.4f%s" % (
                    sigma, " (CLAMPED from %.4f)" % r["heads"]["price"]["sigma_raw"]
                    if r["heads"]["price"]["sigma_clamped"] else ""))
        if bundle_dir is not None:
            from .. import bundle as BU
            root = BU.save_bundle(fitted, s, Path(bundle_dir) / v,
                                  meta={"view": v, "seed": int(seed),
                                        "rows": int(len(master))})
            r["bundle"] = Path(root).name
        del fitted                    # the models and their frames, per view
        gc.collect()
        if not quiet:
            h = r["heads"]
            e = r["economics"]["learned"]
            print("  %-3s %2d cols  click AUC %.4f  install %.4f  win %.4f  "
                  "price rmse_log %.4f  profit %.1f  captured %.4f  "
                  "of oracle %.4f  ev level %.3f order %.3f"
                  % (v, r["columns"], h["click"]["auc"], h["install"]["auc"],
                     h["win"]["auc"], h["price"]["rmse_log"], e["profit"],
                     e["value_captured"], e["value_vs_oracle"], r["ev"]["ratio"],
                     r["ev"]["spearman"]))
    # C1..C4 again. Only the FITTING was reordered, so nothing downstream -- the
    # aggregator, the eval pass, a reader of results.json -- can tell.
    return {v: out[v] for v in (views or CEN.VIEWS) if v in out}
