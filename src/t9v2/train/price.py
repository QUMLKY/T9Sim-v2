# -*- coding: utf-8 -*-
"""The price head: one estimand, four views, one likelihood.

The quantity is `m^win`, the minimum winning price — `max(LU7, floor)`, the
column H5 deposits. Every view predicts the same thing. What differs is how well
each can OBSERVE it, which is the ablation working as intended rather than a
methodological difference smuggled in beside it:

    C3, C4        exact on every row, read straight off `min_winning_price`
    C1, C2 won    floor <= m^win <= bid, a bounded interval
    C1, C2 lost   m^win >= max(bid, floor), right-unbounded

WHY C1 AND C2 KEEP A CENSORED MODEL, in Ken's words: a real DSP without SSP
integration simply does not have the luxury of training a price head on observed
prices. That is exactly the deprivation the ablation exists to measure, so the
censored likelihood is not a workaround, it is the condition.

ONE LIKELIHOOD, NOT TWO. All four views use the same interval-censored class and
C3/C4 supply degenerate intervals [x, x]. Plain regression on one side and a
censored model on the other would make `C3 - C1` a mixture of the data layer and
the estimator, and no reader could tell which produced the contrast. With one
class the estimator is held constant by construction.

So the step's original title, "survival to plain regression", is the wrong
description of what this is. It is the SAME censored regression throughout, with
C3 and C4's intervals collapsed to points by the new column.

THE DIVISION OF LABOUR BETWEEN THE TWO TIER-2 HEADS is what makes the two
mechanisms separable rather than confounded:

    the classifier   varies FEATURES, holds the target fixed
    the price head   varies the LABEL, holds features fixed

A `C3 - C1` gap on the classifier means the price history was informative. The
same gap here means label QUALITY was. Each head holds the other axis constant.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from . import features as F

# every SSP-derived encoder, barred from this head's features by name. Listed
# rather than pattern-matched so adding one is a decision here, not a default.
SSP_ENCODERS = ["_enc_ssp_price", "_enc_ssp_density", "_enc_ssp_lost_price",
                "_enc_ssp_minwin_price"]


def mwp_bounds(df):
    """The interval-censored label for `m^win`, as (lower, upper).

        won    lower = floor_price          upper = bid_price
        lost   lower = max(bid, floor)      upper = +inf

    ONE FUNCTION, NOT A COLUMN, and that is a deliberate refusal. An interval is
    two numbers, so materialising it would take two columns, and `mwp_upper` is
    finite exactly when `won == 1` — which makes it the Tier-2 label in disguise
    and something that would itself need banning from every feature list.
    Writing the censoring boundary into the master invites the leak it exists to
    prevent. The property is verified instead: a test asserts
    `lower <= max(LU7, floor) <= upper` on every row of every view, checked
    against the master where both are known. A column could not be checked that
    way; it would just hold whatever this same arithmetic produced.

    THE LOWER BOUND ON LOST ROWS IS CLOSED, and writing it open would be wrong on
    11.7 percent of rows. Losing means `bid < m^win`, which alone gives an OPEN
    bound at the bid. But `m^win >= floor` always, and on an unsold row
    `LU7 < floor`, so `m^win` equals the floor EXACTLY. Where the bid sits under
    the floor the binding constraint is the floor, attained rather than exceeded,
    and an open bound would exclude the true value on every unsold row.
    XGBoost's `label_lower_bound` is closed by convention, so `max(bid, floor)`
    is the right arithmetic; it was the notation that had to be fixed.
    """
    won = df["won"].to_numpy() == 1
    bid = df["bid_price"].to_numpy(dtype=float)
    floor = df["floor_price"].to_numpy(dtype=float)
    lower = np.where(won, floor, np.maximum(bid, floor))
    upper = np.where(won, bid, np.inf)
    return lower, upper


def exact_label(df):
    """`m^win` itself where the view can see it, else None.

    C3 and C4 hold `min_winning_price`; C1 and C2 do not, and there is no way to
    reconstruct it from what they hold, which is the whole point.
    """
    if "min_winning_price" not in df.columns:
        return None
    return df["min_winning_price"].to_numpy(dtype=float)


def features(view_df, enc_cols):
    """The price head's features, IDENTICAL in all four views.

    That identity is what makes this step's contrast readable, so it is enforced
    by a raise rather than left to convention. Base columns minus `bid_price`,
    plus `_enc_dsp_price`, and nothing else.

    `bid_price` IS DROPPED because it is the label's own censoring boundary.
    Given it as a feature the model learns to predict just under the bid on won
    rows and just over it on lost ones, memorising the censoring pattern instead
    of the price. It still enters at SCORING time, as the point the curve is
    evaluated at, which is a different role entirely.

    THE SSP ENCODERS ARE DROPPED because they exist only in C3 and C4. Including
    them would make this head's feature list differ by view, and its `C3 - C1`
    gap would then mix features with labels — destroying the one property that
    lets the two Tier-2 heads separate the mechanisms.

    THE CALLER PASSES EVERY ENCODER ITS VIEW HAS and this function selects; the
    raises below check this function's OWN OUTPUT, not its input. Rejecting the
    caller would put the rule at each call site, which is where rules drift.
    Asserting the result means a later edit to the selection cannot widen it
    quietly.
    """
    cols = [c for c in F.base_columns(view_df) if c != "bid_price"]
    keep = [c for c in (enc_cols or []) if c == "_enc_dsp_price"]
    if not keep:
        raise RuntimeError(
            "the price head needs `_enc_dsp_price`, which every view has. Without "
            "it the head loses the one price signal available everywhere.")
    out = cols + keep

    bad = [c for c in out if c in SSP_ENCODERS]
    if bad:
        raise RuntimeError(
            "the price head's feature list contains %s. It must be identical in "
            "all four views, and these exist only in C3 and C4; including them "
            "would make its C3 minus C1 gap a mixture of features and labels, "
            "which destroys the one property that lets the two Tier-2 heads "
            "separate the mechanisms." % ", ".join(bad))
    if "bid_price" in out:
        raise RuntimeError(
            "the price head's feature list contains `bid_price`, which IS the "
            "label's censoring boundary. Given it, the model predicts just under "
            "the bid on won rows and just over on lost ones, memorising the "
            "censoring pattern instead of the price.")
    return out


def sigma_from_residuals(y_true, y_pred, settings):
    """The AFT scale, read off exact residuals rather than chosen.

        sigma_hat = std(log m^win - log pred)

    ONLY LEGITIMATE ON A VIEW WITH EXACT LABELS, which is why it is taken from
    C3. The analytic argument is that a normal AFT with exact observations has a
    location MLE independent of the scale; MEASURED, a boosted AFT is not a pure
    MLE and sigma does move it a little, because it scales the gradients and so
    shifts both the splits and the early-stopping point. C3's rmse_log runs
    0.9136 to 0.9060 across sigma from 0.2 to 4.0 — 0.0076 over a twentyfold
    span, a real response rather than scatter, and negligible. So the value read
    off afterwards is near enough free, but "entirely unaffected" would be wrong.

    A DEFENSIBLE SOURCE, NOT A DERIVATION. C3's residual spread and C1's
    likelihood scale are different quantities sharing a unit. C1 predicts worse
    than C3, so if sigma tracked residual spread C1 would want a LARGER value;
    its accuracy optimum is 0.50, smaller. Nothing legitimate does better,
    though: 0.50 is reachable only by minimising C1's error against `m^win`,
    which C1 never observes, and C1's own censored likelihood picks 0.44 and
    scores 2.90 against 1.26. Sigma is weakly identified from interval data.
    Rebuild plan v2.2, step 6 and Q3.

    Clamped, because a pathological seed must not be able to produce nonsense,
    and the clamp is a guard rather than a tuning range. If it ever binds, that
    is a signal to look rather than a value to accept.
    """
    cfg = settings.raw["xgboost"]["sigma"]
    lo, hi = cfg["clamp_low"]["value"], cfg["clamp_high"]["value"]
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    if ok.sum() < 100:
        raise RuntimeError(
            "sigma needs exact labels to be read off and got %d usable rows. "
            "Refusing to fall back on a constant: an unjustified scale on the "
            "censored views is the defect this replaces." % int(ok.sum()))
    raw = float(np.std(np.log(y[ok]) - np.log(p[ok])))
    return float(np.clip(raw, lo, hi)), raw


class PriceHead:
    """Interval-censored AFT on `m^win`. The same class in every view."""

    def __init__(self):
        self.model, self.cols, self.n_train = None, None, 0
        self.sigma = None
        self.exact = None          # did this view train on degenerate intervals

    def fit(self, d, masks, cols, settings, sigma, seed=0):
        """`sigma` is required and has no default.

        No default on purpose. v1's scale was 1.0 copied from nowhere and it sat
        directly on the study's headline contrast; a default here would let the
        same thing happen quietly. C3 and C4 are handed the placeholder because
        the normal-AFT location MLE ignores it when every label is exact, and C1
        and C2 are handed the value read off C3.
        """
        p = dict(settings.raw["xgboost"]["tier2_price"]["value"])
        p.update(random_state=seed, n_jobs=0,
                 aft_loss_distribution_scale=float(sigma))
        self.cols, self.sigma = list(cols), float(sigma)

        lower, upper = mwp_bounds(d)
        y_exact = exact_label(d)
        if y_exact is not None:
            # a degenerate interval [x, x]: same likelihood class, no censoring
            lower = y_exact.copy()
            upper = y_exact.copy()
        self.exact = y_exact is not None

        # AFT is on a positive scale, and a floor of exactly zero is 14 percent
        # of rows. Zero is a legal lower bound for the censored form but not a
        # legal exact label, so exact zeros are lifted to a token minimum rather
        # than dropped: dropping them would train C3 and C4 on a different row
        # set from C1 and C2 and break the one-likelihood property.
        eps = 1e-6
        lower = np.maximum(lower, 0.0)
        upper = np.where(np.isfinite(upper), np.maximum(upper, eps), np.inf)
        if self.exact:
            lower = np.maximum(lower, eps)
            upper = np.maximum(upper, eps)

        tr, va = masks["train"], masks["valid"]
        dtr = xgb.DMatrix(F.matrix(d[tr], cols), enable_categorical=True)
        dtr.set_float_info("label_lower_bound", lower[tr])
        dtr.set_float_info("label_upper_bound", upper[tr])
        dva = xgb.DMatrix(F.matrix(d[va], cols), enable_categorical=True)
        dva.set_float_info("label_lower_bound", lower[va])
        dva.set_float_info("label_upper_bound", upper[va])

        rounds = int(p.pop("n_estimators", 300))
        stop = int(p.pop("early_stopping_rounds", 20))
        p.pop("enable_categorical", None)
        self.model = xgb.train(p, dtr, num_boost_round=rounds,
                               evals=[(dva, "valid")],
                               early_stopping_rounds=stop, verbose_eval=False)
        self.n_train = int(tr.sum())
        return self

    def predict(self, d):
        """The predicted `m^win`, on the price scale rather than the log scale."""
        dm = xgb.DMatrix(F.matrix(d, self.cols), enable_categorical=True)
        return self.model.predict(dm)


def score(y_true, y_pred):
    """Scored against the TRUTH from the master, in every view.

    C1 and C2 never SEE `m^win`, but they are still measured against it, exactly
    as the economics are scored counterfactually against the master's own
    competing bid. That is the ablation's question stated as a number: how much
    worse is a price head that could only ever observe an interval.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    if not ok.any():
        return {"n": 0, "rmse_log": float("nan"), "mae_log": float("nan"),
                "bias_log": float("nan"), "ratio": float("nan")}
    ly, lp = np.log(y[ok]), np.log(p[ok])
    return {"n": int(ok.sum()),
            "rmse_log": float(np.sqrt(np.mean((ly - lp) ** 2))),
            "mae_log": float(np.mean(np.abs(ly - lp))),
            # SIGNED, and the sign is informative. Measured on 100K seed 20250:
            # C3 and C4 read -0.1076 and C1 and C2 read +0.4167, so the censored
            # views OVER-predict. Right-censoring is why. About 70 percent of
            # rows are lost, and a lost row's interval is [max(bid, floor), +inf]
            # -- it says "at least this much" and nothing more, so the likelihood
            # has no evidence pulling those rows down and the fit drifts up. The
            # won rows' bounded intervals are the minority and cannot offset it.
            "bias_log": float(np.mean(lp - ly)),
            "ratio": float(p[ok].mean() / y[ok].mean())}


# ---------------------------------------------------------------- the bidder

def win_curve(m_hat, prices, sigma, floor):
    """P(win | b) from a predicted price, as a (rows, prices) array.

        P(win | b) = Phi( (log b - log m_hat) / sigma ) . 1[b >= floor]

    THE DISTRIBUTION IS ALREADY A WIN CURVE. The head predicts the location of a
    lognormal over `m^win`, and the win rule is `won = bid >= max(LU7, floor)`,
    so the probability of winning at `b` is just the mass of that lognormal below
    `b`. Nothing is fitted here; this is the win rule written down.

    THE FLOOR TERM IS NOT v1's, AND ITS ABSENCE THERE WAS A DEFECT. v1's AFT
    never saw `floor_price`, so its curve was the Phi alone and it assigned
    positive win probability to bids that cannot win at all. That is part of why
    v1's AFT bidder overpaid its classifier two to three times over. The floor is
    observable in EVERY view, so including it costs the ablation nothing.

    MONOTONE BY CONSTRUCTION, unlike the classifier's curve. `m_hat` does not
    depend on `b`, Phi increases in `b`, and the indicator is a single step up,
    so the product is non-decreasing on every row without relying on what the
    trees learned. The classifier needs `monotone_constraints {bid_price: 1}` to
    get the same property, and a constraint can only be as good as its fit.
    """
    m = np.maximum(np.asarray(m_hat, dtype=float), 1e-12)[:, None]
    b = np.asarray(prices, dtype=float)[None, :]
    z = (np.log(b) - np.log(m)) / float(sigma)
    return _phi(z) * (b >= np.asarray(floor, dtype=float)[:, None])


def win_at(m_hat, bid, sigma, floor):
    """The same curve evaluated at ONE bid per row, for `auc_win`.

    Scored at the LOGGED bid, because `won` is the outcome at the bid the row
    actually carried. A ranking scored at per-row recommended bids would pair
    each score with a label from a different counterfactual.
    """
    m = np.maximum(np.asarray(m_hat, dtype=float), 1e-12)
    b = np.asarray(bid, dtype=float)
    z = (np.log(np.maximum(b, 1e-12)) - np.log(m)) / float(sigma)
    return _phi(z) * (b >= np.asarray(floor, dtype=float))


def _phi(z):
    """Standard normal CDF."""
    from scipy.stats import norm
    return norm.cdf(z)
