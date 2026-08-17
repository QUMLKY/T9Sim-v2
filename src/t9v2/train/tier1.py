# -*- coding: utf-8 -*-
"""Tier 1: the 4-head hurdle model of the funnel.

    P(click) . P(install | click) . P(payer | install) . E(spend | payer)

A hurdle rather than one model of spend, because each stage conditions on the one
before it and the generator's gates are exact: no click means no install, ever.
Modelling spend directly would ask one model to learn a point mass at zero that is
about 99.96 percent of rows, and the product of four well-posed problems is better
posed than one badly-posed one.

Each head trains on the rows its condition selects, and never sees an outcome at
or below its own stage; features.py enforces that.

HEAD 4 IS DISTRIBUTIONAL, open item O12 closed at this stage. XGBoost predicts the
LOCATION of log spend and the scale is the residual standard deviation measured on
the VALIDATION split, so the spread is not fitted to the rows that fixed the mean.
The plan scores spend CRPS, which needs a predictive distribution; a point
regression has none. The fitted scale is reported per view and never configured.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from . import features as F

HEADS = ["click", "install", "payer", "spend"]


def _rows_for(d, head):
    """The rows a head trains and is scored on, from its conditioning."""
    if head == "click":
        return np.ones(len(d), dtype=bool)
    if head == "install":
        return d["click"].to_numpy() == 1
    if head == "payer":
        return d["install"].to_numpy() == 1
    if head == "spend":
        return d["is_payer"].to_numpy() == 1
    raise ValueError(head)


def _label(d, head):
    return {"click": "click", "install": "install",
            "payer": "is_payer", "spend": "ltv_value"}[head]


def _observed(d):
    """Rows whose funnel labels this view actually shows.

    In C1 and C3 the funnel is masked on lost rows, so the label is NA there and
    those rows cannot train or score a funnel head. This is the censoring, not a
    missing-data problem to impute: it is exactly what the ablation measures.
    """
    return d["click"].notna().to_numpy()


# Below this many training rows a boosted tree cannot be fitted honestly, and the
# head falls back to an INTERCEPT: the training base rate for a binary head, the
# training mean and spread of log spend for the spend head.
#
# This is not a workaround, it is the correct model when there is nothing to
# condition on, and saying so is better than the alternative. The alternative was
# worse than it looks: a head that returns nothing makes ev exactly zero, the
# bidder then bids the bottom of the ladder in every view, and the run reports
# profit and ev_ratio near zero everywhere. That reads as a modelling result and
# is an absence of one.
#
# It bites hardest exactly where the study is pointed. A payer is about 4 rows in
# 10,000, so a 100K master holds ~49 payers and only ~6 of them on won rows. C1
# and C3 see won rows only, so their spend head has single digits to learn from at
# this scale. That IS the censoring the ablation measures, and the plan already
# says profit is not judged at 100K.
MIN_FIT_ROWS = 50
MIN_VALID_ROWS = 10


class Head:
    """A fitted head, or the intercept that stands in for one."""

    def __init__(self, head, mode, model=None, const=None, scale=None, n=0):
        self.head, self.mode, self.model = head, mode, model
        self.const, self.scale, self.n = const, scale, n

    def predict(self, d, cols):
        if self.mode == "unavailable":
            return np.full(len(d), np.nan)
        if self.mode == "intercept":
            return np.full(len(d), self.const)
        return predict_head(self.model, d, cols, self.head)


def fit_head(d, masks, head, cols, settings, seed=0):
    """Fit one head, falling back to an intercept when the rows are too few."""
    par = settings.raw["xgboost"]
    p = dict(par["tier1_spend" if head == "spend" else "tier1_binary"]["value"])
    p.update(random_state=seed, n_jobs=0)
    early = p.pop("early_stopping_rounds", None)

    sel = _rows_for(d, head) & _observed(d)
    tr, va = masks["train"] & sel, masks["valid"] & sel
    y_col = _label(d, head)

    if tr.sum() < MIN_FIT_ROWS or va.sum() < MIN_VALID_ROWS:
        n = int(tr.sum())
        if n == 0:
            return Head(head, "unavailable", n=0), None, 0
        y = d.loc[tr, y_col].to_numpy(dtype=float)
        if head == "spend":
            ly = np.log(y[y > 0])
            if not len(ly):
                return Head(head, "unavailable", n=n), None, n
            s = float(np.std(ly)) if len(ly) > 1 else 1.0
            return Head(head, "intercept", const=float(np.mean(ly)),
                        scale=s or 1.0, n=n), (s or 1.0), n
        return Head(head, "intercept", const=float(np.mean(y)), n=n), None, n
    Xtr, Xva = F.matrix(d[tr], cols), F.matrix(d[va], cols)

    if head == "spend":
        # the location of LOG spend. Modelling the level directly would let one
        # whale dominate the loss, and the generator's own spend law is lognormal
        ytr = np.log(d.loc[tr, y_col].to_numpy(dtype=float))
        yva = np.log(d.loc[va, y_col].to_numpy(dtype=float))
        m = xgb.XGBRegressor(early_stopping_rounds=early, **p)
    else:
        ytr = d.loc[tr, y_col].to_numpy(dtype=float)
        yva = d.loc[va, y_col].to_numpy(dtype=float)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            # one class only: the base rate IS the model
            return Head(head, "intercept", const=float(ytr.mean()),
                        n=int(tr.sum())), None, int(tr.sum())
        m = xgb.XGBClassifier(early_stopping_rounds=early, **p)

    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    scale = None
    if head == "spend":
        resid = yva - m.predict(Xva)
        # O12: measured on VALIDATION, so the spread is not fitted to the rows
        # that fixed the mean. Reported per view, never hand-set.
        scale = float(np.std(resid)) or 1e-6
    return Head(head, "model", model=m, scale=scale, n=int(tr.sum())), scale, int(tr.sum())


def predict_head(model, d, cols, head):
    X = F.matrix(d, cols)
    if head == "spend":
        return model.predict(X)                       # location on the log scale
    return model.predict_proba(X)[:, 1]


class Tier1:
    """The 4 heads, their fitted spend scale, and the chained expected value."""

    def __init__(self):
        self.models, self.scale, self.n_train, self.cols = {}, None, {}, None
        self.mode = {}

    def fit(self, d, masks, cols, settings, seed=0):
        self.cols = list(cols)
        for h in HEADS:
            m, s, n = fit_head(d, masks, h, cols, settings, seed)
            self.models[h] = m
            self.n_train[h] = n
            self.mode[h] = m.mode
            if h == "spend":
                self.scale = s
        return self

    def predict(self, d):
        """Per-row head predictions, plus the chained ev.

        ev = P(click) . P(install|click) . P(payer|install) . E[spend|payer], and
        the last is exp(mu + s^2/2), the MEAN of the lognormal rather than its
        median. A bidder maximising profit needs the mean; using the median would
        systematically under-value the heavy tail that the whales live in.
        """
        out = {}
        for h in HEADS:
            m = self.models.get(h)
            out[h] = (np.full(len(d), np.nan) if m is None
                      else m.predict(d, self.cols))
        s = self.scale or 0.0
        out["spend_mean"] = np.exp(out["spend"] + s * s / 2.0)
        out["ev"] = (np.nan_to_num(out["click"]) * np.nan_to_num(out["install"])
                     * np.nan_to_num(out["payer"]) * np.nan_to_num(out["spend_mean"]))
        return out
