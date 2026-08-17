# -*- coding: utf-8 -*-
"""Tier 2: P(win | bid), the win curve the bidder searches over.

ONE head, a classifier. Ken dropped v1's AFT price model on 15 August 2026 and
the reason is worth keeping next to the code: in v1 the AFT bidder showed an SSP
overpayment reduction that the classifier showed to be about zero, and the
classifier was right. A classifier assumes nothing about the shape of the price
distribution, and that assumption was the difference between a true answer and a
false one.

MONOTONE IN THE BID, open item O8. `monotone_constraints: {bid_price: 1}` forces
the fitted curve to be non-decreasing in the bid. That is a fact about a
first-price auction rather than a modelling preference: you win by outbidding the
field, so bidding more can never win less often. Without it a tree fits a sag
wherever the training data is thin, and the bidder maximises (ev - b) . p(win|b),
so a sag near the peak moves the chosen bid to the wrong place.

A small drop in raw win AUC is EXPECTED and is not a regression: the constraint
removes the model's freedom to fit noise in that direction. What matters is
whether the bidder's chosen bid improves.

The constraint is keyed by feature NAME, not position, so it cannot silently
attach to the wrong column if the feature list changes.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from . import features as F


class Tier2:
    """The win classifier, and the sweep that turns it into a curve."""

    def __init__(self):
        self.model, self.cols, self.n_train = None, None, 0

    def fit(self, d, masks, cols, settings, seed=0):
        p = dict(settings.raw["xgboost"]["tier2_win_classifier"]["value"])
        mono = p.pop("monotone_constraints", {}) or {}
        # keyed by name, resolved to a tuple in feature order here. A constraint
        # given positionally would move to whatever column happened to take that
        # slot the next time the feature list changed.
        p["monotone_constraints"] = tuple(int(mono.get(c, 0)) for c in cols)
        p.update(random_state=seed, n_jobs=0,
                 n_estimators=p.get("n_estimators", 300),
                 learning_rate=p.get("learning_rate", 0.1))

        self.cols = list(cols)
        tr, va = masks["train"], masks["valid"]
        y = d["won"].to_numpy(dtype=float)
        if len(np.unique(y[tr])) < 2:
            raise RuntimeError(
                "the win classifier has only one class in the training split, so "
                "there is nothing to learn. If `won` is masked in this view that "
                "is the defect, not the data: specification 1.5 makes it always "
                "visible.")
        self.model = xgb.XGBClassifier(early_stopping_rounds=20, **p)
        self.model.fit(F.matrix(d[tr], cols), y[tr],
                       eval_set=[(F.matrix(d[va], cols), y[va])], verbose=False)
        self.n_train = int(tr.sum())
        return self

    def predict(self, d):
        return self.model.predict_proba(F.matrix(d, self.cols))[:, 1]

    def win_curve(self, d, prices):
        """p(win | bid = b) for every candidate b, as a (rows, prices) array.

        The bidder needs the whole curve rather than the probability at the bid
        that happened to be made, because it is choosing a different bid. The
        frame is rewritten once per candidate price, which is what makes the
        monotone constraint matter: this is read across the price axis.
        """
        out = np.empty((len(d), len(prices)), dtype=float)
        work = d.copy()
        for j, b in enumerate(prices):
            work["bid_price"] = b
            out[:, j] = self.predict(work)
        return out
