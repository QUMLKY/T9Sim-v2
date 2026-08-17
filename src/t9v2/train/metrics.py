# -*- coding: utf-8 -*-
"""Scoring: ranking, calibration, the spend distribution, and the economics.

Calibration is MEASURED and never fitted, per open item O7. Nothing here adjusts
a model; ece and mce are reported per head per view and neither gates. A head can
have a small ece and a bad mce when one region is badly off, which is why both
are reported rather than one.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def auc(y, p):
    """Rank AUC by the Mann-Whitney identity, ties averaged.

    Written out rather than imported so a tie between two identical scores is
    handled explicitly: the funnel heads produce many exactly-equal predictions
    on sparse cells, and a tie-blind implementation reads them as a coin flip in
    whichever direction the sort happened to put them.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration(y, p, bins=10):
    """(ece, mce) over equal-width probability bins.

    ece is the average gap between predicted probability and observed frequency,
    weighted by bin population, so it says how wrong calibration is on average.
    mce is the largest single-bin gap, so it says how wrong it gets at its worst.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    mce = 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        gap = abs(float(p[m].mean()) - float(y[m].mean()))
        ece += gap * n / len(y)
        mce = max(mce, gap)
    return float(ece), float(mce)


def crps_lognormal(y, mu, sigma):
    """Mean CRPS of a LogNormal(mu, sigma) predictive against observations y.

    Closed form, from the standard result for the lognormal:

        CRPS = y (2 F((ln y - mu)/s) - 1)
               - 2 exp(mu + s^2/2) [ F((ln y - mu - s^2)/s) + F(s/sqrt2) - 1 ]

    A closed form rather than a sample estimate because the spend distribution has
    a heavy tail: a Monte Carlo CRPS on a lognormal with s near 1.6 needs an
    impractical number of draws before its own noise is below the differences
    being measured.

    Zero observations are dropped rather than clamped. A non-payer's spend is an
    exact zero and is not a draw from this distribution at all; the head is
    E(spend | payer) and is only ever scored on payers.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    ok = (y > 0) & np.isfinite(y) & np.isfinite(mu) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return float("nan")
    y, mu, s = y[ok], mu[ok], np.broadcast_to(s, y.shape)[ok]
    ly = np.log(y)
    z = (ly - mu) / s
    term1 = y * (2.0 * norm.cdf(z) - 1.0)
    term2 = 2.0 * np.exp(mu + s ** 2 / 2.0) * (
        norm.cdf(z - s) + norm.cdf(s / np.sqrt(2.0)) - 1.0)
    return float(np.mean(term1 - term2))


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((y[ok] - yhat[ok]) ** 2))) if ok.any() else float("nan")


# ---------------------------------------------------------------------------
# economics
# ---------------------------------------------------------------------------

def economics(bid, won_truth, value_truth, price_to_beat, floor):
    """What a bidding policy would actually have earned, scored against the truth.

    Counterfactual and exact, not simulated: the master holds the competing bid
    and the floor for every row, so whether a proposed bid would have won is a
    comparison rather than an estimate. In a first-price auction the winner pays
    its own bid, so profit on a won row is value minus bid.

    Returned per policy:
      wins      how many auctions the policy takes
      spend     what it pays, which is its own bid on the rows it wins
      value     the true expected value it acquires
      profit    value minus spend
      ev_ratio  the share of all available true value the policy captures, which
                is the ranking-quality measure that does not depend on the level

    The denominator of ev_ratio is every row's value including the rows no
    policy should buy, so nothing reaches 1.0 and the oracle itself reads about
    0.90. `run_policies` adds `value_vs_oracle` alongside it for the reading
    against the achievable ceiling.
    """
    bid = np.asarray(bid, float)
    hurdle = np.maximum(np.asarray(price_to_beat, float), np.asarray(floor, float))
    win = bid >= hurdle
    v = np.asarray(value_truth, float)
    spend = float(bid[win].sum())
    value = float(v[win].sum())
    total = float(v.sum())
    return {"wins": int(win.sum()), "win_rate": float(win.mean()),
            "spend": spend, "value": value, "profit": value - spend,
            "ev_ratio": value / total if total > 0 else float("nan")}
