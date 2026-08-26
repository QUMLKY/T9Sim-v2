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

# Prices are CPM, dollars per THOUSAND impressions; a row is one impression.
# Defined here rather than in bidders because this is the module that divides by
# it, and bidders already imports metrics so the constant can travel that way
# without a cycle.
PER_MILLE = 1000.0

def economics(bid, won_truth, value_truth, price_to_beat, floor, placed=None):
    """What a bidding policy would actually have earned, scored against the truth.

    Counterfactual and exact, not simulated: the master holds the competing bid
    and the floor for every row, so whether a proposed bid would have won is a
    comparison rather than an estimate. In a first-price auction the winner pays
    its own bid, so profit on a won row is value minus bid.

    THE MONEY IS DIVIDED BY A THOUSAND, and that is the 1000x units fix.

    Every price reaching here is a CPM, dollars per thousand impressions, because
    that is the unit the market clears in and the unit a bid must be comparable
    with a value in. But a won row is ONE impression. Taking it at 9.6 CPM costs
    $0.0096, not $9.60, so summing the CPM once per won row reports a thousand
    times the money.

    Caught by comparing v2's oracle profit against v1's: 16,741,566 against
    19,202, a factor of 1000 and a scale v1 never had. v1 divides and says so in
    its own results file: "profit, total = profit CPM x n_won / 1000".

    ONLY THE MONEY TOTALS MOVE. `value_captured` divides one sum by another of the
    same unit so the factor cancels, `win_rate` is a count, and the argmax is
    scale-invariant. No bid, no ranking metric and no supported-or-not verdict
    changes; the profit LEVELS do, by exactly 1000.

    `placed` IS THE ROAS GATE, and it is passed rather than inferred from the
    bid. Encoding a declined row as a zero bid would make the refusal depend on
    the hurdle never being zero, which is a property of the rival draw rather
    than of the design: the floor is exactly zero on 14 percent of rows and only
    a strictly positive competing bid keeps the hurdle above it. A refusal has
    to be stated. None places every row, which is the ungated behaviour and
    reproduces v2 exactly.

    Returned per policy:
      wins        how many auctions the policy takes
      placed_rate the share of rows it was willing to bid on at all
      spend       what it pays IN DOLLARS: its own bid on the rows it wins
      value       the true expected value it acquires, in dollars
      profit      value minus spend, in dollars
      value_captured  the share of all available true value the policy takes,
                      unitless and therefore untouched by the units fix
      profit_per_1k_wins  profit divided by wins, times a thousand

    `value_captured` WAS CALLED `ev_ratio`, RENAMED 23 August 2026, and the old
    name was a historical accident that came to mean two different things. The
    other one is `bidders.ev_bias`'s `ratio`, the EV LEVEL: mean predicted value
    over mean true value, a property of the VALUATIONS before any bidder acts.
    This one is a property of the BIDDER: what share of available value it took,
    which depends on the win curve, the ladder and the argmax. At 1M the level
    reads 0.302 for C1 against C2's 0.853 while captured reads 0.139 against
    0.229 — different quantities, different sizes, one name. A BREAKING CHANGE
    to results.json: v2's thirty committed files carry the old key, and the
    design fingerprint is what tells the two schemas apart.

    `profit_per_1k_wins` WAS CALLED `profit CPM`, and that name hid its own
    denominator. CPM is per thousand IMPRESSIONS and every price column here is
    one, but this divides by WINS, so the old label invited comparison with
    `value_captured`, whose denominator is all rows. It carries no verdict and
    stays in the no-claim set.

    The denominator of `value_captured` is every row's value including the rows
    no policy should buy, so nothing reaches 1.0 and the oracle itself reads
    about 0.90. `run_policies` adds `value_vs_oracle` alongside it for the
    reading against the achievable ceiling.
    """
    bid = np.asarray(bid, float)
    hurdle = np.maximum(np.asarray(price_to_beat, float), np.asarray(floor, float))
    place = (np.ones(len(bid), dtype=bool) if placed is None
             else np.asarray(placed, dtype=bool))
    win = place & (bid >= hurdle)
    v = np.asarray(value_truth, float)
    won_value = float(v[win].sum())
    total = float(v.sum())
    # a won row is ONE impression and every price here is per thousand
    spend = float(bid[win].sum()) / PER_MILLE
    value = won_value / PER_MILLE
    n_win = int(win.sum())
    profit = value - spend
    return {"wins": n_win, "win_rate": float(win.mean()),
            "placed_rate": float(place.mean()),
            "spend": spend, "value": value, "profit": profit,
            "profit_per_1k_wins": (profit / n_win * 1000.0 if n_win
                                   else float("nan")),
            "value_captured": won_value / total if total > 0 else float("nan")}
