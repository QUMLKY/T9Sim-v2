# -*- coding: utf-8 -*-
"""The bidders: what each view would actually do with what it learned.

A bidder turns two predictions into one number. Tier 1 says what a row is worth,
Tier 2 says how often a given bid wins it, and the bidder picks the bid that
maximises expected profit over a fixed ladder of candidate prices:

    b* = argmax_b  (ev - b) . p(win | b)

The ladder is shared by every policy including the oracle, so no policy can beat
another by searching a finer grid. That matters: with a free choice of bid, a
policy with a slightly better curve could win on grid resolution alone.

THREE POLICIES, and the third is the point of having the other two.

  learned   ev from this view's Tier 1, win curve from this view's Tier 2. What
            the view would really do.
  truth-ev  the TRUE ev from the master, this view's win curve. Isolates how much
            of a view's economic performance comes from valuing rows correctly
            rather than from pricing them correctly.
  oracle    true ev and the true win rule. The ceiling every view is scored
            against, and it is not achievable by any view: it knows the competing
            bid.

Profit is scored counterfactually against the master's own competing bid and
floor, so whether a proposed bid would have won is a comparison and not a
simulation.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M


# THE PRICE UNIT, and the trap it sets.
#
# Every price column in the master is an eCPM: dollars per thousand impressions.
# floor_price, bid_price, lu7_competing_bid and winning_price are all scaled from
# `ecpm_targets_usd`, and the win rule compares them to each other, so the market
# is internally consistent whatever the unit is called.
#
# ev_truth is not. It is p_click . p_install . p_payer . e_ltv, the expected
# dollars from ONE impression, so it is smaller by a factor of a thousand.
#
# Mixing them is silent. The bidder maximises (ev - b) . p(win | b); with ev a
# thousand times too small, every candidate bid looks loss-making, the bidder
# bids the floor of the ladder, wins nothing, and reports profit 0.0 and
# ev_ratio 0.0000 in all four views. That reads like a modelling failure and is
# an arithmetic one. Measured on 100K: at the correct scale the oracle captures
# 97.9 percent of available value on 10.4 percent of rows; at the wrong one, 4.8.
#
# v1's candidate_prices are stated per impression, 1e-4 to 0.12, which is 0.1 to
# 120 eCPM and spans the hurdle range well. They are converted here rather than
# rewritten in the settings, so the recorded value still matches v1's.
PER_MILLE = 1000.0


def to_price_unit(ev_per_impression):
    """Expected value per impression, restated in the eCPM the market clears in."""
    return np.asarray(ev_per_impression, dtype=float) * PER_MILLE


def ladder(settings):
    """The candidate prices in eCPM, geometrically spaced. Shared by every policy.

    Shared deliberately: with a free choice of bid, a policy with a marginally
    better curve could beat another on grid resolution alone rather than on what
    it knows.
    """
    c = settings.raw["candidate_prices"]
    return np.geomspace(c["low"]["value"] * PER_MILLE,
                        c["high"]["value"] * PER_MILLE, int(c["n"]["value"]))


def choose(ev, curve, prices):
    """argmax over the ladder of (ev - b) . p(win | b), per row.

    Ties take the LOWEST price, which is the conservative reading: if two bids
    have equal expected profit there is no reason to pay more for it.
    """
    ev = np.asarray(ev, float)[:, None]
    profit = (ev - prices[None, :]) * curve
    j = np.argmax(profit, axis=1)
    return prices[j], profit[np.arange(len(j)), j]


def true_win_curve(price_to_beat, floor, prices):
    """The exact win rule, as a step per row. No smoothing.

    v1 smoothed this into a normal cdf with a width called AFT_SCALE, which made
    the oracle ceiling depend on an arbitrary sigma. A first-price auction has a
    deterministic win rule: you win when your bid clears the higher of the
    competing bid and the floor. The step IS the truth, so the ceiling is the
    real ceiling rather than one parameterised by a smoothing choice.
    """
    hurdle = np.maximum(np.asarray(price_to_beat, float),
                        np.asarray(floor, float))[:, None]
    return (prices[None, :] >= hurdle).astype(float)


def run_policies(master_test, ev_learned, curve_learned, prices):
    """The 3 policies on one test split, each with its economics.

    Two ratios are reported, and they answer different questions.

    `ev_ratio` is the share of ALL true value on the test rows that a policy
    acquires. Its denominator does not depend on any policy, so it is stable
    across seeds and scales and is the one to compare a view against itself.

    `value_vs_oracle` divides by the oracle's own value instead, so the oracle
    reads exactly 1.000 and every view reads as a fraction of the achievable
    ceiling. Nothing can reach the whole of `ev_ratio`'s denominator: value on a
    row costs more than it is worth whenever the competing bid sits above it, so
    even a policy that knows the truth declines those rows. At 10M the oracle
    captures 0.8958, and reading a view's 0.31 against 1.0 rather than against
    0.8958 understates it by a tenth of itself.
    """
    lu7 = master_test["lu7_competing_bid"].to_numpy(dtype=float)
    floor = master_test["floor_price"].to_numpy(dtype=float)
    # both values restated in the unit the market clears in, see PER_MILLE
    ev_true = to_price_unit(master_test["ev_truth"].to_numpy(dtype=float))
    truth_curve = true_win_curve(lu7, floor, prices)

    out = {}
    for name, ev, curve in [
        ("learned", to_price_unit(ev_learned), curve_learned),
        ("truth_ev", ev_true, curve_learned),
        ("oracle", ev_true, truth_curve),
    ]:
        b, _ = choose(ev, curve, prices)
        out[name] = M.economics(b, None, ev_true, lu7, floor)
        out[name]["mean_bid"] = float(np.mean(b))

    ceiling = out["oracle"]["value"]
    for r in out.values():
        r["value_vs_oracle"] = (r["value"] / ceiling if ceiling > 0
                                else float("nan"))
    return out


def ev_bias(ev_learned, ev_truth):
    """How good a view's VALUATIONS are, before any bidder touches them.

    Two independent things, and a view can be good at one and bad at the other.

    LEVEL, as `ratio` = mean(ev_hat) / mean(ev_truth). This is v1's `ev_ratio`
    under a name that cannot be mistaken for the value-capture ratio next to it;
    v1 called it EV-bias in the code, and its oracle read 1.0 because the
    oracle's prediction IS the truth. It isolates the MMP mechanism on its own:
    a view trained on won rows only sees a value-selected sample, so it
    under-predicts value everywhere, and that shows as a ratio below 1 whatever
    the bidder then does with it.

    ORDER, as `spearman` = rank correlation of predicted against true value.
    Scale-free, so a view that gets every row's value half right but in the
    right order still scores near 1. It is the part of valuation the bidder
    actually spends: the bidder buys the rows it ranks highest, so a view can
    have a badly biased level and still bid well, and the two numbers say which
    failure is which.

    Order is reported TWICE, and the second one is not redundant. `spearman` is
    over all rows, which is what v1 reports and so is the number the two
    versions can be compared on. `spearman_active` drops the rows whose true
    value is exactly zero: about a third of rows are the inactive archetype and
    sit at an exact 0, and a tie block that large gets the average rank of its
    whole run, so it can move the all-rows figure when nothing about the model
    has changed. The active-only cut is the one that says whether the model
    orders the rows a bidder would ever compete for.

    Spearman is computed here rather than through scipy so the tie handling is
    explicit. The funnel heads produce many exactly-equal predictions on sparse
    cells, and average ranks are the standard treatment; scipy agrees to machine
    precision, but this is scored on millions of rows at 10M and ranks plus a
    corrcoef is both cheaper and one less thing that can quietly change under us.
    """
    p = np.asarray(ev_learned, dtype=float)
    t = np.asarray(ev_truth, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if not len(p):
        return {"pred_mean": float("nan"), "true_mean": float("nan"),
                "ratio": float("nan"), "rmse": float("nan"),
                "spearman": float("nan"), "spearman_active": float("nan"),
                "n_active": 0}
    tm = float(t.mean())
    act = t > 0
    return {"pred_mean": float(p.mean()), "true_mean": tm,
            "ratio": float(p.mean() / tm) if tm > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "spearman": _spearman(p, t),
            "spearman_active": (_spearman(p[act], t[act]) if act.sum() > 1
                                else float("nan")),
            "n_active": int(act.sum())}


def _average_ranks(x):
    """Ranks of x with ties averaged, which is what Spearman is defined on."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # walk the runs of equal values and flatten each to its mean rank
    new = np.empty(len(xs), dtype=bool)
    new[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(xs))
    for a, b in zip(starts, ends):
        if b - a > 1:
            r[order[a:b]] = (a + 1 + b) / 2.0
    return r


def _spearman(a, b):
    """Rank correlation. nan when either side is constant, as it is undefined."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))
