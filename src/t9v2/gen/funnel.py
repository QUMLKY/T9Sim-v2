# -*- coding: utf-8 -*-
"""The funnel: click, install, payer, spend.

Step 10, and the correction that the whole ablation turns on. THE FUNNEL IS DRAWN
ON EVERY ROW, won or lost. No law in this module reads `won`, and that is
checkable by inspection: `won` is not an argument to any function here.

Why it matters. The outcomes exist on all rows in the ground truth; what differs
between the conditions is who gets to SEE them. C1 and C3 observe E, F and G on
won rows only, C2 and C4 on all rows. If the generator only drew outcomes where
we won, the C1-to-C2 contrast would be measuring the generator's own gap rather
than a selection-bias correction, and proposition P3 would be vacuous.

Gate identities are exact, not approximate. No click means no install, ever, and
a non-payer spends exactly zero. Each is a branch of one law, which is why these
rows are T3 with a point mass rather than two mechanisms stitched together.
"""
from __future__ import annotations

import numpy as np

from ..network.laws import law

NO_EVENT = -1          # an OBSERVED no-event timestamp, distinct from missing


@law("E1")
def click(rng, p_click):
    """T3. Bernoulli(p_click) on every row."""
    return (rng.random(len(p_click)) < p_click).astype(np.int8)


@law("E2")
def click_timestamp(rng, click_v, ts, s):
    """T3. No click -> -1; click -> floor(timestamp + Exponential(mean 30 s)).

    -1 is a SENTINEL BRANCH, not missing data. A censored cell in a view is NaN;
    this is an observed fact about a row where the event did not happen, and the
    two must not be confused when a model reads the column.
    """
    mean = s.raw["funnel_delays"]["click_delay_mean_s"]["value"]
    out = np.full(len(click_v), NO_EVENT, dtype=np.int64)
    m = click_v == 1
    if m.any():
        out[m] = np.floor(ts[m] + rng.exponential(mean, int(m.sum()))).astype(np.int64)
    return out


@law("F1")
def install(rng, click_v, p_install):
    """T3. No click -> 0 exactly; click -> Bernoulli(p_install).

    The gate identity is exact and is an acceptance test: install = 0 wherever
    click = 0, on every row, with no tolerance.
    """
    out = np.zeros(len(click_v), dtype=np.int8)
    m = click_v == 1
    if m.any():
        out[m] = (rng.random(int(m.sum())) < p_install[m]).astype(np.int8)
    return out


@law("F2")
def install_timestamp(rng, install_v, click_ts, s):
    """T3. No install -> -1; install -> floor(click_ts + LogNormal(mu, sigma)),
    a median of about 30 minutes with a wide right tail in hours."""
    cfg = s.raw["funnel_delays"]["install_delay"]
    mu, sd = cfg["mu"]["value"], cfg["sigma"]["value"]
    out = np.full(len(install_v), NO_EVENT, dtype=np.int64)
    m = install_v == 1
    if m.any():
        out[m] = np.floor(click_ts[m] + rng.lognormal(mu, sd, int(m.sum()))).astype(np.int64)
    return out


@law("G1")
def is_payer(rng, install_v, p_payer):
    """T3. No install -> 0 exactly; install -> Bernoulli(p_payer)."""
    out = np.zeros(len(install_v), dtype=np.int8)
    m = install_v == 1
    if m.any():
        out[m] = (rng.random(int(m.sum())) < p_payer[m]).astype(np.int8)
    return out


@law("G2")
def ltv_value(rng, payer_v, mu_cat, lu5, lc2, plat, s):
    """T3. Non-payer -> 0 exactly; payer -> LogNormal(mu_cat, sigma) * LU5 * LC2
    * plat. The declared 90-day post-install total.

    Two branches under ONE law, which is what makes this T3 rather than a mix.
    Category and OS stay upstream through the deterministic mediators mu_cat and
    plat, which is why neither is itself a parent here.
    """
    sd = s.raw["ltv"]["lognormal_sigma"]["value"]
    out = np.zeros(len(payer_v), dtype=np.float64)
    m = payer_v == 1
    if m.any():
        out[m] = rng.lognormal(mu_cat[m], sd, int(m.sum())) * lu5[m] * lc2[m] * plat[m]
    return out


@law("G3")
def ltv_7d(ltv, s):
    """T4. 0.40 x the 90-day total, the 7-day recognition point."""
    return s.raw["ltv"]["decay_d7"]["value"] * ltv


@law("G4")
def ltv_30d(ltv, s):
    """T4. 0.70 x the 90-day total, the 30-day recognition point."""
    return s.raw["ltv"]["decay_d30"]["value"] * ltv
