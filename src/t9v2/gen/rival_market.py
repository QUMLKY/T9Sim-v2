# -*- coding: utf-8 -*-
"""The market block: floor, participation, rival bids, our bid, win, settlement.

Steps 4 to 9 of specification 2.1, and the implementation of 2.5, the
Private-Rival Market. The one property this module exists to produce is that
rivals hold PRIVATE information about the user that our bidder cannot see. The
dial rho scales how much of it enters their bids, and at rho = 0 the rivals bid
noise around a shared public core, which is the iid market that produced the v8
null result.

Two rules hold throughout and both are load-bearing:

  our bidder never reads LU7, and never reads any latent or the oracle. It bids
  from observable slot, category and tier quantities only. If it could see the
  competing bid the whole ablation would collapse, because C1 would already know
  what only C3 is supposed to learn.

  first price. The winner pays its own bid, so `winning_price` on a won row IS
  our bid and carries no information we did not already have. That is why no
  condition observes its own overpayment, and why bid shading is out of scope.
"""
from __future__ import annotations

import numpy as np

from ..core import rng as R
from ..network.laws import law
from ..pools import read_pmf

FORMATS = ["banner", "interstitial", "rewarded"]


def price_shapes():
    """The two iPinYou price shapes, median-normalised.

    Both are resampled as empirical pmfs rather than fitted, so the floor's zero
    atom keeps its MEASURED mass of about 14 percent. The register carried 57
    percent until 16 August 2026; that number was the floor median over the
    paying median, a different quantity entirely.
    """
    fv, fp = read_pmf("ipinyou_floor_bid_distribution.csv", "floor_price")
    pv, pp = read_pmf("ipinyou_paying_distribution.csv", "paying_price")
    med_f = _weighted_median(fv, fp)
    med_p = _weighted_median(pv, pp)
    return (fv / med_f, fp), (pv / med_p, pp)


def _weighted_median(values, weights):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


@law("floor_shape")
def draw_floor_shape(rng, n, shape):
    """T1. Resample the median-normalised iPinYou floor pmf, atom at 0 kept."""
    return R.resample_pmf(rng, n, shape[0], shape[1])


@law("pay_shape")
def draw_pay_shape(rng, n, shape):
    """T1. Resample the median-normalised iPinYou paying-price pmf."""
    return R.resample_pmf(rng, n, shape[0], shape[1])


def _ecpm(fmt, s):
    t = s.raw["ecpm_targets_usd"]
    return np.array([t[f]["value"] for f in FORMATS])[fmt - 1]


@law("H1")
def floor_price(shape, fmt, s):
    """T4. floor_shape x the format's eCPM target. A zero shape stays zero, so
    the no-floor rows survive the scaling as genuine zeros."""
    return shape * _ecpm(fmt, s)


@law("base_e")
def price_core(shape, fmt, s):
    """T4. max(pay_shape x eCPM target, 0.01 x target).

    The shared, DSP-predictable core of every rival bid: the part of the price
    that depends only on what the slot is, with no private information in it.
    The guard keeps the core strictly positive so its logarithm exists, which the
    bid law needs.
    """
    target = _ecpm(fmt, s)
    return np.maximum(shape * target, 0.01 * target)


@law("participate_k")
def participate(rng, rivals, exch, day):
    """T3. Bernoulli per rival: pi[k, exchange] x flight_k(day) x pace gate.

    Rival 0 is forced true. Without that, an auction could have no rivals at all
    and LU7 would be a maximum over an empty set; forcing one participant is what
    makes the competing bid always defined.

    Drawn K times per auction row. The node's plate is `auction` because it
    reduces to auction scope through LU7 and H9, and the K axis is the reduction
    plate the register names in its law rather than a second graph plate.
    """
    K = rivals["K"]
    pace = rivals["pace"][:, day].T                       # (n, K)
    gate = 0.5 + 1.0 / (1.0 + np.exp(-pace))
    p = np.clip(rivals["pi"][:, exch].T * rivals["flight"][:, day].T * gate, 0.0, 1.0)
    part = rng.random((len(exch), K)) < p
    part[:, 0] = True
    return part


@law("b_k")
def rival_bids(rng, rivals, base_e_v, z, os_idx, dev, day, rho, s):
    """T3. log b_k = log base_e + rho*(w_k*z + beta_R*R[segment, k] + pace_k(day))
    + sigma_k * N(0, 1).

    The rho term is the private structure. w_k is how much of the user's true
    value rival k can see, R is its retargeting taste for this os-by-device
    segment, and pace is its budget state that day. At rho = 0 every rival bids
    lognormal noise around the shared core and the market is iid, which is
    exactly the v8 configuration whose SSP result was null.
    """
    beta_R = s.raw["rival_pool"]["beta_R"]["value"]
    seg = (os_idx.astype(np.int64) * len(("phone", "tablet")) + dev.astype(np.int64))
    private = (rivals["w"][None, :] * z[:, None]
               + beta_R * rivals["R"][seg, :]
               + rivals["pace"][:, day].T)
    noise = rivals["sigma"][None, :] * rng.standard_normal((len(z), rivals["K"]))
    return np.exp(np.log(base_e_v)[:, None] + rho * private + noise)


@law("LU7")
def competing_bid(bids, part):
    """T4. The maximum over PARTICIPATING rivals. A plate reduction over k.

    Latent in every view. No condition observes it, which is what makes the
    competing bid the thing C3 has to infer from SSP signals rather than read.
    """
    return np.where(part, bids, -np.inf).max(axis=1)


@law("H9")
def bid_density(part):
    """T4. The participant count, domain {1..8}. SSP-only: absent in C1 and C2,
    present in C3 and C4. It is the single column that carries the SSP result."""
    return part.sum(axis=1).astype(np.int8)


@law("eltv_b2")
def observable_ltv(cat, s):
    """T4. E[ltv | category], the DSP's OBSERVABLE value estimate.

    Deliberately not the oracle e_ltv. Our bidder knows the category average and
    nothing about this user, which is the information asymmetry the whole study
    measures. Using e_ltv here would make the bidder clairvoyant.
    """
    from .auction import category_location
    mu = category_location(cat, s)
    sd = s.raw["ltv"]["lognormal_sigma"]["value"]
    return np.exp(mu + sd * sd / 2.0)


@law("H2")
def our_bid(rng, tier_of_row, v_slot, ease_v, eltv, s, k_global=None):
    """T3. k_global * k_cpa(tier) * v_slot * ease * eltv_b2 * shade, times
    LogNormal(0, sigma_explore).

    No LU7 input and no latent input, checked by inspection: the arguments are
    the tier, the slot, the category ease and the category-average LTV. The
    exploration noise is what gives Tier 2 a spread of bids to learn a win curve
    from; without it every row at a given slot would bid the same and the curve
    would be a step.
    """
    kc = s.raw["advertiser_scale"]["k_cpa"]
    k_cpa = np.array([kc[t]["value"] for t in ("indie", "mid", "major")])[tier_of_row]
    shade = s.raw["auction"]["shade"]["value"]
    sig = s.raw["auction"]["sigma_explore"]["value"]
    kg = s.raw["auction"]["k_global"]["value"] if k_global is None else k_global
    loc = kg * k_cpa * v_slot * ease_v * eltv * shade
    return loc * rng.lognormal(0.0, sig, len(eltv))


@law("H3")
def won(bid, lu7, floor):
    """T4. 1[our bid >= max(competing bid, floor)]. First price, no tie-break
    beyond the inequality."""
    return (bid >= np.maximum(lu7, floor)).astype(np.int8)


@law("sold_lost")
def sold_but_lost(won_v, lu7, floor):
    """T4. (1 - won) * 1[LU7 >= floor]. The branch selector for the winning
    price: a lost auction that still cleared the floor was sold to a rival, and
    one that did not was not sold at all."""
    return ((won_v == 0) & (lu7 >= floor)).astype(np.int8)


@law("H4")
def winning_price(won_v, sold, bid, lu7):
    """T4. won -> our own bid (first price); sold to a rival -> LU7; unsold -> NaN.

    NaN and not zero. An unsold auction has no winning price, and coding that as
    zero would put a real price of zero in the same column as an absent one.
    """
    out = np.full(len(bid), np.nan, dtype=np.float64)
    out[won_v == 1] = bid[won_v == 1]
    m = (won_v == 0) & (sold == 1)
    out[m] = lu7[m]
    return out
