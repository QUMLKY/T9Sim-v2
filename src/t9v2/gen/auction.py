# -*- coding: utf-8 -*-
"""One auction row: the join, the context, and the truth block.

Steps 1 to 3 of specification 2.1. The join picks which user meets which app and
which campaign; the context stamps the exchange, the slot and the clock; the
truth block computes the true conversion probabilities and the expected value
BEFORE anything is priced or drawn. That order is the point. The market reads the
truth through a standardised score and nothing ever feeds back, which is what
makes ev_truth an oracle rather than a fitted quantity.
"""
from __future__ import annotations

import numpy as np

from ..core import rng as R
from ..network import laws as L
from ..network.laws import law
from ..pools import ARCHETYPES, CATEGORIES, DEVICE, OS

FORMATS = ["banner", "interstitial", "rewarded"]        # codes 1, 2, 3
STAGES = ["click", "install", "pay"]


# ---------------------------------------------------------------------------
# step 1: the join
# ---------------------------------------------------------------------------

def build_pair_table(s, users, apps):
    """The (archetype x app) joint that dependency #1 draws from.

    Seed cell (k, a) = pi_k * (1/A) * (1 + strength * (la2_a[k]/pi_k - 1)), so at
    strength 0 the grid is the independent product and at strength 1 it follows
    the app's audience profile exactly. The cells are FLOORED AT 1e-12 before the
    rake, because an app whose audience gives an archetype zero mass would
    otherwise pin that cell at zero and no amount of raking could move it.

    Then IPF onto BOTH margins: the archetype mix and uniform app popularity. That
    is the whole design of the edge. It tilts who meets whom and leaves how often
    anyone appears exactly where the pools put it, so wiring #1 cannot move the
    archetype mix or make some apps more popular than others.
    """
    strength = s.raw["dependency_strengths"]["pairing_strength"]["value"]
    pi = np.array([s.raw["archetype_shares"][a]["value"] for a in ARCHETYPES])
    pi = pi / pi.sum()
    la2 = apps["la2"]                                    # (A, 5)
    A = la2.shape[0]

    seed = pi[None, :] * (1.0 / A) * (1.0 + strength * (la2 / pi[None, :] - 1.0))
    seed = np.maximum(seed.T, 1e-12)                     # (5, A), archetype-major
    return L.ipf_joint(seed, pi, np.full(A, 1.0 / A))


@law("pair_idx")
def draw_pair(rng, n, table):
    """T2. One draw of the (archetype, app) CELL from the raked joint.

    A joint table with an empty per-row parent set: nothing about the row selects
    the cell, the cell selects the row's archetype and app together. That is why
    the register gives it no parents and why the scope rule calls LA2 and the
    pairing strength build-time inputs rather than parents.
    """
    flat = table.ravel()
    return R.choice_p(rng, n, flat)


@law("app_i")
def decode_app(pair_idx, n_apps):
    """T4. The app half of the decoded cell."""
    return (pair_idx % n_apps).astype(np.int64)


def decode_archetype(pair_idx, n_apps):
    """The archetype half. Not a node: it is pair_idx's own row index, and the
    user drawn from it carries lu1_archetype as its own column."""
    return (pair_idx // n_apps).astype(np.int64)


@law("u_rows")
def draw_user(rng, arch_of_cell, groups):
    """T3. A uniform member draw inside the drawn archetype's user group.

    T3 and not T2: the law is 'uniform over the members of this group', which is
    an implicit index draw, and specification 2.0 says such a draw stays T3
    because its pmf is never written down as a table.
    """
    out = np.empty(len(arch_of_cell), dtype=np.int64)
    for k, members in enumerate(groups):
        m = arch_of_cell == k
        cnt = int(m.sum())
        if cnt:
            if len(members) == 0:
                raise ValueError("archetype %s was drawn but no user in the pool "
                                 "has it; the pool is too small for its shares"
                                 % ARCHETYPES[k])
            out[m] = members[rng.integers(0, len(members), cnt)]
    return out


@law("user_vbin")
def user_value_bins(users, n_bins=10):
    """T4. Decile bins of the standardised log of LU4 x LU5.

    Two things the law has to get right and a naive reading gets wrong.

    The z-score is taken over the POSITIVE-VALUE users only. About 30 percent of
    users are the inactive archetype, for whom LU4 = LU5 = 0 exactly, and log 0
    is undefined; standardising over everyone would evaluate it.

    Those zero-value users are then placed BELOW the standardised range, so they
    collapse into the bottom bin together. They are not spread across the deciles
    and they do not shift the quantiles of the users who do have value.

    Computed once over the pool, not per row: it is a property of the user, and
    the auction inherits it with the rest of the user's row.
    """
    v = users["lu4_payer_prob"] * users["lu5_ltv_mult"]
    pos = v > 0
    z = np.full(len(v), -np.inf, dtype=np.float64)
    if pos.any():
        lg = np.log(v[pos])
        sd = lg.std()
        z[pos] = (lg - lg.mean()) / (sd if sd > 0 else 1.0)
    edges = np.quantile(z[pos], np.linspace(0, 1, n_bins + 1)[1:-1]) if pos.any() else []
    b = np.digitize(z, edges).astype(np.int8)
    b[~pos] = 0
    return b, z


def build_cidx_table(s, camps, vbin, zscore, n_bins=10):
    """P(campaign | user value bin), the exposure edge (#2).

    Row b is sample_weight * exp(beta * zbar_b * standardised log LC2), where
    zbar_b is the bin's MEAN standardised value score, not the bin index and not
    the row's own score. Then each row is normalised and the whole table is IPF
    raked so every campaign still receives exactly its budget share of traffic.

    The rake is what makes this edge honest. Without it, tilting high-value users
    toward high-quality games would also hand those campaigns more traffic, and
    the two effects would be inseparable in the result.
    """
    beta = s.raw["dependency_strengths"]["exposure_beta"]["value"]
    w = camps["sample_weight"]
    lq = np.log(camps["lc2_game_quality"])
    lq = (lq - lq.mean()) / (lq.std() if lq.std() > 0 else 1.0)

    zbar = np.zeros(n_bins, dtype=np.float64)
    finite = np.isfinite(zscore)
    for b in range(n_bins):
        m = (vbin == b) & finite
        if m.any():
            zbar[b] = zscore[m].mean()

    rows = w[None, :] * np.exp(beta * zbar[:, None] * lq[None, :])
    rows /= rows.sum(axis=1, keepdims=True)
    bin_w = np.array([(vbin == b).sum() for b in range(n_bins)], dtype=np.float64)
    bin_w = bin_w / bin_w.sum()
    return L.ipf_conditional(rows, bin_w, w / w.sum(), tol=1e-8, iterations=300)


@law("c_idx")
def draw_campaign(rng, vbin_rows, table):
    """T2. The campaign, drawn from the row's own value-bin pmf."""
    out = np.empty(len(vbin_rows), dtype=np.int64)
    for b in range(table.shape[0]):
        m = vbin_rows == b
        cnt = int(m.sum())
        if cnt:
            out[m] = R.choice_p(rng, cnt, table[b])
    return out


# ---------------------------------------------------------------------------
# step 2: the context
# ---------------------------------------------------------------------------

@law("B3")
def draw_exchange(rng, n, s):
    """T1. Per-auction draw from the 2025 US mediation shares."""
    ex = s.raw["context"]["ad_exchanges"]
    names = list(ex.keys())
    return R.choice_p(rng, n, [ex[k]["value"] for k in names]).astype(np.int8)


@law("B6")
def draw_format(rng, n, s):
    """T1. Slot format, integer codes banner 1, interstitial 2, rewarded 3."""
    sf = s.raw["context"]["slot_format_shares"]
    return (R.choice_p(rng, n, [sf[f]["value"] for f in FORMATS]) + 1).astype(np.int8)


@law("_size")
def draw_size(rng, fmt, s):
    """T2 nested. The WxH label from the format's own pmf. Never a column: the
    two components are, and they are split from this."""
    table = s.raw["tables"]["size_given_format"]["values"]
    out = np.empty(len(fmt), dtype=object)
    for i, f in enumerate(FORMATS):
        m = fmt == i + 1
        cnt = int(m.sum())
        if cnt:
            keys = np.array(list(table[f].keys()), dtype=object)
            out[m] = keys[R.choice_p(rng, cnt, list(table[f].values()))]
    return out


@law("B4")
def slot_width(size):
    """T4. The first component of the label split."""
    return np.array([int(x.split("x")[0]) for x in size], dtype=np.int16)


@law("B5")
def slot_height(size):
    """T4. The second component of the same split."""
    return np.array([int(x.split("x")[1]) for x in size], dtype=np.int16)


@law("D4")
def draw_week(rng, n, s):
    """T1. A uniform integer over the whole-week blocks of the window.

    NOT a calibrated pmf. No week-share vector exists in any settings file, and
    the register says so rather than implying one. Generator-internal, never a
    column: the timestamp carries it.
    """
    n_weeks = max(1, s.raw["time"]["window_days"]["value"] // 7)
    return rng.integers(0, n_weeks, n).astype(np.int8)


@law("D2")
def draw_hour(rng, arch, tables):
    """T2 raked. The hour from the archetype's own row of the von Mises CPT."""
    cpt = tables["D2"]
    out = np.empty(len(arch), dtype=np.int8)
    for k in range(cpt.shape[0]):
        m = arch == k
        cnt = int(m.sum())
        if cnt:
            out[m] = R.choice_p(rng, cnt, cpt[k])
    return out


@law("D3")
def draw_dow(rng, arch, tables):
    """T2 raked. Day of week from the archetype's row; 0 is Monday."""
    cpt = tables["D3"]
    out = np.empty(len(arch), dtype=np.int8)
    for k in range(cpt.shape[0]):
        m = arch == k
        cnt = int(m.sum())
        if cnt:
            out[m] = R.choice_p(rng, cnt, cpt[k])
    return out


@law("D1")
def make_timestamp(rng, week, dow, hour, s):
    """T3. window start + ((7*week + dow)*24 + hour)*3600 + U{0..3599}.

    Assembled HOUR FIRST, per open item O9: the hour and day are drawn from their
    own laws and the timestamp is built to agree with them, never the reverse.
    The sub-hour offset is the only randomness here.
    """
    import datetime as _dt
    start = _dt.datetime.fromisoformat(
        s.raw["time"]["window_start_utc"]["value"]).replace(tzinfo=_dt.timezone.utc)
    base = int(start.timestamp())
    day = 7 * week.astype(np.int64) + dow.astype(np.int64)
    return (base + (day * 24 + hour.astype(np.int64)) * 3600
            + rng.integers(0, 3600, len(week))).astype(np.int64)


def day_index(week, dow):
    """The day within the 28-day window. Open item O3 3e anchors the rival day
    index to the window START, so a rival's flight and pacing line up with the
    calendar rather than with an arbitrary offset."""
    return (7 * week.astype(np.int64) + dow.astype(np.int64))


# ---------------------------------------------------------------------------
# step 3: the truth block
# ---------------------------------------------------------------------------

@law("r_genre")
def genre_relevance(lu6_rows, genre):
    """T4. The user's interest weight in the campaign's genre, LU6[ad_genre]."""
    return lu6_rows[np.arange(len(genre)), genre]


@law("m_stage")
def stage_multipliers(r, s):
    """T4. m_s = (1 - w_s) + w_s * r, one per funnel stage.

    Relevance matters more the further down the funnel: w rises click to install
    to pay, so an interested user converts disproportionately rather than merely
    clicking more.
    """
    w = s.raw["funnel_relevance_weights"]
    return {st: (1.0 - w[st]["value"]) + w[st]["value"] * r for st in STAGES}


@law("v_slot")
def slot_value(fmt, size, s):
    """T4. format weight x size weight, landing in roughly [0.45, 2]."""
    q = s.raw["context"]["slot_quality"]
    fw = np.array([q["format_weight"][f]["value"] for f in FORMATS])
    sw = q["size_weight"]
    sz = np.array([sw[str(x)]["value"] for x in size], dtype=np.float64)
    return fw[fmt - 1] * sz


@law("ease")
def install_ease(cat, s):
    """T4. Install ease per app category."""
    e = s.raw["app_categories"]["install_ease"]
    return np.array([e[c]["value"] for c in CATEGORIES])[cat]


@law("plat")
def platform_multiplier(os_idx, s):
    """T4. The iOS spend multiplier, dependency #3. 1.8 on iOS, 1.0 on Android."""
    m = s.raw["dependency_strengths"]["ios_ltv_multiplier"]["value"]
    return np.where(os_idx == 0, m, 1.0)


def expected_plat(s):
    """E[plat] under the population os split. mu_cat subtracts its log, so
    dependency #3 sets the iOS-to-Android RATIO and not the overall LTV level."""
    m = s.raw["dependency_strengths"]["ios_ltv_multiplier"]["value"]
    p_ios = s.raw["device"]["os_split"]["iOS"]["value"]
    return m * p_ios + 1.0 * (1.0 - p_ios)


@law("mu_cat")
def category_location(cat, s):
    """T4. mu_cat = lognormal_mu + ln(category LTV tier) - ln E[plat].

    The final term is the recentre, constraint R5: without it, wiring the iOS
    multiplier would raise the population LTV mean as well as tilting it by
    platform, and the $6 median anchor would no longer hold.
    """
    mu = s.raw["ltv"]["lognormal_mu"]["value"]
    tier = s.raw["app_categories"]["ltv_multiplier"]
    lt = np.array([tier[c]["value"] for c in CATEGORIES])
    return mu + np.log(lt)[cat] - np.log(expected_plat(s))


@law("t_pay")
def payer_timing(hour, dow, tables):
    """T4. The hour-by-day payer multiplier, raked to population mean 1."""
    return tables["t_pay"][hour, dow]


@law("p_click")
def true_p_click(users_rows, v_slot, m, la1, lc1, s):
    """T4. clip(base_ctr * LU2 * v_slot * m_click * LA1 * LC1, 0, 1)."""
    base = s.raw["funnel"]["base_ctr"]["start"]["value"]
    return np.clip(base * users_rows * v_slot * m["click"] * la1 * lc1, 0.0, 1.0)


@law("p_install")
def true_p_install(lu3, ease_v, m, la1, s):
    """T4. clip(base_ir * LU3 * ease * m_install * LA1, 0, 1).

    No creative-appeal term. The asymmetry against click is deliberate: a
    creative wins the click, the app wins the install.
    """
    base = s.raw["funnel"]["base_ir"]["start"]["value"]
    return np.clip(base * lu3 * ease_v * m["install"] * la1, 0.0, 1.0)


@law("p_payer")
def true_p_payer(lu4, m, t_pay_v, s):
    """T4. clip(base_payer * LU4 * m_pay, 0, 1), then times t_pay and re-clipped.

    Two clips, not one. Clipping once at the end is a different function wherever
    the first clip binds, and the register states the order for that reason.
    """
    base = s.raw["funnel"]["base_payer"]["start"]["value"]
    p = np.clip(base * lu4 * m["pay"], 0.0, 1.0)
    return np.clip(p * t_pay_v, 0.0, 1.0)


@law("e_ltv")
def expected_ltv(mu_cat_v, lu5, lc2, plat_v, s):
    """T4. exp(mu_cat + sigma^2/2) * LU5 * LC2 * plat, the lognormal MEAN."""
    sd = s.raw["ltv"]["lognormal_sigma"]["value"]
    return np.exp(mu_cat_v + sd * sd / 2.0) * lu5 * lc2 * plat_v


@law("ev_truth")
def expected_value(p_c, p_i, p_p, e_l):
    """T4. The product of the four. Ground truth, formula frozen.

    Exactly zero for the inactive archetype, because LU4 and LU5 are both fixed
    at zero upstream. That exact zero is what the value-bin collapse and the
    log-space adverse-selection check both depend on.
    """
    return p_c * p_i * p_p * e_l


@law("z")
def value_score(ev, z_mu, z_sigma):
    """T4. (log ev - z_mu)/z_sigma on ev > 0, else 0.

    z_mu and z_sigma are frozen on a warm-up sample BEFORE generation and never
    updated from the rows they score. If they were re-estimated per block the
    market would be reading a statistic of itself, and the oracle would stop
    being exogenous.
    """
    out = np.zeros(len(ev), dtype=np.float64)
    pos = ev > 0
    if pos.any():
        out[pos] = (np.log(ev[pos]) - z_mu) / (z_sigma if z_sigma > 0 else 1.0)
    return out
