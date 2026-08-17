# -*- coding: utf-8 -*-
"""The 4 frozen pools: users, apps, campaigns, rivals.

Built once per seed, then every auction inherits one row from each. EMPTY UNTIL STAGE 2.

Built and frozen before a single auction is drawn, which is what the scope rule
in specification 2.1 rests on: a pool quantity reaches auction scope by LOOKUP,
and a lookup never closes a cycle. Nothing in this module reads anything an
auction produces, and that is checkable by inspection because no auction data
exists yet when these functions run.

Order matters in one place. Advertisers are built first and campaigns beneath
them, because a campaign's identity depends on which advertiser owns it while no
advertiser depends on its campaigns.
"""
from __future__ import annotations

import csv

import numpy as np

from .core import rng as R
from .core.config import CALIBRATION, val
from .network import laws as L
from .network.laws import law

ARCHETYPES = ["whale", "engaged_spender", "casual", "time_filler", "inactive"]
CATEGORIES = ["casual", "strategy", "rpg", "hypercasual"]
GENRES = ["casual", "strategy", "rpg", "hypercasual"]
OS = ["iOS", "Android"]
DEVICE = ["phone", "tablet"]
TIERS = ["indie", "mid", "major"]


def read_pmf(name, key, weight="proportion"):
    """A (value, proportion) calibration file as two aligned arrays."""
    with (CALIBRATION / name).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([float(r[key]) for r in rows]),
            np.array([float(r[weight]) for r in rows]))


def read_hourdow():
    """The (dow, hour) count table, and its two marginals.

    One file supplies three things: D2's base hour shape, D3's day-of-week
    marginal, and the joint they came from. Reading it once here keeps the two
    tables consistent with each other by construction.
    """
    with (CALIBRATION / "ipinyou_hourdow_distribution.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    joint = np.zeros((7, 24), dtype=np.float64)
    for r in rows:
        joint[int(r["dow"]), int(r["hour"])] += float(r["proportion"])
    joint /= joint.sum()
    return joint, joint.sum(axis=0), joint.sum(axis=1)     # joint, by hour, by dow


# ---------------------------------------------------------------------------
# tables built at load, from the recipes in tables.yaml
# ---------------------------------------------------------------------------

def build_tables(s, tau_scale=1.0, untilted=False):
    """Every runtime-built table, in one dict, built once per run.

    The four recipes in tables.yaml land here: the three tilted CPTs wired by
    open item O1 and the hour CPT. Building rather than storing is what makes
    tau = 0 an exact null; see network/laws.py.

    `tau_scale` multiplies all 3 tilt strengths at once and is what the stage-4
    sensitivity sweep varies: 0, 0.5, 1, 2. `untilted=True` builds the same 3
    tables by a DIFFERENT route, straight from the marginal with no ladder and no
    rake, which is what the tau = 0 arm is checked against. Comparing tau = 0
    against itself would prove nothing; comparing it against a generator built
    without the tilt machinery at all is the real null.
    """
    shares = {a: s.raw["archetype_shares"][a]["value"] for a in ARCHETYPES}
    ladder = s.raw["meta"]["archetype_ladder"]["scores"]
    T = s.raw["tables"]
    _, hour_marg, dow_marg = read_hourdow()

    out = {}
    os_split = {k: s.raw["device"]["os_split"][k]["value"] for k in OS}
    dev_split = {k: s.raw["device"]["device_split"][k]["value"] for k in DEVICE}
    base_dow = {d: dow_marg[d] for d in range(7)}

    if untilted:
        # every archetype gets the population marginal, full stop
        out["A3"] = {a: dict(os_split) for a in ARCHETYPES}
        out["A5"] = {a: dict(dev_split) for a in ARCHETYPES}
        row = np.array([base_dow[d] for d in range(7)], dtype=np.float64)
        out["D3"] = np.tile(row / row.sum(), (len(ARCHETYPES), 1))
    else:
        spec = T["A3_os_given_archetype"]
        out["A3"] = L.build_binary_cpt(
            os_split, ladder, ARCHETYPES,
            spec["tilt"]["tau"]["value"] * tau_scale, OS, spec["tilt"]["level"], shares)

        spec = T["A5_device_given_archetype"]
        out["A5"] = L.build_binary_cpt(
            dev_split, ladder, ARCHETYPES,
            spec["tilt"]["tau"]["value"] * tau_scale, DEVICE, spec["tilt"]["level"], shares)

        spec = T["D3_dow_given_archetype"]
        out["D3"] = L.build_dow_cpt(
            base_dow, ladder, ARCHETYPES,
            spec["tilt"]["tau"]["value"] * tau_scale, shares)

    spec = T["D2_hour_given_archetype"]
    out["D2"] = L.build_hour_cpt(
        spec["base_hour_marginal"],
        {a: spec["archetype_tilts"]["mu_kappa"][a] for a in ARCHETYPES},
        ARCHETYPES, shares)

    pt = s.raw["dependency_strengths"]["payer_timing"]
    out["t_pay"] = L.build_t_pay(pt["hour_mult"], pt["dow_mult"],
                                 {h: hour_marg[h] for h in range(24)},
                                 {d: dow_marg[d] for d in range(7)})

    out["hour_marginal"], out["dow_marginal"] = hour_marg, dow_marg
    return out


# ---------------------------------------------------------------------------
# 1. the user pool
# ---------------------------------------------------------------------------

@law("LU1")
def draw_archetype(rng, n, s):
    """T1. Categorical over the 5 archetypes, market.yaml archetype_shares."""
    p = [s.raw["archetype_shares"][a]["value"] for a in ARCHETYPES]
    return R.choice_p(rng, n, p)


@law("A1")
def draw_region(rng, n, s):
    """T1. The iPinYou region pmf, anonymised codes carried as integers."""
    vals, p = read_pmf("ipinyou_region_distribution.csv", "region")
    return vals[R.choice_p(rng, n, p)].astype(np.int32)


@law("A2")
def draw_city(rng, region, s):
    """T2 nested. City from the region's own row."""
    table = s.raw["tables"]["A2_city_given_region"]["values"]
    out = np.empty(len(region), dtype=np.int32)
    for reg, dist in table.items():
        m = region == int(reg)
        k = int(m.sum())
        if k:
            keys = np.array(list(dist.keys()), dtype=np.int32)
            out[m] = keys[R.choice_p(rng, k, list(dist.values()))]
    return out


@law("A3")
def draw_os(rng, arch, tables):
    """T2. os from the archetype's raked row (O1)."""
    cpt = np.array([[tables["A3"][a][o] for o in OS] for a in ARCHETYPES])
    u = rng.random(len(arch))
    return (u >= cpt[arch, 0]).astype(np.int8)          # 0 iOS, 1 Android


@law("A4")
def draw_os_version(rng, os_idx, s):
    """T2 nested. Version from the os's own pmf."""
    table = s.raw["tables"]["A4_osversion_given_os"]["values"]
    out = np.empty(len(os_idx), dtype=object)
    for i, name in enumerate(OS):
        m = os_idx == i
        k = int(m.sum())
        if k:
            keys = np.array(list(table[name].keys()), dtype=object)
            out[m] = keys[R.choice_p(rng, k, list(table[name].values()))]
    return out


@law("A5")
def draw_device(rng, arch, tables):
    """T2. device_type from the archetype's raked row (O1)."""
    cpt = np.array([[tables["A5"][a][d] for d in DEVICE] for a in ARCHETYPES])
    u = rng.random(len(arch))
    return (u >= cpt[arch, 0]).astype(np.int8)          # 0 phone, 1 tablet


def _beta_by_archetype(rng, arch, pairs):
    ab = np.array([val(pairs[k]) for k in ARCHETYPES], dtype=np.float64)
    return rng.beta(ab[arch, 0], ab[arch, 1])


@law("LU2")
def draw_click_prop(rng, arch, s):
    """T3. Beta(a, b) keyed by archetype."""
    return _beta_by_archetype(rng, arch, s.raw["archetype_propensity"]["click"])


@law("LU3")
def draw_install_prop(rng, arch, s):
    """T3. Beta(a, b) keyed by archetype."""
    return _beta_by_archetype(rng, arch, s.raw["archetype_propensity"]["install"])


@law("LU4")
def draw_payer_prob(rng, arch, s):
    """T3. Beta(a, b) keyed by archetype, FIXED AT ZERO for inactive.

    The forced zero is one cell of one law, not a second mechanism, and it is
    tied to LU5's fixed zero: together they make ev_truth exactly 0 for the
    inactive archetype, which is the exact identity the direction checks rely on
    and the reason user_vbin has to standardise over positive users only.
    """
    p = _beta_by_archetype(rng, arch, s.raw["archetype_propensity"]["payer"])
    p[arch == ARCHETYPES.index("inactive")] = 0.0
    return p


@law("LU5")
def draw_ltv_mult(rng, arch, s):
    """T3. LogNormal(mu, sigma) per archetype, FIXED AT ZERO for inactive.

    casual is the reference at mu = 0, so its median multiplier is 1 and the
    $6 median anchor lives only in the base LTV curve. Counting it twice is the
    defect this arrangement exists to prevent.
    """
    cfg = s.raw["archetype_ltv_multiplier"]
    mu = np.array([val(cfg[a]["mu"]) if "mu" in cfg[a] else 0.0 for a in ARCHETYPES])
    sd = np.array([val(cfg[a]["sigma"]) if "sigma" in cfg[a] else 0.0 for a in ARCHETYPES])
    out = rng.lognormal(mu[arch], np.maximum(sd[arch], 1e-12))
    out[arch == ARCHETYPES.index("inactive")] = 0.0
    return out


@law("LU6")
def draw_interest(rng, arch, s):
    """T3. ONE Dirichlet draw per user, stored as 4 columns.

    One node, not four. The four columns are the parts of a single simplex draw,
    so they are not independently drawable and the register carries them on one
    row for that reason.
    """
    cfg = s.raw["archetype_interest"]
    k = cfg["concentration_k"]["value"]
    order = cfg["order"]
    cent = np.array([val(cfg["centroids"][a]) for a in ARCHETYPES], dtype=np.float64)
    out = np.empty((len(arch), len(order)), dtype=np.float64)
    for i in range(len(ARCHETYPES)):
        m = arch == i
        if m.any():
            out[m] = rng.dirichlet(np.maximum(k * cent[i], 1e-9), size=int(m.sum()))
    return out


@law("user_id")
def make_user_id(n):
    """T4. Pool index, row lineage only. Never a feature: it is hidden in all 4
    views, so it identifies a row's origin without being learnable from one."""
    return np.arange(n, dtype=np.int64)


def build_users(s, seed, n, tables):
    """The user pool. One stream, one draw per user, nothing auction-dependent."""
    g = lambda sub: R.stream(seed, "user_pool", sub=sub)
    arch = draw_archetype(g("LU1"), n, s)
    region = draw_region(g("A1"), n, s)
    os_idx = draw_os(g("A3"), arch, tables)
    dev = draw_device(g("A5"), arch, tables)
    return {
        "user_id": make_user_id(n),
        "lu1_archetype": arch,
        "region": region,
        "city": draw_city(g("A2"), region, s),
        "os": os_idx,
        "os_version": draw_os_version(g("A4"), os_idx, s),
        "device_type": dev,
        "lu2_click_prop": draw_click_prop(g("LU2"), arch, s),
        "lu3_install_prop": draw_install_prop(g("LU3"), arch, s),
        "lu4_payer_prob": draw_payer_prob(g("LU4"), arch, s),
        "lu5_ltv_mult": draw_ltv_mult(g("LU5"), arch, s),
        "lu6": draw_interest(g("LU6"), arch, s),
    }


# ---------------------------------------------------------------------------
# 2. the app pool
# ---------------------------------------------------------------------------

@law("B1")
def make_app_id(n):
    """T4. `app_{a:05d}` from the app's own catalogue index, which is the plate
    index and therefore not a parent."""
    return np.array(["app_%05d" % i for i in range(n)], dtype=object)


@law("B2")
def assign_category(n, s):
    """T4. A COUNT QUOTA, not a draw: `n_cat = max(1, round(share * n))` apps per
    category laid down in order, then the pool is trimmed back to n.

    The max(1, .) guard is deliberate, so no category is ever empty at a small
    pool size; trimming afterwards is what keeps the pool exactly n.
    """
    share = s.raw["app_categories"]["share"]
    counts = [max(1, int(round(share[c]["value"] * n))) for c in CATEGORIES]
    out = np.concatenate([np.full(k, i, dtype=np.int8)
                          for i, k in enumerate(counts)])
    return out[:n] if len(out) >= n else np.resize(out, n)


@law("LA1")
def draw_app_quality(rng, n, s):
    """T1. LogNormal(0, sigma_app) per app."""
    return rng.lognormal(0.0, s.raw["entity_latents"]["sigma_app"]["value"], size=n)


@law("LA2")
def draw_audience(rng, cat, s):
    """T3. ONE Dirichlet per app over the 5 archetypes, stored as 5 columns.

    A BUILD-TIME INPUT to the pairing table, not a parent of it. The scope rule
    calls that a parameter edge: the table is built from this, and the draw that
    reads the table is a different node.
    """
    cfg = s.raw["app_audience"]
    k = cfg["concentration_k"]["value"]
    cent = np.array([val(cfg["centroids"][c]) for c in CATEGORIES], dtype=np.float64)
    out = np.empty((len(cat), 5), dtype=np.float64)
    for i in range(len(CATEGORIES)):
        m = cat == i
        if m.any():
            out[m] = rng.dirichlet(np.maximum(k * cent[i], 1e-9), size=int(m.sum()))
    return out


def build_apps(s, seed, n):
    g = lambda sub: R.stream(seed, "app_pool", sub=sub)
    cat = assign_category(n, s)
    return {"app_id": make_app_id(n), "app_category": cat,
            "la1_app_quality": draw_app_quality(g("LA1"), n, s),
            "la2": draw_audience(g("LA2"), cat, s)}


# ---------------------------------------------------------------------------
# 3. the campaign pool: advertisers first, campaigns beneath them
# ---------------------------------------------------------------------------

@law("adv_tier")
def draw_adv_tier(rng, n_adv, s):
    """T1. One advertiser FORCED into each tier, the remainder drawn by
    advertiser share, then the whole vector shuffled.

    The forcing guarantees every tier exists at any pool size, and it makes the
    per-advertiser draws weakly dependent, which is why the register's constraint
    cell declares them so rather than calling them iid.
    """
    share = s.raw["advertiser_scale"]["advertiser_share"]
    p = [share[t]["value"] for t in TIERS]
    if n_adv < len(TIERS):
        raise ValueError("n_advertisers = %d cannot hold one of each of the %d "
                         "tiers" % (n_adv, len(TIERS)))
    rest = R.choice_p(rng, n_adv - len(TIERS), p)
    tier = np.concatenate([np.arange(len(TIERS), dtype=np.int8), rest.astype(np.int8)])
    rng.shuffle(tier)
    return tier


@law("C1")
def make_advertiser_id(n_adv):
    """T4. `adv_{idx:03d}` over the shuffled advertiser index."""
    return np.array(["adv_%03d" % i for i in range(n_adv)], dtype=object)


@law("C2")
def make_advertiser_scale(tier):
    """T4. The tier label carried by the advertiser. A lookup, not a second draw:
    its downstream role is as the k_cpa key inside H2."""
    return tier.copy()


@law("C3")
def allocate_campaigns(tier, n_camp, s):
    """T4. Campaign counts per advertiser, proportional to tier campaign share
    over tier size, every advertiser guaranteed at least one, rounding drift
    absorbed into the advertiser with the largest weight.

    campaign_share is CAMPAIGN VOLUME, deliberately not advertiser_share, which
    is pool composition. Building volume from composition is the silent defect
    Appendix A of the specification records, and it is silent precisely because
    win-rate autocalibration would absorb the level shift.
    """
    cs = s.raw["advertiser_scale"]["campaign_share"]
    per_tier = np.array([cs[t]["value"] for t in TIERS], dtype=np.float64)
    size = np.array([(tier == i).sum() for i in range(len(TIERS))], dtype=np.float64)
    w = per_tier[tier] / np.maximum(size[tier], 1.0)
    w = w / w.sum()
    counts = np.maximum(1, np.round(w * n_camp).astype(np.int64))
    drift = n_camp - int(counts.sum())
    if drift:
        counts[int(np.argmax(w))] += drift
        counts = np.maximum(counts, 1)
    owner = np.repeat(np.arange(len(tier), dtype=np.int64), counts)[:n_camp]
    return owner, w


@law("C4")
def draw_genre(rng, n_camp, s):
    """T1. One draw per campaign at pool build, frozen thereafter. An auction
    inherits it with the campaign row, which is a lookup and not a parent edge."""
    p = [s.raw["ad_genre_mix"][gname]["value"] for gname in GENRES]
    return R.choice_p(rng, n_camp, p).astype(np.int8)


@law("LC1")
def draw_creative_appeal(rng, n, s):
    """T1. LogNormal(0, sigma_cre) per campaign."""
    return rng.lognormal(0.0, s.raw["entity_latents_extra"]["sigma_cre"]["value"], size=n)


@law("LC2")
def draw_game_quality(rng, n, s):
    """T1. LogNormal(0, sigma_game) per campaign. Drives the LTV tails, and is a
    build-time input to the c_idx exposure table."""
    return rng.lognormal(0.0, s.raw["entity_latents_extra"]["sigma_game"]["value"], size=n)


@law("sample_weight")
def campaign_weights(owner, tier, s):
    """T4. Each campaign's share of traffic: its tier's campaign share divided by
    the campaigns in that tier, normalised to sum to 1.

    This is the budget share the c_idx rake preserves. It is dropped at the join
    and never emitted, because it is a property of the campaign pool rather than
    of any row.
    """
    cs = s.raw["advertiser_scale"]["campaign_share"]
    per_tier = np.array([cs[t]["value"] for t in TIERS], dtype=np.float64)
    t_of_camp = tier[owner]
    n_in_tier = np.array([(t_of_camp == i).sum() for i in range(len(TIERS))], dtype=np.float64)
    w = per_tier[t_of_camp] / np.maximum(n_in_tier[t_of_camp], 1.0)
    return w / w.sum()


def build_campaigns(s, seed, n_adv, n_camp):
    g = lambda sub: R.stream(seed, "campaign_pool", sub=sub)
    tier = draw_adv_tier(g("adv_tier"), n_adv, s)
    owner, _ = allocate_campaigns(tier, n_camp, s)
    return {
        "adv_tier": tier,
        "advertiser_id": make_advertiser_id(n_adv),
        "advertiser_scale": make_advertiser_scale(tier),
        "owner": owner,
        "campaign_id": np.array(["camp_%04d" % i for i in range(n_camp)], dtype=object),
        "ad_genre": draw_genre(g("C4"), n_camp, s),
        "lc1_creative_appeal": draw_creative_appeal(g("LC1"), n_camp, s),
        "lc2_game_quality": draw_game_quality(g("LC2"), n_camp, s),
        "sample_weight": campaign_weights(owner, tier, s),
    }


# ---------------------------------------------------------------------------
# 4. the rival pool (specification 2.5, the Private-Rival Market)
# ---------------------------------------------------------------------------

@law("LR1")
def draw_value_loading(rng, K, n_gaming, s):
    """T1. How much of the user's value each rival can see. Gaming rivals load
    high, non-gaming near zero, which is what makes rho's dial meaningful."""
    lo_g, hi_g = s.raw["rival_pool"]["value_loading_gaming"]["value"]
    lo_n, hi_n = s.raw["rival_pool"]["value_loading_nongaming"]["value"]
    w = np.where(np.arange(K) < n_gaming,
                 rng.uniform(lo_g, hi_g, K), rng.uniform(lo_n, hi_n, K))
    return w


@law("LR2")
def draw_retarget(rng, K):
    """T1. iid N(0, 1) over (segment, rival).

    The segment axis is the FULL 2x2 product of the A3 os domain and the A5
    device domain, so R is 4 x K under every seed and at every scale. v1 used the
    combinations a sampled pool happened to contain, which made the array's shape
    depend on the sample; specification 2.5 fixes the axis to the domains instead.
    """
    return rng.standard_normal((4, K))


@law("LR3")
def draw_pacing(rng, K, n_days, s):
    """T1. An AR(1) day series per rival: p_0 = sigma*N(0,1), then
    p_d = phi*p_{d-1} + sigma*N(0,1).

    NOT stationary, and deliberately so: the initial draw's variance is sigma^2
    against the stationary sigma^2/(1-phi^2), so the series ramps over the first
    days of the window. The register's constraint cell says so rather than
    asserting a stationarity that generated data would refute.
    """
    phi = s.raw["rival_pool"]["pacing_ar"]["value"]
    sd = s.raw["rival_pool"]["pacing_sigma"]["value"]
    p = np.empty((K, n_days), dtype=np.float64)
    p[:, 0] = sd * rng.standard_normal(K)
    for d in range(1, n_days):
        p[:, d] = phi * p[:, d - 1] + sd * rng.standard_normal(K)
    return p


@law("LR4")
def draw_exch_participation(rng, K, n_exch, s):
    """T1. Per (rival, exchange) taste. Row k = 0 forced to 1.0, so at least one
    rival is always present and LU7's maximum is never taken over an empty set."""
    lo, hi = s.raw["rival_pool"]["exch_participation"]["value"]
    pi = rng.uniform(lo, hi, (K, n_exch))
    pi[0, :] = 1.0
    return pi


@law("LR5")
def draw_dispersion(rng, K, s):
    """T1. Per-rival lognormal bid dispersion."""
    lo, hi = s.raw["rival_pool"]["log_bid_dispersion"]["value"]
    return rng.uniform(lo, hi, K)


@law("LR6")
def draw_flight(rng, K, n_days):
    """T1. The flight mask. k = 0 is always on; each k >= 1 goes dark with
    probability 0.5 for ONE contiguous block, start U{0..n_days-6} and length
    U{2..5} days."""
    f = np.ones((K, n_days), dtype=np.float64)
    for k in range(1, K):
        if rng.random() < 0.5:
            start = int(rng.integers(0, max(1, n_days - 5)))
            length = int(rng.integers(2, 6))
            f[k, start:start + length] = 0.0
    return f


def build_rivals(s, seed, n_days, n_exch):
    g = lambda sub: R.stream(seed, "rival_pool", sub=sub)
    K = s.raw["rival_pool"]["K"]["value"]
    n_gaming = s.raw["rival_pool"]["n_gaming"]["value"]
    return {
        "K": K,
        "w": draw_value_loading(g("LR1"), K, n_gaming, s),
        "R": draw_retarget(g("LR2"), K),
        "pace": draw_pacing(g("LR3"), K, n_days, s),
        "pi": draw_exch_participation(g("LR4"), K, n_exch, s),
        "sigma": draw_dispersion(g("LR5"), K, s),
        "flight": draw_flight(g("LR6"), K, n_days),
    }
