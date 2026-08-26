# -*- coding: utf-8 -*-
"""The law registry, and the table machinery the tilted CPTs are built from.

Two jobs.

FIRST, the registry that check V3 reads. Every node in graph.yaml must name a
function that exists, and `@law("LU1")` is how a function claims its node. The
implementations live next to each other in the `gen` modules rather than here,
because a law reads best beside the laws it is drawn with; this module only holds
the mapping and the check.

SECOND, the tables that tables.yaml describes rather than stores. Four of the
nine are recipes, not values: the three tilted CPTs wired by open item O1, and
the hour CPT. Building them at load rather than storing a cell grid is what stops
the dial and the grid drifting apart, and it is what makes tau = 0 reproduce a
parentless draw EXACTLY rather than approximately, which the O3 3b decision
depends on and the stage-4 sweep tests.

The rake is the load-bearing part. Wiring a parent into a draw moves that draw's
marginal, and the marginals here are calibrated quantities. So every tilted table
is raked back onto the marginal the design already fixed: the tilt changes who
gets what, never how much of it there is in total.
"""
from __future__ import annotations

import numpy as np

LAWS = {}


def law(node_id):
    """Register a function as the law of one node. V3 checks the mapping."""
    def wrap(fn):
        if node_id in LAWS:
            raise RuntimeError("two functions claim the law of node %r: %s and %s"
                               % (node_id, LAWS[node_id].__name__, fn.__name__))
        LAWS[node_id] = fn
        fn.node_id = node_id
        return fn
    return wrap


def _load_law_modules():
    """Import every module that registers a law.

    A decorator only runs when its module is imported, so before 16 August 2026
    this check reported all 79 nodes missing when called from a tool that had not
    happened to import the generator. A check whose verdict depends on the
    caller's import order is worse than no check, because it fails loudly in the
    safe direction and passes silently in the dangerous one. Imported here rather
    than at module scope because those modules import this one.
    """
    from .. import pools                                   # noqa: F401
    from ..gen import auction, funnel, rival_market        # noqa: F401


def v3_laws_exist(settings):
    """V3. Every node names a function that exists.

    Deferred out of stage 1 because it cannot run until the functions do. The
    other ten checks read settings alone; this one reads the code.
    """
    _load_law_modules()
    missing = [n["id"] for n in settings.nodes if n["id"] not in LAWS]
    if missing:
        raise RuntimeError(
            "V3 failed. %d node(s) in graph.yaml have no law function registered "
            "in t9v2.network.laws: %s" % (len(missing), ", ".join(missing)))
    extra = [k for k in LAWS if k not in settings.by_id]
    if extra:
        raise RuntimeError(
            "V3 failed. %d law function(s) claim a node graph.yaml does not "
            "declare: %s" % (len(extra), ", ".join(sorted(extra))))
    return True


# ---------------------------------------------------------------------------
# the rake
# ---------------------------------------------------------------------------

def ipf_conditional(cond, parent_w, target, tol=1e-6, iterations=500):
    """Rake a conditional table so its parent-weighted marginal hits `target`.

    cond      (A, C) array, row a is P(child | parent = a), rows sum to 1
    parent_w  (A,) the parent's own distribution, summing to 1
    target    (C,) the marginal that must survive, summing to 1

    Returns a conditional of the same shape. The procedure is plain IPF on the
    joint J[a, c] = parent_w[a] * cond[a, c]: scale columns onto the target,
    renormalise rows back onto parent_w, repeat. Both constraints are satisfied
    at convergence, so the marginal is exact and each row is still a pmf.

    Convergence is not assumed. It is measured and raised on, because a table
    that quietly fails to converge produces a dataset whose calibration is off by
    an amount no gate would attribute to the rake.
    """
    cond = np.asarray(cond, dtype=np.float64)
    parent_w = np.asarray(parent_w, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    parent_w = parent_w / parent_w.sum()
    target = target / target.sum()

    joint = cond * parent_w[:, None]
    for _ in range(iterations):
        col = joint.sum(axis=0)
        # a column with no mass cannot be scaled onto a positive target, and
        # silently leaving it at zero would miss the marginal by that much
        if np.any((col <= 0) & (target > 0)):
            bad = np.where((col <= 0) & (target > 0))[0]
            raise ValueError("rake cannot reach the target: column(s) %s carry no "
                             "mass but the target asks for some" % bad.tolist())
        joint *= np.where(col > 0, target / np.where(col > 0, col, 1.0), 0.0)
        joint *= (parent_w / joint.sum(axis=1))[:, None]
        gap = np.abs(joint.sum(axis=0) - target).max()
        if gap <= tol:
            break
    else:
        raise ValueError("rake did not converge in %d iterations; worst marginal "
                         "gap %.3g against tolerance %g" % (iterations, gap, tol))
    return joint / parent_w[:, None]


def ipf_joint(seed, row_target, col_target, tol=1e-9, iterations=500):
    """Rake a joint onto both margins. Used by the pairing table (edge #1).

    seed is the untilted-plus-tilt starting grid; the two targets are the
    archetype mix and uniform app popularity. Preserving BOTH is what makes the
    pairing edge change who meets whom without changing how often anyone appears,
    which is the property the register states and P1 relies on.
    """
    j = np.asarray(seed, dtype=np.float64).copy()
    r = np.asarray(row_target, dtype=np.float64); r = r / r.sum()
    c = np.asarray(col_target, dtype=np.float64); c = c / c.sum()
    for _ in range(iterations):
        j *= (r / j.sum(axis=1))[:, None]
        j *= np.where(j.sum(axis=0) > 0, c / np.where(j.sum(axis=0) > 0, j.sum(axis=0), 1.0), 0.0)
        if max(np.abs(j.sum(axis=1) - r).max(), np.abs(j.sum(axis=0) - c).max()) <= tol:
            break
    return j / j.sum()


# ---------------------------------------------------------------------------
# the three tilted tables (open item O1, strengths O3 3b)
# ---------------------------------------------------------------------------

def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def binary_tilt(marginal_p, ladder, order, tau):
    """P(level | archetype) = logistic(logit(marginal) + tau * s_a).

    Returns the per-archetype probability of the tilted level, in `order`.
    At tau = 0 every archetype gets exactly `marginal_p`, so the table collapses
    to a parentless draw and the rake that follows is the identity. That exact
    collapse is why the form was chosen over a hand-written cell grid.
    """
    s = np.array([ladder[a] for a in order], dtype=np.float64)
    return _logistic(_logit(float(marginal_p)) + float(tau) * s)


def build_binary_cpt(marginal, ladder, order, tau, levels, tilt_level, shares, tol=1e-6):
    """A raked 2-column CPT over `levels`, tilted on `tilt_level` by the ladder.

    A3 (os given archetype) and A5 (device given archetype) are both this shape.
    The rake puts the population marginal back exactly where `marginal` says.
    """
    other = [x for x in levels if x != tilt_level][0]
    p = binary_tilt(marginal[tilt_level], ladder, order, tau)
    cond = np.stack([p, 1.0 - p], axis=1)                       # [tilt_level, other]
    w = np.array([shares[a] for a in order], dtype=np.float64)
    target = np.array([marginal[tilt_level], marginal[other]], dtype=np.float64)
    raked = ipf_conditional(cond, w, target, tol=tol)
    return {a: {tilt_level: float(raked[i, 0]), other: float(raked[i, 1])}
            for i, a in enumerate(order)}


def build_dow_cpt(base_dow, ladder, order, tau, shares, weekend=(5, 6), tol=1e-6):
    """P(day of week | archetype), tilted on the WEEKEND BLOCK, then raked.

    The tilt has one dial and moves the weekend as a whole; inside the weekend,
    and inside the week, the base shape's relative proportions are untouched. So
    a stronger tau moves Saturday and Sunday together and never invents a
    Wednesday effect the design never claimed.

    Day 0 is Monday, fixed by market.yaml time.window_start_utc being a Monday.
    """
    base = np.asarray([base_dow[d] for d in range(7)], dtype=np.float64)
    base = base / base.sum()
    is_we = np.zeros(7, dtype=bool)
    is_we[list(weekend)] = True

    p_we = binary_tilt(base[is_we].sum(), ladder, order, tau)
    within_we = base[is_we] / base[is_we].sum()
    within_wk = base[~is_we] / base[~is_we].sum()

    cond = np.empty((len(order), 7), dtype=np.float64)
    cond[:, is_we] = p_we[:, None] * within_we[None, :]
    cond[:, ~is_we] = (1.0 - p_we)[:, None] * within_wk[None, :]

    w = np.array([shares[a] for a in order], dtype=np.float64)
    return ipf_conditional(cond, w, base, tol=tol)


def build_hour_cpt(base_hour, mu_kappa, order, shares, tol=1e-6):
    """P(hour | archetype), a circular von Mises tilt on the iPinYou hour shape.

    Each archetype has a peak hour mu and a concentration kappa, kappa = 0 being
    flat, so the inactive archetype reproduces the base shape exactly. The rake
    then restores the population hour marginal, which is constraint R1: archetypes
    differ in WHEN they appear without changing the overall traffic curve.
    """
    h = np.arange(24, dtype=np.float64)
    base = np.asarray([base_hour[i] for i in range(24)], dtype=np.float64)
    base = base / base.sum()

    cond = np.empty((len(order), 24), dtype=np.float64)
    for i, a in enumerate(order):
        mu, kappa = mu_kappa[a]
        tilt = np.exp(float(kappa) * np.cos(2.0 * np.pi * (h - float(mu)) / 24.0))
        row = base * tilt
        cond[i] = row / row.sum()

    w = np.array([shares[a] for a in order], dtype=np.float64)
    return ipf_conditional(cond, w, base, tol=tol)


def build_t_pay(hour_mult, dow_mult, hour_marginal, dow_marginal):
    """The payer-timing multiplier t_pay(hour, dow), raked to population mean 1.

    Dependency #4. Without the rake, multiplying p_payer by a table whose mean is
    not 1 would move the install-to-payer rate, and that rate is a calibrated
    quantity with its own band. Raking makes the edge redistribute WHEN payers
    convert while leaving HOW MANY convert to the funnel constant.

    The expectation is taken under the product of the two population marginals.
    Hour and day are conditionally independent given archetype in this generator,
    so the product is the population joint only up to the archetype coupling; the
    residual is order 1e-3 and the win-rate and funnel gates measure what actually
    lands, not what this predicts.
    """
    hm = np.asarray(hour_mult, dtype=np.float64)
    dm = np.asarray(dow_mult, dtype=np.float64)
    ph = np.asarray([hour_marginal[i] for i in range(24)], dtype=np.float64)
    ph = ph / ph.sum()
    pd = np.asarray([dow_marginal[i] for i in range(7)], dtype=np.float64)
    pd = pd / pd.sum()
    table = hm[:, None] * dm[None, :]                        # (24, 7)
    mean = float((table * (ph[:, None] * pd[None, :])).sum())
    return table / mean
