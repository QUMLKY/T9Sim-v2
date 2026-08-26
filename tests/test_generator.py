# -*- coding: utf-8 -*-
"""Generator invariants. Stage 2's own tests, per the rebuild plan.

These are not calibration tests. Whether the CTR lands in its band is the
validator's job and the solver's; these check the things that must hold at ANY
calibration, because they are properties of the structure rather than of the
numbers. A calibration drifts and gets re-solved; one of these failing is a
defect in the generator.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from t9v2 import generate as G
from t9v2 import pools as P
from t9v2.core import config, rng as R
from t9v2.gen import auction as A
from t9v2.gen import funnel as F
from t9v2.gen import rival_market as M
from t9v2.network import laws as L

ROWS = 12_000


@pytest.fixture(scope="module")
def s():
    return config.load("default")


@pytest.fixture(scope="module")
def df():
    return G.frame("100K", seed=20250, n_rows=ROWS)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_v3_every_node_has_a_law(s):
    """The check stage 1 could not run: every node names a function that exists."""
    assert L.v3_laws_exist(s)


def test_law_count_matches_the_node_count(s):
    """79 since H5, and the law must exist as well as the register row."""
    assert len(L.LAWS) == len(s.nodes) == 79


def test_no_funnel_law_reads_won():
    """The correction the whole ablation turns on.

    Outcomes are drawn on EVERY row, won or lost. Checked structurally rather
    than statistically: no function in the funnel module takes `won` as an
    argument, so it cannot condition on it however the numbers come out.
    """
    for name, fn in inspect.getmembers(F, inspect.isfunction):
        if getattr(fn, "node_id", None) is None:
            continue
        args = set(inspect.signature(fn).parameters)
        assert not (args & {"won", "won_v", "h3"}), \
            "funnel law %s takes a win argument; the funnel must not read won" % name


def test_our_bid_reads_no_latent_and_no_competing_bid():
    """H2's arguments are observable quantities only.

    If the bidder could see LU7 or the oracle, C1 would already know what only
    C3 is meant to be able to learn, and the ablation would measure nothing.
    """
    args = set(inspect.signature(M.our_bid).parameters)
    for banned in ("lu7", "competing", "ev", "ev_truth", "z", "e_ltv"):
        assert banned not in args, "our_bid takes %r" % banned


def test_columns_are_exactly_the_frozen_56(s, df):
    assert list(df.columns) == s.raw["column_order"]
    assert len(df.columns) == 56


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_bit_for_bit():
    a = G.frame("100K", seed=20250, n_rows=3000)
    b = G.frame("100K", seed=20250, n_rows=3000)
    assert a.equals(b)


def test_different_seed_gives_different_data():
    a = G.frame("100K", seed=20250, n_rows=3000)
    b = G.frame("100K", seed=20251, n_rows=3000)
    assert not a["ev_truth"].equals(b["ev_truth"])


def test_streams_are_independent_of_each_other():
    """Two names must not correlate, however close their text.

    This is the property the whole named-stream design exists for: adding a node
    to one stream must not disturb another.
    """
    a = R.stream(7, "funnel").standard_normal(4000)
    b = R.stream(7, "funne1").standard_normal(4000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


def test_stream_name_is_required():
    with pytest.raises(ValueError):
        R.stream(1, "")


def test_blocks_partition_the_rows_exactly():
    got = R.block_bounds(1000, 256)
    assert [x[:2] for x in got] == [(0, 256), (256, 512), (512, 768), (768, 1000)]
    assert sum(b - a for a, b, _ in got) == 1000


# ---------------------------------------------------------------------------
# the exact identities, on every row
# ---------------------------------------------------------------------------

def test_gate_identities_hold_exactly(df):
    assert ((df["click"] == 0) & (df["install"] == 1)).sum() == 0
    assert ((df["install"] == 0) & (df["is_payer"] == 1)).sum() == 0
    assert (df.loc[df["is_payer"] == 0, "ltv_value"] != 0).sum() == 0
    assert ((df["click"] == 0) == (df["click_timestamp"] == -1)).all()
    assert ((df["install"] == 0) == (df["install_timestamp"] == -1)).all()


def test_inactive_users_have_exactly_zero_expected_value(df):
    d = df[df["lu1_archetype"] == "inactive"]
    assert len(d) > 0
    assert (d["ev_truth"] == 0).all()
    assert (d["lu4_payer_prob"] == 0).all()
    assert (d["lu5_ltv_mult"] == 0).all()


def test_first_price_means_the_winner_pays_its_own_bid(df):
    w = df[df["won"] == 1]
    assert np.allclose(w["winning_price"], w["bid_price"])


def test_unsold_rows_have_no_winning_price(df):
    unsold = (df["won"] == 0) & (df["lu7_competing_bid"] < df["floor_price"])
    assert (df["winning_price"].isna() == unsold).all()


def test_both_price_shapes_are_normalised_by_the_paying_median():
    """The floor fix. ONE denominator, and the medians say which one.

    The floor and the paying price are two prices in one market, and the ratio
    between them is the market's structure. Dividing each by its own median
    forces both to 1 and erases it. Measured on the pmfs, the floor's weighted
    median is 40 against the paying median's 70, so the correct floor shape has
    median 40/70 exactly; the bug made it 1 and inflated every floor by 1.750.

    ASSERTED ON THE MEDIANS, NOT ON A REALISED RATIO. The fix commit on the
    abandoned branch reported 0.5664, which is the MEAN ratio over drawn rows
    and moves with the seed. These two numbers are properties of the pmfs, so
    exact equality is the right test where a drifting one would not be.
    """
    (fv, fp), (pv, pp) = M.price_shapes()
    assert M._weighted_median(pv, pp) == 1.0
    assert M._weighted_median(fv, fp) == 40.0 / 70.0


def test_the_floor_zero_atom_keeps_its_measured_mass():
    """14.15 percent, and NOT the 57 percent the documents carried.

    Both shapes are resampled as empirical pmfs rather than fitted, so the atom
    survives normalisation whatever the denominator is. 57 percent was the shape
    ratio 40/70 misread as a share; the Node Register corrected it on 16 August
    2026 and the Specification did not follow until the floor fix.
    """
    (fv, fp), _ = M.price_shapes()
    assert abs(float(fp[fv == 0].sum()) - 0.1415) < 0.0005


def test_ev_truth_is_the_product_of_its_four_parents(df):
    assert np.allclose(df["ev_truth"],
                       df["p_click"] * df["p_install"] * df["p_payer"] * df["e_ltv"])


def test_ltv_horizons_are_fixed_fractions(df):
    assert np.allclose(df["ltv_7d"], 0.40 * df["ltv_value"])
    assert np.allclose(df["ltv_30d"], 0.70 * df["ltv_value"])


def test_probabilities_stay_in_range(df):
    for c in ("p_click", "p_install", "p_payer"):
        assert df[c].between(0.0, 1.0).all()


def test_at_least_one_rival_always_participates(df):
    assert df["bid_density"].between(1, 8).all()
    assert np.isfinite(df["lu7_competing_bid"]).all()


def test_the_funnel_fires_on_lost_rows_too(df):
    """The censoring correction, measured rather than assumed."""
    lost = df[df["won"] == 0]
    assert len(lost) > 0
    assert lost["click"].sum() > 0, "no lost row produced a click"


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------

def test_rake_hits_its_target_marginal():
    cond = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    w = np.array([0.2, 0.3, 0.5])
    target = np.array([0.59, 0.41])
    raked = L.ipf_conditional(cond, w, target)
    assert np.allclose(raked.sum(axis=1), 1.0)
    assert np.allclose((raked * w[:, None]).sum(axis=0), target, atol=1e-6)


def test_tilt_of_zero_reproduces_a_parentless_draw_exactly():
    """The precondition of the O3 3b sensitivity sweep.

    At tau = 0 every archetype must get exactly the marginal, so the tau = 0 arm
    of the stage-4 sweep reproduces a generator with the tilt tables removed. If
    this is approximate rather than exact, the sweep has no true null.
    """
    ladder = {a: v for a, v in zip(P.ARCHETYPES, [2.9, 1.9, 0.9, -0.1, -1.1])}
    p = L.binary_tilt(0.59, ladder, P.ARCHETYPES, tau=0.0)
    assert np.allclose(p, 0.59, atol=1e-12)


def test_the_tilt_orders_archetypes_by_the_ladder(s):
    tables = P.build_tables(s)
    ios = [tables["A3"][a]["iOS"] for a in P.ARCHETYPES]
    assert all(ios[i] > ios[i + 1] for i in range(len(ios) - 1)), ios


def test_hour_table_preserves_the_population_hour_marginal(s):
    tables = P.build_tables(s)
    shares = np.array([s.raw["archetype_shares"][a]["value"] for a in P.ARCHETYPES])
    got = (tables["D2"] * shares[:, None]).sum(axis=0)
    base = s.raw["tables"]["D2_hour_given_archetype"]["base_hour_marginal"]
    want = np.array([base[h] for h in range(24)])
    assert np.allclose(got, want / want.sum(), atol=1e-6)


def test_pair_table_preserves_both_margins(s):
    run_users = P.build_users(s, 20250, 4000, P.build_tables(s))
    apps = P.build_apps(s, 20250, 40)
    t = A.build_pair_table(s, run_users, apps)
    pi = np.array([s.raw["archetype_shares"][a]["value"] for a in P.ARCHETYPES])
    assert np.allclose(t.sum(axis=1), pi / pi.sum(), atol=1e-6)
    assert np.allclose(t.sum(axis=0), 1.0 / 40, atol=1e-6)


def test_t_pay_has_population_mean_one(s):
    tables = P.build_tables(s)
    ph, pd = tables["hour_marginal"], tables["dow_marginal"]
    mean = float((tables["t_pay"] * (ph[:, None] * pd[None, :])).sum())
    assert abs(mean - 1.0) < 1e-9


def test_value_bins_put_every_zero_value_user_in_the_bottom_bin(s):
    users = P.build_users(s, 20250, 6000, P.build_tables(s))
    b, _ = A.user_value_bins(users)
    zero = (users["lu4_payer_prob"] * users["lu5_ltv_mult"]) == 0
    assert (b[zero] == 0).all()
    assert b.max() == 9
