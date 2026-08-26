# -*- coding: utf-8 -*-
"""The price head: one estimand, four views, one likelihood, and one sigma.

Step 6. Every view predicts `m^win = max(LU7, floor)`. What differs is how well
each can OBSERVE it — exact in C3 and C4, an interval in C1 and C2 — which is the
ablation working rather than a methodological difference smuggled in beside it.

THE CENTRAL TEST IS THE BRACKET. `mwp_bounds` builds the censoring interval on
the fly rather than storing it, because storing it would take two columns and
`mwp_upper` is finite exactly when `won == 1`, making it the Tier-2 label in
disguise. So the property is checked instead of the result: `lower <= m^win <=
upper` on EVERY row of EVERY view, against the master where LU7 and the floor are
both known. A column could not be checked this way — it would simply hold
whatever the same arithmetic produced.

THE CLOSED LOWER BOUND IS TESTED SEPARATELY because it is wrong on 11.7 percent
of rows if written open. Losing means `bid < m^win`, an open bound at the bid.
But on an unsold row `LU7 < floor`, so `m^win` IS the floor exactly, and where
the bid sits under the floor an open bound would exclude the true value.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from t9v2 import censor as CEN
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.train import encoder as E
from t9v2.train import features as F
from t9v2.train import price as PR
from t9v2.train.runner import run_seed, sigma_order

warnings.filterwarnings("ignore")

VIEWS = ["C1", "C2", "C3", "C4"]


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    p = tmp_path_factory.mktemp("price") / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def fitted(master, s):
    return run_seed(master, s, seed=0, quiet=True)


# --------------------------------------------------------------- the interval

@pytest.mark.parametrize("view", VIEWS)
def test_the_interval_brackets_the_truth_on_every_row(master, s, view):
    """The property, checked rather than the arithmetic re-run.

    `m^win` is known from the master in every view even where the view cannot
    see it, so the bracket can be verified everywhere. Any slip in `mwp_bounds`
    fails here immediately.
    """
    d = F.prepare(CEN.censor(master, view, s), s)
    lower, upper = PR.mwp_bounds(d)
    truth = np.maximum(master["lu7_competing_bid"].to_numpy(dtype=float),
                       master["floor_price"].to_numpy(dtype=float))
    assert len(lower) == len(truth)
    assert (lower <= truth + 1e-12).all(), \
        "%s: lower bound above the truth on %d rows" % (view, int((lower > truth).sum()))
    assert (truth <= upper + 1e-12).all(), \
        "%s: truth above the upper bound on %d rows" % (view, int((truth > upper).sum()))


def test_the_lower_bound_on_lost_rows_is_closed_and_binds(master, s):
    """Written OPEN it would exclude the true value on every unsold row.

    On an unsold row `LU7 < floor`, so `m^win` equals the floor exactly. Where
    the bid is below the floor the binding constraint is the floor, ATTAINED
    rather than exceeded. XGBoost's `label_lower_bound` is closed by convention,
    so `max(bid, floor)` is the right arithmetic.
    """
    d = F.prepare(CEN.censor(master, "C1", s), s)
    lower, upper = PR.mwp_bounds(d)
    truth = np.maximum(master["lu7_competing_bid"].to_numpy(dtype=float),
                       master["floor_price"].to_numpy(dtype=float))
    lost = master["won"].to_numpy() == 0
    at_bound = lost & np.isclose(lower, truth)
    assert at_bound.sum() > 0, "no lost row attains its lower bound; is this data right?"
    share = at_bound.mean()
    assert 0.05 < share < 0.25, (
        "%.1f%% of rows sit exactly ON the closed lower bound. An open bound "
        "would exclude the truth on every one of them." % (100 * share))
    assert (upper[lost] == np.inf).all(), "a lost row is right-unbounded"


def test_won_rows_are_bounded_both_sides(master, s):
    d = F.prepare(CEN.censor(master, "C1", s), s)
    lower, upper = PR.mwp_bounds(d)
    won = master["won"].to_numpy() == 1
    assert np.isfinite(upper[won]).all()
    assert (lower[won] <= upper[won]).all()
    assert np.allclose(upper[won], master.loc[won, "bid_price"].to_numpy())
    assert np.allclose(lower[won], master.loc[won, "floor_price"].to_numpy())


def test_the_interval_is_never_materialised_as_a_column(master, s):
    """Storing it would take two columns, and one of them IS the win label.

    `mwp_upper` is finite exactly when `won == 1`, so a stored upper bound would
    be `won` in disguise and would itself need banning from every feature list.
    """
    for v in VIEWS:
        cols = set(F.prepare(CEN.censor(master, v, s), s).columns)
        assert not (cols & {"mwp_lower", "mwp_upper"}), v
    d = F.prepare(CEN.censor(master, "C1", s), s)
    _, upper = PR.mwp_bounds(d)
    assert np.array_equal(np.isfinite(upper),
                          master["won"].to_numpy() == 1), \
        "the upper bound IS the win label; this is why it is not stored"


# ---------------------------------------------------------------- the features

def test_the_feature_list_is_identical_in_all_four_views(fitted):
    """What makes this step's contrast readable at all.

    The classifier varies features and holds the target fixed; the price head
    varies the label and holds features fixed. Each holds the other axis
    constant, which is what separates the two mechanisms rather than confounding
    them.
    """
    lists = {v: fitted[v]["price_cols"] for v in VIEWS}
    assert lists["C1"] == lists["C2"] == lists["C3"] == lists["C4"]
    assert len(set(map(len, lists.values()))) == 1


def test_neither_the_bid_nor_any_ssp_encoder_is_a_price_feature(fitted):
    for v in VIEWS:
        cols = set(fitted[v]["price_cols"])
        assert "bid_price" not in cols, "%s: the censoring boundary is a feature" % v
        assert not (cols & set(PR.SSP_ENCODERS)), v
        assert "_enc_dsp_price" in cols, "%s: lost the one price signal all views have" % v


def test_features_asserts_its_own_output_not_its_input(master, s):
    """The raise is a backstop on the selection, not a rule at the call site.

    A caller hands over every encoder its view has; `features` selects. Rejecting
    the caller would put the rule at each call site, which is where rules drift.
    """
    d = F.prepare(CEN.censor(master, "C3", s), s)
    enc = E.build(d[F.split(d, s)["train"]].copy(), "C3", s)
    all_cols = E.apply(enc, d)
    assert any(c in PR.SSP_ENCODERS for c in all_cols), "fixture must include SSP encoders"
    out = PR.features(d, all_cols)                    # must NOT raise
    assert not (set(out) & set(PR.SSP_ENCODERS))
    with pytest.raises(RuntimeError, match="_enc_dsp_price"):
        PR.features(d, [])


# ------------------------------------------------------------------ the sigma

def test_c3_is_fitted_first_so_sigma_exists_before_c1_needs_it():
    assert sigma_order(["C1", "C2", "C3", "C4"]) == ["C3", "C1", "C2", "C4"]
    assert sigma_order(["C1", "C2"]) == ["C1", "C2"], "no C3, no reorder"
    assert set(sigma_order(VIEWS)) == set(VIEWS), "reordering must not drop a view"


def test_the_result_comes_back_in_view_order_whatever_the_fit_order(fitted):
    assert list(fitted) == VIEWS


def test_sigma_is_measured_on_c3_and_handed_to_the_censored_views(fitted):
    """The replacement for v1's AFT_SCALE = 1.0, which was copied from nowhere.

    C3 and C4 take the placeholder because a normal AFT with exact labels has a
    location MLE independent of the scale, so it cannot touch their numbers. C1
    and C2 take the value read off C3's residuals, because their labels are
    intervals and the scale shapes the likelihood.
    """
    c3 = fitted["C3"]["heads"]["price"]
    assert "sigma_hat" in c3 and 0.2 <= c3["sigma_hat"] <= 2.0
    assert fitted["C3"]["heads"]["price"]["sigma"] == 1.0, "the placeholder, and inert"
    assert fitted["C4"]["heads"]["price"]["sigma"] == 1.0
    for v in ("C1", "C2"):
        assert fitted[v]["heads"]["price"]["sigma"] == c3["sigma_hat"], v
    assert not fitted["C1"]["heads"]["price"]["exact_labels"]
    assert fitted["C3"]["heads"]["price"]["exact_labels"]


def test_a_censored_view_refuses_to_train_without_a_sigma(master, s):
    """No default. v1's constant sat on the headline contrast unremarked."""
    from t9v2.train.runner import train_view
    with pytest.raises(RuntimeError, match="INTERVALS"):
        train_view(master, "C1", s, seed=0, sigma=None)


def test_sigma_refuses_rather_than_falling_back_on_too_few_rows(s):
    with pytest.raises(RuntimeError, match="exact labels"):
        PR.sigma_from_residuals(np.array([1.0, 2.0]), np.array([1.0, 2.0]), s)


def test_sigma_is_clamped_and_says_so(s):
    lo = PR.sigma_from_residuals(np.ones(500), np.ones(500), s)
    assert lo[0] == 0.2 and lo[1] < 0.2, "a zero-spread fit must clamp up"
    rng = np.random.default_rng(0)
    wild = np.exp(rng.normal(0, 6, 500))
    hi = PR.sigma_from_residuals(wild, np.ones(500), s)
    assert hi[0] == 2.0 and hi[1] > 2.0, "a wild fit must clamp down"


# ------------------------------------------------------------- what it measures

def test_the_censored_views_predict_worse_than_the_exact_ones(fitted):
    """The ablation's answer for this head, and the reason it exists.

    Same estimand, same features, same likelihood class. The ONLY difference is
    label quality, so any gap here is label quality and nothing else.
    """
    c1 = fitted["C1"]["heads"]["price"]["rmse_log"]
    c3 = fitted["C3"]["heads"]["price"]["rmse_log"]
    assert c3 < c1, "exact labels must beat intervals: C3 %.4f, C1 %.4f" % (c3, c1)


def test_the_censored_views_over_predict_and_the_exact_ones_do_not(fitted):
    """Right-censoring pushes the fit UP, and the sign records which way.

    About 70 percent of rows are lost, and a lost row's interval says "at least
    this much" with no upper bound, so nothing in the likelihood pulls those rows
    down. Measured: C1 and C2 at +0.4167, C3 and C4 at -0.1076.
    """
    assert fitted["C1"]["heads"]["price"]["bias_log"] > 0.1
    assert fitted["C3"]["heads"]["price"]["bias_log"] < 0.1


def test_mmp_cannot_move_the_price_head(fitted):
    """C1 and C2 must be identical here, and C3 and C4 too.

    MMP adds funnel-label ROWS, not columns, and the price head uses neither
    funnel labels nor anything MMP touches. So the price head has exactly ONE
    contrast, C3 minus C1, which is the SSP label-quality question and nothing
    else. If this ever fails, something has leaked funnel visibility into it.
    """
    for a, b in [("C1", "C2"), ("C3", "C4")]:
        pa, pb = fitted[a]["heads"]["price"], fitted[b]["heads"]["price"]
        assert pa["rmse_log"] == pb["rmse_log"], (a, b)
        assert pa["bias_log"] == pb["bias_log"], (a, b)


# --------------------------------------------------------- through the bundle

def test_the_price_head_survives_the_bundle_round_trip(master, s, tmp_path):
    """It has to, or the eval file cannot re-score it and step 7 cannot bid on it.

    The price head is a raw Booster rather than a sklearn wrapper, because
    interval-censored AFT needs `label_lower_bound` on a DMatrix and the wrapper
    has no path to them — so its save and load are their own code, and their own
    chance to be wrong.
    """
    from t9v2 import bundle as BU
    run_seed(master, s, seed=0, quiet=True, bundle_dir=tmp_path / "b")
    for v in ("C1", "C3"):
        pieces = BU.load_bundle(tmp_path / "b" / v)
        ph = pieces["price"]
        assert ph is not None and ph.cols == pieces["price_cols"]
        assert ph.exact == (v == "C3")

        d = F.prepare(CEN.censor(master, v, s), s)
        E.apply(pieces["encoders"], d)
        d_te = d[F.split(d, s)["test"]]
        assert np.all(np.isfinite(ph.predict(d_te)))


def test_the_bundle_keeps_the_sigma_that_shaped_the_fit(master, s, tmp_path):
    """For C1 sigma IS part of what produced these trees, so losing it would mean
    the bundle could not say what it had fitted."""
    from t9v2 import bundle as BU
    out = run_seed(master, s, seed=0, quiet=True, bundle_dir=tmp_path / "b")
    for v in ("C1", "C2", "C3", "C4"):
        pieces = BU.load_bundle(tmp_path / "b" / v)
        assert pieces["price"].sigma == out[v]["heads"]["price"]["sigma"], v


def test_the_eval_file_carries_the_price_prediction_and_the_truth(master, s, tmp_path):
    """So a new price metric is a re-score of numbers on disk, not a refit."""
    from t9v2.evalfile import write_eval
    run_seed(master, s, seed=0, quiet=True, bundle_dir=tmp_path / "b")
    write_eval(master, tmp_path / "b", tmp_path / "e", s, quiet=True)
    for v in VIEWS:
        d = pd.read_parquet(tmp_path / "e" / ("%s.parquet" % v))
        assert {"m_win_pred", "m_win_true"} <= set(d.columns), v
        assert (d["m_win_true"] > 0).all()
        assert np.isfinite(d["m_win_pred"]).all()
    # and the file reproduces the reported metric, which is the point of keeping it
    from t9v2.train import price as PR2
    out = run_seed(master, s, seed=0, quiet=True)
    d = pd.read_parquet(tmp_path / "e" / "C3.parquet")
    again = PR2.score(d["m_win_true"].to_numpy(), d["m_win_pred"].to_numpy())
    assert abs(again["rmse_log"] - out["C3"]["heads"]["price"]["rmse_log"]) < 1e-9
