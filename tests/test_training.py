# -*- coding: utf-8 -*-
"""Stage 4: the training stack.

The gate conditions the plan can state as a test are tests here. The rest, the
AUC floors and the sweep, are measurements and live in the gate artifact, because
a threshold that a thin sample can fail is a finding to report rather than a
build to break.

What these guard is the thing a metric cannot see: that no model is reading
something it would not have at bid time.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from t9v2 import api, generate as G                              # noqa: E402
from t9v2.core.config import load                                # noqa: E402
from t9v2.train import bidders as B                              # noqa: E402
from t9v2.train import features as F                             # noqa: E402
from t9v2.train import metrics as M                              # noqa: E402
from t9v2.train.runner import train_view                         # noqa: E402

ROWS = 40_000


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def master():
    return G.frame("100K", seed=20250, n_rows=ROWS)


@pytest.fixture(scope="module")
def fitted(master, s):
    return api.train(master, "C4", s)


# ---------------------------------------------------------------------------
# leakage: what a model is allowed to see
# ---------------------------------------------------------------------------

def test_no_outcome_is_ever_a_feature(master, s):
    """Tier 1 predicts the funnel, so no funnel column may be an input."""
    d = F.prepare(api.censor(master, "C4", s), s)
    cols = F.tier1_features(d, "click")
    for banned in F.ALL_OUTCOME_COLS:
        assert banned not in cols, "%s is a Tier-1 feature" % banned


def test_settlement_columns_are_never_per_row_features(master, s):
    """winning_price and bid_density are realised AFTER the auction clears.

    A bidder cannot know either at the moment it bids. They reach the model only
    as cell-level history through the SSP encoder, which is what an SSP
    integration actually gives you.
    """
    d = F.prepare(api.censor(master, "C4", s), s)
    for cols in (F.tier1_features(d, "click"), F.tier2_features(d)):
        for banned in F.SETTLEMENT_COLS:
            assert banned not in cols, "%s used as a per-row feature" % banned


def test_won_is_never_a_tier1_feature(master, s):
    d = F.prepare(api.censor(master, "C4", s), s)
    assert "won" not in F.tier1_features(d, "click")
    assert "won" not in F.tier2_features(d)      # it is the LABEL of tier 2


def test_user_id_cannot_reach_any_model(master, s):
    for v in ("C1", "C2", "C3", "C4"):
        d = F.prepare(api.censor(master, v, s), s)
        assert "user_id" not in F.tier1_features(d, "click")


# ---------------------------------------------------------------------------
# the gate-4 conditions that ARE tests
# ---------------------------------------------------------------------------

def test_all_four_views_produce_all_five_models(master, s):
    """5 models: the 4 Tier-1 heads and the Tier-2 win classifier."""
    for v in ("C1", "C2", "C3", "C4"):
        r, f = train_view(master, v, s)
        assert set(r["head_mode"]) == {"click", "install", "payer", "spend"}
        assert all(m in ("model", "intercept") for m in r["head_mode"].values()), \
            "%s left a head unavailable: %s" % (v, r["head_mode"])
        assert f["tier2"].model is not None


def test_c4_is_trained_on_more_columns_than_c1(master, s):
    r1, _ = train_view(master, "C1", s)
    r4, _ = train_view(master, "C4", s)
    assert r4["features_tier2"] > r1["features_tier2"]
    assert r4["columns"] > r1["columns"]


def test_predict_of_one_row_matches_the_batch(master, s, fitted):
    """Gate 4: `predict` returns for one row what the vectorised batch returns.

    The only way to know a bidder lifted out of this code would behave the way
    the reported numbers say it does. A per-row path that disagrees with the
    batch is a bidder that does something different in production from what was
    measured in the study.
    """
    view = api.censor(master, "C4", s)
    ev_batch, pw_batch = fitted.score(view)
    for i in (0, 7, 100, len(view) - 1):
        ev_one, pw_one = fitted.predict(view.iloc[[i]])
        assert np.isclose(ev_one, ev_batch[i], rtol=1e-9, atol=1e-12), \
            "row %d: ev %r vs batch %r" % (i, ev_one, ev_batch[i])
        assert np.isclose(pw_one, pw_batch[i], rtol=1e-9, atol=1e-12), \
            "row %d: p(win) %r vs batch %r" % (i, pw_one, pw_batch[i])


def test_the_win_curve_is_non_decreasing_in_the_bid(master, s, fitted):
    """Open item O8. Bidding more can never win less often.

    A fact about a first-price auction, not a modelling preference, and it is
    what stops the bidder choosing a bid from a sag in a thin region.
    """
    view = api.censor(master, "C4", s).iloc[:600]
    d = fitted._prep(view)
    prices = B.ladder(s)
    curve = fitted.t2.win_curve(d, prices)
    diffs = np.diff(curve, axis=1)
    assert (diffs >= -1e-9).all(), \
        "the win curve falls as the bid rises on %d row(s)" % int((diffs < -1e-9).any(1).sum())


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def test_the_split_is_temporal_and_identical_in_every_view(master, s):
    ref = None
    for v in ("C1", "C2", "C3", "C4"):
        d = F.prepare(api.censor(master, v, s), s)
        m = F.split(d, s)
        assert not (m["train"] & m["test"]).any()
        assert not (m["train"] & m["valid"]).any()
        key = tuple(int(m[k].sum()) for k in ("train", "valid", "test"))
        ref = ref or key
        assert key == ref, "view %s splits differently: %s vs %s" % (v, key, ref)


def test_no_test_row_is_in_training(master, s):
    d = F.prepare(api.censor(master, "C4", s), s)
    m = F.split(d, s)
    day = F.day_of_run(d, s)
    assert day[m["train"]].max() < day[m["test"]].min(), \
        "a training day is at or after a test day"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_auc_matches_a_known_case():
    assert M.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert M.auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert M.auc([0, 1], [0.5, 0.5]) == 0.5           # a tie is a coin flip


def test_auc_handles_ties_without_bias():
    y = np.r_[np.zeros(50), np.ones(50)]
    p = np.full(100, 0.3)
    assert M.auc(y, p) == 0.5


def test_calibration_is_zero_for_a_perfect_forecaster():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    ece, mce = M.calibration(y, p, bins=10)
    assert ece < 0.01 and mce < 0.03


def test_crps_falls_as_the_forecast_sharpens_on_the_truth():
    y = np.exp(np.random.default_rng(1).normal(1.0, 0.5, 4000))
    good = M.crps_lognormal(y, np.full(len(y), 1.0), 0.5)
    wide = M.crps_lognormal(y, np.full(len(y), 1.0), 2.0)
    off = M.crps_lognormal(y, np.full(len(y), 3.0), 0.5)
    assert good < wide and good < off


def test_economics_pays_its_own_bid_on_won_rows():
    """First price: the winner pays what it bid, so profit is value minus bid."""
    bid = np.array([2.0, 5.0, 1.0])
    hurdle = np.array([1.0, 6.0, 0.5])
    value = np.array([10.0, 10.0, 10.0])
    e = M.economics(bid, None, value, hurdle, np.zeros(3))
    assert e["wins"] == 2
    assert e["spend"] == 3.0                      # 2.0 + 1.0, not the hurdles
    assert e["profit"] == 20.0 - 3.0


# ---------------------------------------------------------------------------
# the tilt sweep's null
# ---------------------------------------------------------------------------

def test_tau_zero_reproduces_the_untilted_generator_bit_for_bit():
    """The precondition of the whole sensitivity sweep.

    Compared against a generator built by a DIFFERENT route, straight from the
    marginal with no ladder and no rake. Comparing tau = 0 against itself would
    prove nothing.
    """
    a = G.frame("100K", seed=20250, n_rows=12000, tau_scale=0.0)
    b = G.frame("100K", seed=20250, n_rows=12000, untilted=True)
    assert a.equals(b)


def test_a_nonzero_tau_actually_changes_the_data():
    b = G.frame("100K", seed=20250, n_rows=12000, untilted=True)
    c = G.frame("100K", seed=20250, n_rows=12000, tau_scale=1.0)
    assert not c["os"].equals(b["os"])
