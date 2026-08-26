# -*- coding: utf-8 -*-
"""Leave-one-out on the price encoders, and the leak it is the only defence against.

`won = 1[bid_price >= min_winning_price]` is an identity. H5 is barred from the
feature lists, so the raw column cannot reach a model — but `_enc_ssp_minwin_price`
is a cell MEAN of that column, and a cell mean is legitimate history right up
until the row being scored is inside it. `build()` fits on the training split and
`apply()` runs over the whole frame, so without a correction every training row
reads a mean containing itself.

WHAT IT WOULD COST, if the correction were absent. A row's own weight in its cell
is `1 / (count + k)`, and against `k = 20` the median cell puts it near four
percent. Four percent of an ordinary feature is nothing; four percent of the
label is a manufactured SSP win-AUC lift, which is the exact artifact v2's
rebuild was undertaken to remove.

THE ARITHMETIC IS CHECKED AGAINST A HAND CALCULATION, not against itself. A test
that recomputes leave-one-out the same way the code does proves only that the
code is self-consistent. The tests below build a frame whose cells are small
enough to work out on paper.

AND THE EXCLUSION IS SHOWN TO BE LOAD-BEARING. The last test force-adds the raw
`min_winning_price` to a C3 fit and asserts the win head scores essentially
perfectly, so the ban is demonstrated to matter rather than merely being present.
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
from t9v2.train import metrics as MET
from t9v2.train.tier2 import Tier2

warnings.filterwarnings("ignore")

K = 20.0


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    p = tmp_path_factory.mktemp("loo") / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    return pd.read_parquet(p)


def toy(values, keys=("a", "b", "c", "d")):
    """A frame with ONE cell and one parent, so the EB algebra is doable by hand."""
    n = len(values)
    return pd.DataFrame({
        "app_id": ["app1"] * n, "slot_format": [1] * n,
        "ad_exchange": ["x"] * n, "_daypart": [0] * n,
        "min_winning_price": np.asarray(values, dtype=float),
        "won": np.ones(n, dtype=int),
        "winning_price": np.asarray(values, dtype=float),
        "bid_price": np.asarray(values, dtype=float),
    })


KEYS = ["app_id", "slot_format", "ad_exchange", "_daypart"]


# ----------------------------------------------------------- the arithmetic

def test_leave_one_out_matches_the_hand_calculation():
    """One cell, four rows, worked out on paper.

    Cell and parent are the same rows here (one app), so with root = mean:

        root       = 10
        parent_i   = (40 - v + 20.10) / (3 + 20)
        cell_i     = (40 - v + 20.parent_i) / (3 + 20)
    """
    d = toy([4.0, 8.0, 12.0, 16.0])
    e = E.PriceEncoder(KEYS, K, "t", "min_winning_price", "all").fit(
        d, d["min_winning_price"].to_numpy())
    assert e.root == 10.0
    got = e.transform_train(d)
    for i, v in enumerate([4.0, 8.0, 12.0, 16.0]):
        par = (40.0 - v + K * 10.0) / (3.0 + K)
        cell = (40.0 - v + K * par) / (3.0 + K)
        assert abs(got[i] - cell) < 1e-12, (i, got[i], cell)


def test_a_singleton_cell_collapses_to_its_parent():
    """Nothing left but the backoff, which is the right answer not an edge case."""
    d = toy([7.0])
    e = E.PriceEncoder(KEYS, K, "t", "min_winning_price", "all").fit(
        d, d["min_winning_price"].to_numpy())
    got = e.transform_train(d)[0]
    par = (7.0 - 7.0 + K * 7.0) / (0.0 + K)
    assert abs(got - par) < 1e-12
    assert abs(got - 7.0) < 1e-12, "with one row the root IS the value"


def test_the_correction_always_moves_the_estimate_away_from_the_row():
    """A high row's estimate must fall, a low row's must rise. The direction is
    the whole point: leaving it in pulls every estimate toward its own value."""
    d = toy([1.0, 2.0, 3.0, 40.0])
    e = E.PriceEncoder(KEYS, K, "t", "min_winning_price", "all").fit(
        d, d["min_winning_price"].to_numpy())
    plain, loo = e.transform(d), e.transform_train(d)
    v = d["min_winning_price"].to_numpy()
    assert ((v > plain) == (loo < plain))[v != plain].all(), \
        "the correction must move each estimate AWAY from that row's own value"


def test_rows_outside_the_fit_are_left_alone():
    """A row the encoder never averaged has nothing of its own to remove.

    The DSP encoder fits on won rows only, so a lost row contributed nothing and
    must come back with the plain estimate untouched.
    """
    d = toy([4.0, 8.0, 12.0, 16.0])
    d.loc[[1, 3], "won"] = 0
    e = E.PriceEncoder(KEYS, K, "dsp", "bid_price", "won").fit(
        d, d["bid_price"].to_numpy(), weight_mask=E.mask_of("won", d))
    plain, loo = e.transform(d), e.transform_train(d)
    assert np.allclose(plain[[1, 3]], loo[[1, 3]]), "lost rows must be untouched"
    assert not np.allclose(plain[[0, 2]], loo[[0, 2]]), "won rows must be corrected"


def test_an_encoder_that_cannot_do_it_refuses_rather_than_guessing():
    """The failure mode being guarded is silence, so the guard must not be silent."""
    d = toy([1.0, 2.0])
    e = E.PriceEncoder(KEYS, K, "nameless").fit(d, d["min_winning_price"].to_numpy())
    with pytest.raises(RuntimeError, match="leave-one-out"):
        e.transform_train(d)


# ------------------------------------------------------- on the real pipeline

def test_every_encoder_column_is_corrected_on_training_rows(df, s):
    """All five, not only the new one.

    The other four are not identities, but they are still a row's own history
    leaking into its own features, and a correction applied selectively is one
    somebody has to remember to extend.
    """
    d = F.prepare(CEN.censor(df, "C3", s), s)
    tr = F.split(d, s)["train"]
    enc = E.build(d[tr].copy(), "C3", s)
    plain = d.copy()
    cols = E.apply(enc, plain)
    corrected = d.copy()
    E.apply(enc, corrected, train_mask=tr)

    assert len(cols) == 5, cols

    # EXACTLY THE ROWS EACH ENCODER AVERAGED, and no others. The share corrected
    # is not a threshold to clear, it is each encoder's own mask rate: `dsp` fits
    # on won rows so about 30 percent of training rows move, `ssp` on cleared
    # rows about 76, `ssp_lost` on lost-sold about 46, and `ssp_minwin` on all of
    # them, 100. A column moving on more rows than its mask allows would mean the
    # correction is reaching rows that were never in the fold.
    named = {"_enc_dsp_price": "dsp", "_enc_ssp_price": "ssp",
             "_enc_ssp_lost_price": "ssp_lost", "_enc_ssp_minwin_price": "ssp_minwin"}
    for c in cols:
        a, b = plain[c].to_numpy(), corrected[c].to_numpy()
        if c == "_enc_ssp_density":
            assert np.allclose(a, b), "the density extra is deliberately uncorrected"
            continue
        e = enc[named[c]]
        in_fold = E.mask_of(e.mask_kind, d) & np.isfinite(
            d[e.value_col].to_numpy(dtype=float))
        moved = ~np.isclose(a, b)
        assert not moved[~tr].any(), \
            "%s: a non-training row was corrected; it was never in the fit" % c
        assert not moved[tr & ~in_fold].any(), \
            "%s: a training row OUTSIDE this encoder's mask was corrected" % c
        share = moved[tr].mean() / in_fold[tr].mean()
        assert share > 0.99, "%s: only %.1f%% of its own fold moved" % (c, 100 * share)

    # the new encoder's mask is `all`, so its fold IS the training split and the
    # loop above has already required better than 99 percent of it to move. Not
    # asserted at exactly 100: 24 rows in 71,447 sit close enough to their own
    # cell estimate that removing them does not move it by a float's width, which
    # is a coincidence of the data rather than a gap in the correction.
    assert enc["ssp_minwin"].mask_kind == "all"


def test_the_uncorrected_minwin_encoder_leaks_the_label(df, s):
    """MEASURED, and it is the number that justifies the whole correction.

    Fit a win classifier on nothing but the minwin encoder, and score it on the
    TRAINING rows both ways. Uncorrected, the encoder carries a shrunk copy of
    each row's own `min_winning_price`, and `won` is an identity in that column.
    """
    d = F.prepare(CEN.censor(df, "C3", s), s)
    tr = F.split(d, s)["train"]
    enc = E.build(d[tr].copy(), "C3", s)
    e = enc["ssp_minwin"]
    y = d.loc[tr, "won"].to_numpy(dtype=int)
    bid = d.loc[tr, "bid_price"].to_numpy(dtype=float)

    # the margin the win rule actually turns on, as the encoder reports it
    leaky = MET.auc(y, bid - e.transform(d)[tr])
    clean = MET.auc(y, bid - e.transform_train(d)[tr])
    assert leaky > clean, (
        "the uncorrected encoder should rank the label better than the corrected "
        "one; measured %.4f against %.4f" % (leaky, clean))


def test_the_raw_column_would_score_almost_perfectly(df, s):
    """The backstop the plan requires: show the exclusion is LOAD-BEARING.

    Force `min_winning_price` into a C3 Tier-2 fit and the win head scores
    essentially 1.0, because `won = 1[bid_price >= min_winning_price]` is an
    identity and both sides are in the feature list. A ban whose absence changed
    nothing would not need to be a ban.
    """
    d = F.prepare(CEN.censor(df, "C3", s), s)
    masks = F.split(d, s)
    te = masks["test"]
    y = d.loc[te, "won"].to_numpy(dtype=int)

    cols = F.tier2_features(d) + ["min_winning_price"]
    assert "bid_price" in cols and "min_winning_price" in cols
    leaked = MET.auc(y, Tier2().fit(d, masks, cols, s, 0).predict(d[te]))
    assert leaked >= 0.999, (
        "the raw column scored only %.4f; if this is no longer near-perfect the "
        "identity has changed and the ban's justification must be revisited"
        % leaked)

    # AND THE ORDINARY FIT BESIDE IT, which is what makes the number above mean
    # something. A backstop that only showed the leaked figure would not say
    # whether 0.999 is remarkable; the honest C3 fit reads about 0.81.
    honest = MET.auc(y, Tier2().fit(d, masks, F.tier2_features(d), s, 0).predict(d[te]))
    assert honest <= 0.95, (
        "the ordinary C3 win head scored %.4f. Either something else is leaking "
        "or the auction has stopped being hard to predict." % honest)
    assert leaked - honest > 0.15, (
        "the ban is not load-bearing: leaked %.4f against honest %.4f"
        % (leaked, honest))
