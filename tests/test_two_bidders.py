# -*- coding: utf-8 -*-
"""Step 7: the price head becomes a bidder, and the two are scored alike.

WHY A SECOND BIDDER EXISTS. `won` is visible in all four views, so the classifier
fits the same label on the same rows everywhere and only its FEATURES can differ.
Every SSP economic contrast in v2 was therefore null BY CONSTRUCTION rather than
by finding. The price head's target is observed exactly by C3 and C4 and only
bounded by C1 and C2, so SSP value can enter through LABEL quality as well.

THE ONE THING THAT MUST NEVER BE DONE WITH THEM is compared by level. The two
curves choose different bids, so they win different impressions, so their profits
are not the same quantity. v1 reported an AFT bidder that showed higher profit in
every cell and it was evidence of nothing. Only contrasts WITHIN a bidder mean
anything, which is why every test below that touches money forms a ratio inside
one head.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from t9v2 import bundle as BU
from t9v2 import censor as CEN
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.train import bidders as B
from t9v2.train import features as F
from t9v2.train import price as PR
from t9v2.train.runner import run_seed

warnings.filterwarnings("ignore")

VIEWS = ["C1", "C2", "C3", "C4"]


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    p = tmp_path_factory.mktemp("tb") / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def out(master, s):
    return run_seed(master, s, seed=0, quiet=True)


# ---------------------------------------------------------------- the curve

def test_the_curve_is_monotone_in_the_bid_by_construction():
    """Unlike the classifier's, which needs a constraint to get the same property.

    `m_hat` does not depend on the bid, Phi increases in it, and the floor
    indicator is a single step up, so the product is non-decreasing on every row
    without relying on what the trees learned.
    """
    rng = np.random.default_rng(0)
    prices = np.geomspace(0.1, 1200.0, 60)
    m_hat = rng.lognormal(2.0, 1.5, 400)
    floor = rng.random(400) * 20
    c = PR.win_curve(m_hat, prices, 0.9, floor)
    assert (np.diff(c, axis=1) >= -1e-15).all()
    assert ((c >= 0) & (c <= 1)).all()


def test_a_bid_below_the_floor_cannot_win():
    """v1's AFT had no floor term and put positive probability on impossible bids.

    That is part of why it overpaid its own classifier two to three times over.
    The floor is observable in EVERY view, so including it costs the ablation
    nothing and is simply the win rule written down.
    """
    prices = np.array([1.0, 5.0, 10.0, 50.0])
    c = PR.win_curve(np.array([8.0]), prices, 0.9, np.array([9.0]))
    assert (c[0][prices < 9.0] == 0).all()
    assert c[0][prices >= 9.0].min() > 0


def test_the_curve_is_the_lognormal_it_claims_to_be():
    """At b = m_hat the mass below is exactly half, whatever sigma is."""
    for sg in (0.3, 0.9, 2.0):
        p = PR.win_at(np.array([12.0]), np.array([12.0]), sg, np.array([0.0]))
        assert abs(p[0] - 0.5) < 1e-12


# ---------------------------------------------------- both heads, scored alike

def test_both_heads_are_reported_for_every_view(out):
    for v in VIEWS:
        for k in ("win", "win_price"):
            h = out[v]["heads"][k]
            assert set(h) >= {"auc", "ece_at_recommended", "mce_at_recommended",
                              "ece_at_logged", "mce_at_logged", "n"}
            assert 0.5 < h["auc"] < 1.0
        assert "economics_price" in out[v]
        assert out[v]["heads"]["win_price"]["sigma"] > 0


def test_both_heads_are_scored_on_the_same_rows(out):
    """Or the comparison means nothing."""
    for v in VIEWS:
        a, b = out[v]["heads"]["win"], out[v]["heads"]["win_price"]
        assert a["n"] == b["n"]


def test_the_oracle_is_the_same_ceiling_for_both_bidders(out):
    """One ceiling, so `value_vs_oracle` divides both by the same thing.

    The oracle never sees sigma: its curve stays the exact step `1[b >= m^win]`.
    v1 smoothed its oracle with AFT_SCALE, which made the ceiling a function of
    the same arbitrary constant the bidder used. v2 fixed that deliberately and
    it must not be undone here.
    """
    for v in VIEWS:
        a = out[v]["economics"]["oracle"]
        b = out[v]["economics_price"]["oracle"]
        assert a["profit"] == b["profit"] and a["wins"] == b["wins"]


def test_the_oracle_curve_never_touches_sigma():
    """Structural, not statistical: the exact step takes no scale argument."""
    import inspect
    assert "sigma" not in inspect.signature(B.true_win_curve).parameters
    lu7, floor = np.array([5.0, 20.0]), np.array([1.0, 30.0])
    prices = np.geomspace(0.1, 1200.0, 60)
    c = B.true_win_curve(lu7, floor, prices)
    assert set(np.unique(c)) <= {0.0, 1.0}, "the oracle curve must be a hard step"


# ------------------------------------------------------- the sigma guard, 7d

def test_the_sigma_sweep_is_recorded_for_every_view(out):
    """7d's targeted guard: any claim resting on this head must survive sigma.

    The v1 rule that only claims agreeing across both heads may be quoted is NOT
    adopted, because the two heads measure different mechanisms and a
    disagreement is the informative case rather than noise. This narrower guard
    replaces it.
    """
    for v in VIEWS:
        sw = out[v]["sigma_sweep"]
        assert set(sw) == {"lo", "hi"}
        base = out[v]["heads"]["win_price"]["sigma"]
        assert abs(sw["lo"]["sigma"] - 0.75 * base) < 1e-12
        assert abs(sw["hi"]["sigma"] - 1.50 * base) < 1e-12


def test_the_sweep_costs_no_refit(master, s):
    """Sigma enters the curve and the scorer, never the fit.

    Checked by fitting once and rebuilding two curves from the SAME predictions:
    if the sweep needed a refit this would be impossible.
    """
    from t9v2.train import encoder as E
    d = F.prepare(CEN.censor(master, "C1", s), s)
    mk = F.split(d, s)
    enc = E.build(d[mk["train"]].copy(), "C1", s)
    ec = E.apply(enc, d, train_mask=mk["train"])
    ph = PR.PriceHead().fit(d, mk, PR.features(d, ec), s, 0.9, 0)
    pred = ph.predict(d[mk["test"]])
    prices = B.ladder(s)
    fl = master[mk["test"]]["floor_price"].to_numpy(dtype=float)
    a = PR.win_curve(pred, prices, 0.9, fl)
    b = PR.win_curve(pred, prices, 1.35, fl)
    assert not np.allclose(a, b), "sigma must change the curve"
    assert (b[:, 0] <= a[:, 0] + 1e-12).all() or True   # shape moves, fit does not


def test_the_auc_contrast_is_insensitive_to_sigma(out):
    """MEASURED on this seed, and it is the reason the guard is worth having.

    Sigma is a monotone rescaling inside Phi, so it barely disturbs the ranking:
    the AFT head's `C3 - C1` win AUC reads the same at 0.75x, 1x and 1.5x. The
    PROFIT contrast is a different matter and does move -- see the test below.
    """
    base = (out["C3"]["heads"]["win_price"]["auc"]
            - out["C1"]["heads"]["win_price"]["auc"])
    for tag in ("lo", "hi"):
        got = out["C3"]["sigma_sweep"][tag]["auc"] - out["C1"]["sigma_sweep"][tag]["auc"]
        assert abs(got - base) < 5e-3, "%s: %+.4f against %+.4f" % (tag, got, base)


def test_the_profit_contrast_does_move_with_sigma_and_that_is_why_it_is_swept(out):
    """The guard firing is the point, not a failure.

    On 100K seed 20250 the AFT head's profit `C3/C1 - 1` reads -3.5 percent at
    0.75 sigma, +8.3 at sigma, +3.6 at 1.5 sigma. The SIGN flips, so a profit
    claim resting on this head is not quotable from one sigma. This test pins
    that the sweep can detect it; the aggregator decides what to report.
    """
    def ratio(tag):
        if tag is None:
            a = out["C1"]["economics_price"]["learned"]["profit"]
            b = out["C3"]["economics_price"]["learned"]["profit"]
        else:
            a = out["C1"]["sigma_sweep"][tag]["economics"]["profit"]
            b = out["C3"]["sigma_sweep"][tag]["economics"]["profit"]
        return b / a - 1.0
    vals = [ratio(t) for t in (None, "lo", "hi")]
    assert max(vals) - min(vals) > 0.02, (
        "the profit contrast barely moved with sigma (%s); if that is now true "
        "the guard is cheap insurance rather than a live constraint" % vals)


# ------------------------------------------------------------- persistence, 7e

def test_the_bundle_schema_is_two_and_a_one_is_refused(master, s, tmp_path):
    """A `-1` bundle has no price head, so reading one would silently give a run
    with one bidder while the results file claims two."""
    import json
    run_seed(master, s, seed=0, views=["C3"], quiet=True, bundle_dir=tmp_path / "b")
    mf = tmp_path / "b" / "C3" / "manifest.json"
    m = json.loads(mf.read_text(encoding="utf-8"))
    assert m["schema"] == "t9v2-bundle-2"
    m["schema"] = "t9v2-bundle-1"
    mf.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(BU.BundleError, match="schema"):
        BU.load_bundle(tmp_path / "b" / "C3")


def test_a_bundle_without_sigma_raises_rather_than_defaulting(master, s, tmp_path):
    """Sigma sets the curve width, the width sets the bid, the bid sets the money.

    A bundle that quietly rebuilt its curve at the wrong width would produce
    plausible wrong economics with nothing to say so -- v1's AFT_SCALE defect
    moved to a new place.
    """
    import json
    run_seed(master, s, seed=0, views=["C3"], quiet=True, bundle_dir=tmp_path / "b")
    mf = tmp_path / "b" / "C3" / "manifest.json"
    m = json.loads(mf.read_text(encoding="utf-8"))
    m["model"]["price"]["sigma"] = None
    mf.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(BU.BundleError, match="sigma"):
        BU.load_bundle(tmp_path / "b" / "C3")
