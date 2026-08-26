# -*- coding: utf-8 -*-
"""The per-row eval file, and the calibration that could not exist without it.

The file is written by a pass that LOADS a bundle rather than by the fit that
produced one. That is the design: the frozen artifact sits on the critical path,
so numbers come from the file that shipped and a bundle that reloads differently
is caught here rather than never.

SCALE MATTERS IN THESE TESTS AND IS NOT ARBITRARY. At 1,000 rows the win
classifier never splits on `bid_price` — 600 training rows and early stopping —
so its curve is flat and the two win probabilities are equal by degeneracy rather
than by agreement. The distinction this file exists to capture is invisible
there. Anything asserting that the two differ therefore runs at 100K, where they
differ on 85 percent of rows.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from t9v2 import bundle as BU
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.evalfile import write_eval
from t9v2.train.runner import run_seed

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def small(s, tmp_path_factory):
    """1,000 rows: fast, and the degenerate paths live here."""
    p = tmp_path_factory.mktemp("d1k") / "m.parquet"
    G.generate(scale="100K", seed=20250, n_rows=1000, out=str(p), quiet=True)
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def full(s, tmp_path_factory):
    """100K: the smallest scale at which the win curve responds to the bid."""
    p = tmp_path_factory.mktemp("d100k") / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def scored(full, s, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    run_seed(full, s, seed=0, quiet=True, bundle_dir=root / "bundles")
    write_eval(full, root / "bundles", root / "eval", s, quiet=True)
    return root


def test_the_eval_file_is_written_per_view(scored):
    m = json.loads((scored / "eval" / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(m["views"]) == ["C1", "C2", "C3", "C4"]
    for v in m["views"]:
        assert (scored / "eval" / ("%s.parquet" % v)).exists()


def test_the_recommended_bid_is_the_rung_the_index_names(scored, s):
    """The file must not re-derive the bid; it records what the bidder chose."""
    from t9v2.train import bidders as B
    prices = B.ladder(s)
    d = pd.read_parquet(scored / "eval" / "C4.parquet")
    assert np.allclose(prices[d["rung"].to_numpy()], d["bid_recommended"].to_numpy())


def test_the_two_win_probabilities_are_different_numbers(scored):
    """The reason this file exists.

    `p_win_at_logged` is the classifier at the bid the generator recorded;
    `p_win_at_recommended` is the curve at the rung the bidder chose. v2 measured
    calibration at the first, and the bidder operates at the second.
    """
    d = pd.read_parquet(scored / "eval" / "C1.parquet")
    differ = (d["p_win_at_logged"] != d["p_win_at_recommended"]).mean()
    assert differ > 0.5, "only %.1f%% of rows differ; is the curve flat?" % (100 * differ)
    assert d["p_win_at_recommended"].mean() > d["p_win_at_logged"].mean()


def test_won_at_recommended_is_the_true_win_rule(scored, full, s):
    """The counterfactual outcome is read from the master, not predicted."""
    from t9v2.train import features as F
    from t9v2 import censor as CEN
    d = pd.read_parquet(scored / "eval" / "C1.parquet")
    prep = F.prepare(CEN.censor(full, "C1", s), s)
    te = np.flatnonzero(np.asarray(F.split(prep, s)["test"]))
    assert np.array_equal(d["row"].to_numpy(), te), "the file must name its rows"
    m = full.iloc[te]
    hurdle = np.maximum(m["lu7_competing_bid"].to_numpy(dtype=float),
                        m["floor_price"].to_numpy(dtype=float))
    # THE GATE IS PART OF THE RULE, and was invisible in this assertion until
    # 25 August 2026. `bid_recommended` is the UNGATED argmax on every row, kept
    # that way so the file records what the bidder would pay as well as whether
    # it was willing to. A win needs both: the row placed, and the bid cleared.
    # At the old target of 1.0 `placed` was all ones, so dropping it from the
    # comparison changed nothing and the test passed while checking less than it
    # claimed. At 3.0 it does not.
    placed = d["placed"].to_numpy().astype(bool)
    assert not placed.all(),         "the gate declined nothing, so this test is not checking the gate"
    assert np.array_equal(
        d["won_at_recommended"].to_numpy(),
        (placed & (d["bid_recommended"].to_numpy() >= hurdle)).astype(np.int8))


def test_calibration_is_reported_at_both_bids(full, s):
    """Both keys, and the bare `ece_win` gone.

    v2's thirty committed results.json files mean the LOGGED bid by `ece_win`,
    so reusing that name for the recommended one would make two schemas
    indistinguishable.
    """
    # sigma stated: C1 is censored and C3 is not in this run, so the scale the
    # campaign reads off C3 has to be supplied. Calibration is what is measured.
    out = run_seed(full, s, seed=0, views=["C1"], quiet=True, sigma=1.0)
    w = out["C1"]["heads"]["win"]
    for k in ("ece_at_recommended", "mce_at_recommended",
              "ece_at_logged", "mce_at_logged"):
        assert k in w and np.isfinite(w[k]), k
    assert "ece" not in w and "mce" not in w, "the ambiguous bare key must be gone"


def test_auc_stays_at_the_logged_bid(full, s):
    """`won` is the outcome at the bid the row carried.

    A ranking scored at per-row recommended bids would pair each score with a
    label from a different counterfactual, so AUC does not move with calibration.
    """
    from t9v2.train import metrics as M
    out = run_seed(full, s, seed=0, views=["C1"], quiet=True, sigma=1.0)
    w = out["C1"]["heads"]["win"]
    assert 0.5 < w["auc"] < 1.0
    # AUC IS SCORED ON EVERY TEST ROW. Calibration at the recommended bid is
    # not: a declined row carries a probability the bidder never bet on, so
    # `n_at_recommended` counts PLACED rows only. The two were equal until 25
    # August 2026 because the gate declined nothing at a target of 1.0, which
    # made this line read like a statement about the population when it was an
    # accident of the target.
    e = out["C1"]["economics"]["learned"]
    assert w["n_at_recommended"] < w["n"], "the gate declined nothing"
    assert abs(w["n_at_recommended"] / w["n"] - e["placed_rate"]) < 1e-6,         "calibration must be scored on exactly the placed rows: %d/%d against a "        "placed rate of %.4f" % (w["n_at_recommended"], w["n"], e["placed_rate"])


def test_the_eval_pass_refuses_a_missing_bundle(full, s, tmp_path):
    with pytest.raises(FileNotFoundError, match="no bundle"):
        write_eval(full, tmp_path / "nothing", tmp_path / "eval", s, quiet=True)


def test_the_floor_fix_woke_the_1k_win_curve_up(small, s, tmp_path):
    """This test asserted the OPPOSITE until 22 August 2026, and the fix flipped it.

    It was written as a tripwire. At 1,000 rows the win classifier never split on
    `bid_price`, so its curve was flat and the two win probabilities were equal
    by degeneracy rather than by agreement; the assertion was `allclose`, with a
    message saying that if a change ever made the 1K curve respond, the fixtures
    could be revisited. The floor fix is that change, and it found it.

    WHY THE FLOOR WAS SILENCING THE CURVE. The buggy floor was inflated by 1.750,
    so on about a third of rows it already exceeded our bid and the outcome was
    settled before the bid mattered. Those rows carry nothing about the bid and
    there were enough of them to swamp 600 training rows. Correcting the
    denominator removes them: the 1K win rate goes from about 0.30 to 0.3770, the
    curve separates on 46.2 percent of test rows, and it rises from 0.3441 at the
    bottom rung to 0.3839 at the top.

    THE 100K FIXTURES STAY. 46.2 percent is a response, not a strong one, and the
    claim two tests above — that the two win probabilities differ on more than
    half of rows — was measured at 100K and has not been re-measured here.
    Simplifying the fixtures is a separate question from recording that the floor
    fix moved this, and only the second is in scope.
    """
    from t9v2.train import bidders as B, features as F, encoder as E
    from t9v2 import censor as CEN
    run_seed(small, s, seed=0, views=["C4"], quiet=True, bundle_dir=tmp_path / "b")
    pieces = BU.load_bundle(tmp_path / "b" / "C4")
    d = F.prepare(CEN.censor(small, "C4", s), s)
    E.apply(pieces["encoders"], d)
    d_te = d[F.split(d, s)["test"]]
    curve = pieces["tier2"].win_curve(d_te, B.ladder(s))
    assert not np.allclose(curve[:, 0], curve[:, -1]), \
        "the 1K curve has gone flat again; has the floor regressed?"
    assert curve[:, -1].mean() > curve[:, 0].mean(), "and it must rise with the bid"
