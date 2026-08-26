# -*- coding: utf-8 -*-
"""The frozen bundle must reload into the same predictor, exactly.

Everything downstream reads a bundle rather than a fitted object: the per-row
eval file, the calibration at the recommended bid, and any metric added later.
So "close enough" is not a passing grade. If a reloaded model predicts a
thousandth differently the published number and the shipped model disagree, and
nothing anywhere would say so.

These run at 1,000 rows on purpose. That scale has one payer, so the spend head
falls back to an intercept or to unavailable, and `spend_scale` is None rather
than a number. That is the path where a bundle silently writes 0.0 for None and
changes both the ev and the CRPS, and it is a path 100K never reaches.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from t9v2 import bundle as BU
from t9v2 import censor as CEN
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.train import encoder as E
from t9v2.train import features as F
from t9v2.train.runner import train_view

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def master(s, tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "wiring_1K.parquet"
    G.generate(scale="100K", seed=20250, n_rows=1000, out=str(p), quiet=True)
    return pd.read_parquet(p)


def _fit_and_freeze(master, s, view, root):
    # sigma stated because a censored view is trained here without C3, which is
    # where the campaign reads it off. Its value is not what this test measures.
    res, fitted = train_view(master, view, s, seed=0, sigma=1.0)
    BU.save_bundle(fitted, s, root, meta={"view": view, "scale": "1K"})
    return fitted, BU.load_bundle(root)


def _scored(master, s, view, pieces):
    """Rebuild the encoder columns the way the trainer does, then predict."""
    d = F.prepare(CEN.censor(master, view, s), s)
    E.apply(pieces["encoders"], d)
    return pieces["tier1"].predict(d), pieces["tier2"].predict(d)


@pytest.mark.parametrize("view", ["C1", "C4"])
def test_reloaded_bundle_predicts_identically(master, s, tmp_path, view):
    """Bit-identical, not close. C1 and C4 differ in which columns they hold."""
    fitted, back = _fit_and_freeze(master, s, view, tmp_path / view)
    a1, a2 = _scored(master, s, view, fitted)
    b1, b2 = _scored(master, s, view, back)

    for head in a1:
        assert np.array_equal(np.nan_to_num(a1[head], nan=-1.0),
                              np.nan_to_num(b1[head], nan=-1.0)), \
            "%s: tier1 %s moved on reload" % (view, head)
    assert np.array_equal(a2, b2), "%s: the win model moved on reload" % view


def test_spend_scale_none_is_not_zero(master, s, tmp_path):
    """An absent measurement must not come back as a number.

    `Tier1.predict` reads `self.scale or 0.0` and the CRPS path reads
    `t1.scale or 1.0`, so None and 0.0 take different branches. At 1,000 rows
    C1 sees conversions on won rows only and the spend head is unavailable.
    """
    fitted, back = _fit_and_freeze(master, s, "C1", tmp_path / "C1")
    if fitted["tier1"].scale is None:
        assert back["tier1"].scale is None, "None came back as %r" % back["tier1"].scale
    else:
        assert back["tier1"].scale == fitted["tier1"].scale
    manifest = json.loads((tmp_path / "C1" / "manifest.json").read_text(encoding="utf-8"))
    assert "spend_scale" in manifest["model"], "the key must be present even when null"


def test_a_tampered_file_is_refused(master, s, tmp_path):
    """The sha256 per file is the point of recording it."""
    _fit_and_freeze(master, s, "C4", tmp_path / "C4")
    victim = tmp_path / "C4" / "tier2" / "win.ubj"
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(BU.BundleError, match="has changed since it was saved"):
        BU.load_bundle(tmp_path / "C4")


def test_an_unknown_schema_is_refused(master, s, tmp_path):
    """A future bundle read by this build must fail loudly, not partially."""
    _fit_and_freeze(master, s, "C4", tmp_path / "C4")
    mf = tmp_path / "C4" / "manifest.json"
    m = json.loads(mf.read_text(encoding="utf-8"))
    m["schema"] = "t9v2-bundle-99"
    mf.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(BU.BundleError, match="schema"):
        BU.load_bundle(tmp_path / "C4")


def test_the_encoders_travel_with_the_models(master, s, tmp_path):
    """A bundle without its cell tables loads and cannot score anything.

    A head's feature list names `_enc_ssp_price`; without the table the column
    cannot be rebuilt, so the encoders are as load-bearing as the models.
    """
    fitted, back = _fit_and_freeze(master, s, "C4", tmp_path / "C4")
    assert sorted(back["encoders"]) == sorted(fitted["encoders"])
    for name, e in back["encoders"].items():
        o = fitted["encoders"][name]
        assert e.keys == o.keys and e.shrink == o.shrink
        assert np.isclose(e.root, o.root)
        d = F.prepare(CEN.censor(master, "C4", s), s)
        assert np.allclose(e.transform(d), o.transform(d), equal_nan=True), \
            "%s: the reloaded encoder scores differently" % name


def test_the_feature_contract_is_recorded(master, s, tmp_path):
    """A bundle must say what it was fitted on, or nothing can check a leak."""
    fitted, back = _fit_and_freeze(master, s, "C4", tmp_path / "C4")
    assert back["t1_cols"] == fitted["t1_cols"]
    assert back["t2_cols"] == fitted["t2_cols"]
    assert "bid_price" in back["t2_cols"], "the win model reads the bid it is scored at"


def test_run_seed_writes_a_bundle_per_view(master, s, tmp_path):
    """The models must be frozen while still in hand.

    run_seed releases each view's fitted pieces and calls gc immediately, because
    at 10M two views' frames do not fit in memory at once. So the save happens
    before the release or not at all, and this asserts it happened.
    """
    from t9v2.train.runner import run_seed

    out = run_seed(master, s, seed=0, views=["C1", "C4"], quiet=True, sigma=1.0,
                   bundle_dir=tmp_path / "bundles")
    for v in ("C1", "C4"):
        root = tmp_path / "bundles" / v
        assert (root / "manifest.json").exists(), "%s has no bundle" % v
        assert out[v]["bundle"] == v, "results.json must name the bundle"
        BU.load_bundle(root)                      # loads, checksums, or raises


def test_run_seed_without_a_bundle_dir_is_unchanged(master, s):
    """The default stays exactly what v2 did, so nothing else has to change."""
    from t9v2.train.runner import run_seed

    out = run_seed(master, s, seed=0, views=["C4"], quiet=True)
    assert "bundle" not in out["C4"]
