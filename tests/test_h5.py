# -*- coding: utf-8 -*-
"""H5 `min_winning_price`: the law, the graph, and the leak it would otherwise be.

H5 is `max(LU7, floor)` on every row — the smallest bid that would have won. It
is the TARGET of the Tier-2 price head and must never be an input to anything,
because `won = 1[bid_price >= min_winning_price]` is an identity.

THREE THINGS CAN GO WRONG AND ONLY ONE OF THEM IS NOISY.

  The law could be wrong. Loud: the branch assertions below fail.
  The graph could grant it to all four views. SILENT. `build_graph.py` decides
    censoring from `SSP_ONLY_NODES` and never parses H5's role text, so omitting
    it there writes a well-formed graph that hands C1 and C2 the exact hurdle.
  It could reach a feature list. SILENT, and worse: C3's win head would score
    near 1.0 while C1's did not, and the gap would read as the SSP result.

The second and third are tested against the built artifacts and the real feature
functions rather than against the constants that are supposed to prevent them. A
test that asserts `min_winning_price in SETTLEMENT_COLS` proves only that
somebody typed it there.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from t9v2 import censor as CEN
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.gen import rival_market as M
from t9v2.train import encoder as E
from t9v2.train import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["C1", "C2", "C3", "C4"]


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    p = tmp_path_factory.mktemp("h5") / "m.parquet"
    G.generate(scale="100K", seed=20250, n_rows=20_000, out=str(p), quiet=True)
    return pd.read_parquet(p)


# ------------------------------------------------------------------- the law

def test_h5_is_the_max_of_the_competing_bid_and_the_floor(df):
    assert np.array_equal(
        df["min_winning_price"].to_numpy(),
        np.maximum(df["lu7_competing_bid"].to_numpy(dtype=float),
                   df["floor_price"].to_numpy(dtype=float)))


def test_h5_is_present_on_every_row_including_unsold(df):
    """The difference from `winning_price`, and from v1's abandoned B1.

    `winning_price` is NaN on an unsold row because no price was paid. H5 is
    counterfactual: the floor is still the smallest bid that would have won,
    whether or not anybody bid it. v1's `min_bid_to_win` existed on won rows
    only, which is a value-selected sample and the defect the MMP mechanism
    turns on.
    """
    assert df["min_winning_price"].notna().all()
    unsold = (df["won"] == 0) & (df["lu7_competing_bid"] < df["floor_price"])
    assert unsold.sum() > 0, "no unsold rows in the fixture; the branch is untested"
    assert df.loc[unsold, "winning_price"].isna().all()


def test_h5_takes_the_right_value_in_each_of_the_three_branches(df):
    w = df["won"] == 1
    sold = (df["won"] == 0) & (df["lu7_competing_bid"] >= df["floor_price"])
    unsold = (df["won"] == 0) & (df["lu7_competing_bid"] < df["floor_price"])
    assert (w | sold | unsold).all() and (w.sum() and sold.sum() and unsold.sum())
    m = df["min_winning_price"]
    assert (m[w] == np.maximum(df["lu7_competing_bid"][w],
                               df["floor_price"][w])).all()
    assert (m[sold] == df["lu7_competing_bid"][sold]).all()
    assert (m[unsold] == df["floor_price"][unsold]).all()


def test_won_is_an_identity_in_h5_which_is_why_it_cannot_be_a_feature(df):
    """The reason for every exclusion below, stated as the fact it rests on."""
    assert np.array_equal(
        df["won"].to_numpy(dtype=int),
        (df["bid_price"].to_numpy(dtype=float)
         >= df["min_winning_price"].to_numpy(dtype=float)).astype(int))


def test_h5_consumes_no_randomness(tmp_path):
    """A pure function of two drawn columns, so the RNG stream is untouched.

    This is what lets step 5b's `k_global` solve stand without being redone
    after H5 landed. Checked here by calling the law directly on arbitrary
    inputs: no rng argument exists to pass, and the result is deterministic.
    """
    import inspect
    assert "rng" not in inspect.signature(M.min_winning_price).parameters
    a = M.min_winning_price(np.array([1.0, 5.0, 0.0]), np.array([2.0, 1.0, 0.0]))
    b = M.min_winning_price(np.array([1.0, 5.0, 0.0]), np.array([2.0, 1.0, 0.0]))
    assert np.array_equal(a, b) and np.array_equal(a, [2.0, 5.0, 0.0])


# ----------------------------------------------------------------- the graph

def test_the_graph_makes_h5_ssp_only():
    """The silent failure, tested on the BUILT graph rather than on the set.

    `build_graph.py` never parses H5's role text for the words "SSP-visible", so
    leaving H5 out of `SSP_ONLY_NODES` writes a well-formed graph that grants
    `min_winning_price` to all four views. Nothing raises and nothing in the
    output looks wrong.
    """
    g = yaml.safe_load((ROOT / "config" / "graph.yaml").read_text(encoding="utf-8"))
    node = {n["id"]: n for n in g["nodes"]}["H5"]
    assert node["columns"] == ["min_winning_price"]
    assert node["observability"] == {"C1": "none", "C2": "none",
                                     "C3": "all", "C4": "all"}
    assert node["observability"] == {n["id"]: n for n in g["nodes"]}["H4"]["observability"], \
        "H5 must be censored exactly as `winning_price` is"


def test_the_graph_carries_79_nodes_and_56_columns():
    g = yaml.safe_load((ROOT / "config" / "graph.yaml").read_text(encoding="utf-8"))
    assert len(g["nodes"]) == 79
    assert len(g["column_order"]) == 56
    i = g["column_order"].index("min_winning_price")
    assert g["column_order"][i - 1] == "winning_price", \
        "H5 sits immediately after H4 in deposit order"


def test_censoring_hides_h5_from_c1_and_c2(df, s):
    for v in VIEWS:
        cols = CEN.censor(df, v, s).columns
        assert ("min_winning_price" in cols) == (v in ("C3", "C4")), v


# ------------------------------------------------------------------ the leak

@pytest.mark.parametrize("view", VIEWS)
def test_h5_never_reaches_a_feature_list(df, s, view):
    """Built from the real feature functions on real data, not from the ban list.

    The weaker version of this test asserts `min_winning_price in
    SETTLEMENT_COLS`, which proves only that somebody typed it there. This runs
    the pipeline the training stack runs and asks what came out.
    """
    d = F.prepare(CEN.censor(df, view, s), s)
    t1 = set(F.tier1_features(d, "click"))
    t2 = set(F.tier2_features(d))
    assert "min_winning_price" not in t1, "%s Tier 1 holds the win label" % view
    assert "min_winning_price" not in t2, "%s Tier 2 holds its own label" % view
    for stage in ("install", "payer", "spend"):
        assert "min_winning_price" not in set(F.tier1_features(d, stage)), stage


def test_the_ssp_advantage_is_entirely_the_encoder_and_never_a_raw_column(df, s):
    """Measured, and it is the fact that makes H5's exclusion safe to make.

    C3's RAW Tier-2 feature count equals C1's — 20 each — because every column
    C3 gains over C1 is a settlement column and every settlement column is
    banned. `winning_price`, `bid_density` and now `min_winning_price` are all
    barred from entering as the current row's value.

    So the whole SSP contrast travels through the empirical-Bayes encoders, and
    only there. Price knowledge reaches the model as cell-level HISTORY and never
    as the row's own answer, which is the design v2 adopted after v1's
    `bid_density` leak, and is why banning H5 as a raw feature costs the ablation
    nothing: `_enc_ssp_minwin_price` carries what H5 knows, leave-one-out
    corrected, without ever handing a row its own value.
    """
    def counts(v):
        d = F.prepare(CEN.censor(df, v, s), s)
        enc = E.build(d[F.split(d, s)["train"]].copy(), v, s)
        return len(F.tier2_features(d)), len(F.tier2_features(d, extra=E.apply(enc, d))), sorted(enc)

    raw1, enc1, e1 = counts("C1")
    raw3, enc3, e3 = counts("C3")
    assert raw3 == raw1, "a raw SSP column has escaped the settlement ban"
    assert e1 == ["dsp"]
    assert e3 == ["dsp", "ssp", "ssp_lost", "ssp_minwin"]
    assert enc3 > enc1, "the SSP views must still have something C1 does not"


def test_no_eval_file_column_ever_becomes_a_feature(df, s):
    """The Phase A gate-3 skip, settled by measurement instead of argument.

    The per-row eval file holds `won_at_recommended` and `m_win_true`, both
    derived from `lu7_competing_bid` and `floor_price` — columns C1 and C2 cannot
    see. That is exactly the shape gate 3 exists to catch, and the reason given
    for skipping it was a JUDGEMENT: that the eval file is a scoring artifact,
    that `economics` already reads the uncensored master the same way, and that
    no training path reads the file back.

    The judgement is right, and an argument is a worse guard than an assertion.
    This is the assertion. It costs one test and removes the one skip in that
    walk that rested on reasoning.
    """
    from t9v2.evalfile import SCHEMA                       # noqa: F401
    written = ["row", "p_click", "p_install", "p_payer", "spend_hat", "ev",
               "bid_logged", "p_win_at_logged", "bid_recommended", "rung",
               "p_win_at_recommended", "profit_at_recommended", "placed",
               "won_logged", "won_at_recommended", "m_win_true", "m_win_pred"]
    from t9v2.train import price as PR
    for v in VIEWS:
        d = F.prepare(CEN.censor(df, v, s), s)
        enc = E.build(d[F.split(d, s)["train"]].copy(), v, s)
        ec = E.apply(enc, d)
        lists = {"tier1": set(F.tier1_features(d, "click", extra=ec)),
                 "tier2": set(F.tier2_features(d, extra=ec)),
                 "price": set(PR.features(d, ec))}
        for name, cols in lists.items():
            bad = sorted(cols & set(written))
            assert not bad, "%s %s holds eval-file column(s) %s" % (v, name, bad)


def test_the_eval_file_is_never_read_by_a_training_path():
    """The other half of the same judgement, checked structurally.

    If nothing under `train/` imports `evalfile`, the file cannot feed a fit
    however its columns are named. Checked by import rather than by reading,
    because a name-based check would miss an alias.
    """
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "t9v2" / "train"
    offenders = []
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names = [(n.module or "")] + [a.name for a in n.names]
            if any("evalfile" in x for x in names):
                offenders.append(f.name)
    assert not offenders, "the training stack imports evalfile: %s" % offenders
