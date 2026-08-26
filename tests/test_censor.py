# -*- coding: utf-8 -*-
"""Stage 3: the censoring into C1 to C4.

The plan states four pass conditions for gate 3, and the first four tests here
are those conditions, one each. The rest guard the properties that make the
ablation an ablation rather than four separate experiments: the views must be
masks over one master, and the three kinds of absence must stay distinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t9v2 import censor as C
from t9v2 import generate as G
from t9v2.core import config

ROWS = 20_000


@pytest.fixture(scope="module")
def s():
    return config.load("default")


@pytest.fixture(scope="module")
def master():
    return G.frame("100K", seed=20250, n_rows=ROWS)


@pytest.fixture(scope="module")
def views(master, s):
    return C.all_views(master, s)


# ---------------------------------------------------------------------------
# the 4 gate-3 pass conditions
# ---------------------------------------------------------------------------

def test_click_reads_won_all_won_all(s):
    """Gate 3, condition 1."""
    obs = s.node("E1")["observability"]
    assert [obs[v] for v in C.VIEWS] == ["won", "all", "won", "all"]


def test_every_latent_is_hidden_in_all_four_views(s, views):
    """Gate 3, condition 2."""
    latent = [c for n in s.emitting() if n["observability"] == "latent"
              for c in n["columns"]]
    assert len(latent) == 24
    for v, d in views.items():
        leaked = sorted(set(latent) & set(d.columns))
        assert not leaked, "view %s exposes latent column(s) %s" % (v, leaked)


def test_user_id_is_hidden_everywhere(views):
    """Gate 3, condition 3. Master lineage only: it identifies a row's origin and
    must never be learnable from a view."""
    for v, d in views.items():
        assert "user_id" not in d.columns, "view %s carries user_id" % v


def test_the_views_nest(views):
    """Gate 3, condition 4. C4 sees at least what C2 sees, C2 at least what C1.

    Checked on both axes, because the two layers add different things. SSP adds
    COLUMNS, so the column sets nest. MMP adds ROWS to columns already present,
    so the count of visible cells nests too. A test on columns alone would pass
    for a C2 that hid every funnel cell.
    """
    for a, b in [("C1", "C2"), ("C2", "C4"), ("C1", "C3"), ("C3", "C4")]:
        assert set(views[a].columns) <= set(views[b].columns), \
            "%s has columns %s does not" % (a, b)
        for col in views[a].columns:
            seen_a = int(views[a][col].notna().sum())
            seen_b = int(views[b][col].notna().sum())
            assert seen_a <= seen_b, \
                "%s shows %d cells of %s, more than %s's %d" % (a, seen_a, col, b, seen_b)


# ---------------------------------------------------------------------------
# won is always visible, the defect found at the start of stage 3
# ---------------------------------------------------------------------------

def test_won_is_visible_on_every_row_in_every_view(views):
    """Specification 1.5 lists H3 under "Always visible: A, B, C, D, H1, H2, H3".

    It had been derived as a funnel outcome, which would have hidden it on lost
    rows in C1 and C3. Tier 2 learns P(win | bid) from this column, so those two
    views would have had no losses to learn from, and they are exactly the two
    the SSP result is measured across.
    """
    for v, d in views.items():
        assert "won" in d.columns, "view %s has no won column" % v
        assert d["won"].notna().all(), "view %s hides won on some rows" % v
        assert set(d["won"].unique()) == {0, 1}, \
            "view %s does not show both outcomes of won" % v


# ---------------------------------------------------------------------------
# the views are masks over one master
# ---------------------------------------------------------------------------

def test_every_view_has_the_master_rows_in_the_master_order(master, views):
    for v, d in views.items():
        assert len(d) == len(master), "view %s has %d rows, master has %d" % (
            v, len(d), len(master))
        assert d.index.equals(master.index)
        assert d["bid_price"].equals(master["bid_price"]), \
            "view %s reordered or altered an always-visible column" % v


def test_visible_cells_equal_the_master(master, views):
    """Censoring hides, it never changes. Any cell a view shows must be the
    master's own value."""
    for v, d in views.items():
        for col in d.columns:
            vis = d[col].notna()
            a = d.loc[vis, col].to_numpy()
            b = master.loc[vis.to_numpy(), col].to_numpy()
            if a.dtype.kind == "f":
                assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), \
                    "%s changed %s" % (v, col)
            else:
                assert (a == b).all(), "%s changed %s" % (v, col)


def test_masked_cells_are_hidden_exactly_on_lost_rows(master, views, s):
    for v in ("C1", "C3"):
        lost = master["won"].to_numpy() == 0
        for col in C.plan(s, v)["mask"]:
            assert (views[v][col].isna().to_numpy() == lost).all(), \
                "%s hid %s on the wrong rows" % (v, col)


def test_c2_and_c4_hide_no_funnel_cell(views, s):
    for v in ("C2", "C4"):
        assert C.plan(s, v)["mask"] == []
        for col in ("click", "install", "is_payer", "ltv_value"):
            assert views[v][col].notna().all()


# ---------------------------------------------------------------------------
# the three kinds of absence stay distinguishable
# ---------------------------------------------------------------------------

def test_no_event_sentinel_survives_censoring(master, views):
    """-1 is an OBSERVED no-event timestamp and must not become NA.

    Three kinds of absence live in these columns and collapsing any pair would
    change what a model learns: NA is "you were not told", -1 is "you were told
    it did not happen", and 0 is a measured negative label.
    """
    won = master["won"].to_numpy() == 1
    for v in ("C1", "C3"):
        col = views[v]["click_timestamp"]
        shown = col[won]
        assert (shown == -1).any(), "%s shows no -1 sentinel on won rows" % v
        assert shown.notna().all()
    for v in ("C2", "C4"):
        assert (views[v]["click_timestamp"] == -1).any()
        assert views[v]["click_timestamp"].notna().all()


def test_zero_labels_are_not_turned_into_missing(views):
    for v, d in views.items():
        seen = d["click"].dropna()
        assert (seen == 0).any(), "%s shows no negative click label" % v
        assert (seen == 1).any(), "%s shows no positive click label" % v


def test_masked_integer_columns_stay_integral(views):
    """Masking must not turn a count into a float. A nullable integer keeps the
    column integral and gains a real missing value."""
    for v in ("C1", "C3"):
        assert str(views[v]["click"].dtype) in ("Int8", "Int16", "Int32", "Int64")
        assert str(views[v]["install"].dtype).startswith("Int")


# ---------------------------------------------------------------------------
# what each layer adds
# ---------------------------------------------------------------------------

def test_ssp_adds_exactly_three_columns(views):
    """Three since H5, and every one of them is barred as a raw feature.

    `min_winning_price` joins `winning_price` and `bid_density` in C3 and C4,
    which means the SSP views gain a column and gain no raw feature: all three
    are settlement columns. The whole SSP contrast travels through the
    empirical-Bayes encoders instead, which is the point of the design and is
    measured in tests/test_h5.py.
    """
    added = set(views["C3"].columns) - set(views["C1"].columns)
    assert added == {"bid_density", "winning_price", "min_winning_price"}
    assert set(views["C4"].columns) - set(views["C2"].columns) == added


def test_mmp_adds_rows_and_no_columns(views):
    assert set(views["C2"].columns) == set(views["C1"].columns)
    assert set(views["C4"].columns) == set(views["C3"].columns)
    gained = int(views["C2"]["click"].notna().sum() - views["C1"]["click"].notna().sum())
    assert gained > 0, "C2 adds no funnel rows over C1"


def test_the_column_counts_match_the_declared_map(s):
    got = {r["view"]: r["columns"] for r in C.summary(s)}
    assert got == {"C1": 29, "C2": 29, "C3": 32, "C4": 32}
