# -*- coding: utf-8 -*-
"""The second scoring population, and the claims the fix rests on.

`runner.py` scores every Tier-1 head under one mask, `shown = click.notna()`, so
C1 and C3 are graded on the rows they won and C2 and C4 on every test row. The
two are not the same exam, and about two thirds of the reported MMP click lift is
the difference between them.

The fix asserts three things about the artifacts, and each is tested here rather
than argued, because each is the kind of claim that sounds obvious and would be
expensive to have wrong:

  1. censoring is a MASK, so the master still holds every label
  2. every view PREDICTED every test row, so nothing needs retraining
  3. restricting to won rows LOWERS a Tier-1 AUC, which is why the reported
     contrast is inflated rather than deflated

The last two guard the wiring, and the contract they assert CHANGED on 23 August
2026. A missing `tier1_allrows.json` used to leave the report working, because
the funnel rows sat beside the per-view figures and only supplementary rows were
lost. The own-rows rows were then dropped and the remaining funnel rows became
wholly dependent on that file, so the report now REFUSES to write rather than
publishing a table with no Tier-1 ranking metrics and no gap where they were.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from t9v2 import censor as CEN
from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.train import metrics as M

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s():
    return load("default")


@pytest.fixture(scope="module")
def df(tmp_path_factory, s):
    p = tmp_path_factory.mktemp("allrows") / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    return pd.read_parquet(p)


def test_the_master_keeps_every_label_the_view_hides(df, s):
    """Censoring is a mask over an intact master, which is why no re-censoring
    is needed to score on all rows. If this ever stops holding, the all-rows
    figure stops being computable and the whole fix goes with it."""
    lost = (df["won"] == 0).to_numpy()
    assert lost.sum() > 0, "a 100K seed with no lost rows cannot test this"
    for col in ("click", "install", "is_payer"):
        assert df[col].isna().sum() == 0, \
            "the MASTER must hold %s on every row, including lost ones" % col
    for view in ("C1", "C3"):
        d = CEN.censor(df, view, s)
        assert d["click"].isna().sum() == lost.sum(), \
            "%s must hide click on exactly the lost rows" % view
    for view in ("C2", "C4"):
        d = CEN.censor(df, view, s)
        assert d["click"].isna().sum() == 0, \
            "%s has MMP, so it sees the funnel on every row" % view


def test_restricting_to_won_rows_lowers_the_auc(df, s):
    """The direction that makes the reported contrast INFLATED, not deflated.

    Won rows are the slice the bidder bid up, which correlates with what it
    predicted, so the predictor has less spread to discriminate across. A ranking
    metric falls under that restriction. Measured here on the generator's own
    truth probability, so the test needs no fitted model and cannot drift with
    one: the effect is a property of the row selection.
    """
    won = (df["won"] == 1).to_numpy()
    y = df["click"].to_numpy(dtype=float)
    p = df["p_click"].to_numpy(dtype=float)          # the truth probability
    all_rows, own_rows = M.auc(y, p), M.auc(y[won], p[won])
    assert p[won].std() < p.std(), (
        "won rows should carry LESS predictor spread than all rows; measured "
        "%.5f against %.5f" % (p[won].std(), p.std()))
    assert own_rows < all_rows, (
        "restricting to won rows should LOWER the AUC; measured %.4f against "
        "%.4f. If this reverses, the sign of the reported bias reverses with it."
        % (own_rows, all_rows))


def test_the_tool_agrees_with_a_hand_computed_auc(df, s):
    """The tool's arithmetic against a calculation that does not use the tool."""
    t1 = _tool("tier1_allrows")
    assert t1.MASKED == ("C1", "C3"), \
        "the masked views are the ones without MMP; anything else is a mistake"
    names = [h[0] for h in t1.HEADS]
    assert names == ["click", "install", "payer"], names
    # the parent conditions must match the hurdle: install is scored on clicked
    # rows and payer on installed ones, or the denominators are wrong
    assert [h[3] for h in t1.HEADS] == [None, "click", "install"]


def test_the_report_refuses_to_write_without_the_json(tmp_path, monkeypatch):
    """It used to survive. It must not any more, and the change is deliberate.

    While the funnel rows sat BESIDE the per-view figures, an absent
    `tier1_allrows.json` cost three supplementary rows and left a complete
    table, so carrying on was right. The own-rows rows were then dropped,
    because a figure that must not be compared has no place in a table whose
    every row carries a contrast column, and the remaining funnel rows became
    wholly dependent on a file this script does not produce.

    An absent row leaves no gap. Carrying on would now publish a results
    document with no click, install or payer AUC in it and nothing to say so,
    which is worse than not writing at all.
    """
    rr = _tool("results_report")
    runs = ROOT / "output" / "runs"
    if not (runs / "100K").exists():
        pytest.skip("no campaign output on this machine")
    fake = tmp_path / "gates"
    fake.mkdir()
    monkeypatch.setattr(rr, "GATES", fake)          # no tier1_allrows.json here
    monkeypatch.setattr(rr, "RUNS", runs)

    rs = rr.load("100K")
    assert rs, "no seeds loaded"
    for r in rs:
        for v in ("C1", "C2", "C3", "C4"):
            assert "click_all" in r[v]["heads"], (
                "the key must exist even with no file, so accessors do not "
                "KeyError")
            assert np.isnan(r[v]["heads"]["click_all"]["auc"])

    with pytest.raises(SystemExit) as e:
        rr.require_allrows({"100K": rs})
    msg = str(e.value)
    assert "REFUSING TO WRITE" in msg
    assert "make_report.py" in msg, \
        "the refusal must name the command that fixes it"
    for h in ("click", "install", "payer"):
        assert h in msg, "it should say which heads are missing"


def test_a_row_with_no_data_anywhere_is_dropped_not_printed_as_nan(tmp_path,
                                                                   monkeypatch):
    """The guard above is the policy; this is the mechanism underneath it.

    `view_table` must still drop an all-NaN row rather than render `nan`, both
    because a printed `nan` reads as a failed computation rather than a
    measurement not taken, and because any metric added in future gets this
    behaviour for free before anyone thinks to guard it.
    """
    rr = _tool("results_report")
    runs = ROOT / "output" / "runs"
    if not (runs / "100K").exists():
        pytest.skip("no campaign output on this machine")
    fake = tmp_path / "gates"
    fake.mkdir()
    monkeypatch.setattr(rr, "GATES", fake)
    monkeypatch.setattr(rr, "RUNS", runs)
    body = "\n".join(rr.view_table(rr.load("100K")))
    assert "nan" not in body.lower(), "an all-NaN row must be dropped"
    assert "| click AUC |" not in body, \
        "with no all-rows data the funnel row has nothing to show and goes"
    assert "| EV level |" in body, "rows that DO have data must survive"


def test_the_json_merges_by_seed_not_by_position(tmp_path, monkeypatch):
    """Pairing two lists by index is how a dropped seed becomes a silent
    mislabelling. The merge must key on the seed in the path."""
    rr = _tool("results_report")
    runs = ROOT / "output" / "runs"
    if not (runs / "100K").exists():
        pytest.skip("no campaign output on this machine")
    seeds = sorted(p.name[4:] for p in (runs / "100K").glob("seed*"))
    if len(seeds) < 2:
        pytest.skip("need two seeds")
    # give ONLY the second seed a value, and a distinctive one
    fake = tmp_path / "gates"
    fake.mkdir()
    (fake / "tier1_allrows.json").write_text(json.dumps(
        {"100K": {seeds[1]: {"C1": {"click": {"auc_all": 0.4242}}}}}),
        encoding="utf-8")
    monkeypatch.setattr(rr, "GATES", fake)
    monkeypatch.setattr(rr, "RUNS", runs)
    rs = rr.load("100K")
    got = [r["C1"]["heads"]["click_all"]["auc"] for r in rs]
    assert got[1] == 0.4242, "the value must land on the seed it names"
    assert np.isnan(got[0]), "and on no other seed"
