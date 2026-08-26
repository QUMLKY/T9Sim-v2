# -*- coding: utf-8 -*-
"""The driver's three guards: one fit, every artifact, and the right design.

These run the REAL `run_one` at 2,000 rows rather than a stand-in. The thing
being tested is its control flow — that the fit happens once and feeds both the
bundle and the eval pass, that the results file records which generator produced
the data, and that a seed missing any of it does not read as done — and a
stand-in for `run_one` would test the stand-in.

2,000 rows because the wiring is what is under test, not the numbers. The spend
head falls back to an intercept at this scale, which is a path 100K never takes,
so the bundle round trip is exercised on its awkward case for free.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from t9v2 import campaign as C
from t9v2 import generate as G

warnings.filterwarnings("ignore")

ROWS = 2000


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One real `run_one`, with the output tree redirected and the scale cut.

    Returns (output, runs, counts). `counts` is how many times each tier was
    FITTED across the whole call, which is the single-pass claim as a number.
    """
    root = tmp_path_factory.mktemp("campaign")
    from t9v2.train.tier1 import Tier1
    from t9v2.train.tier2 import Tier2

    counts = {"tier1": 0, "tier2": 0}
    real_gen, t1_fit, t2_fit = G.generate, Tier1.fit, Tier2.fit

    def small_generate(scale="100K", seed=20250, **kw):
        kw.pop("n_rows", None)
        kw.pop("out", None)
        return real_gen(scale=scale, seed=seed, n_rows=ROWS,
                        out=str(C.master_path(scale, seed)), **kw)

    def count1(self, *a, **k):
        counts["tier1"] += 1
        return t1_fit(self, *a, **k)

    def count2(self, *a, **k):
        counts["tier2"] += 1
        return t2_fit(self, *a, **k)

    old_out, old_runs = C.OUTPUT, C.RUNS
    C.OUTPUT, C.RUNS = root / "output", root / "output" / "runs"
    C.OUTPUT.mkdir(parents=True)
    G.generate, Tier1.fit, Tier2.fit = small_generate, count1, count2
    try:
        C.run_one("100K", 20250, quiet=True)
    finally:
        keep_out, keep_runs = C.OUTPUT, C.RUNS
        G.generate, Tier1.fit, Tier2.fit = real_gen, t1_fit, t2_fit
        C.OUTPUT, C.RUNS = old_out, old_runs
    return keep_out, keep_runs, counts


@pytest.fixture
def at(ran, monkeypatch):
    """The same run, with the module globals pointed back at it for one test."""
    out, runs, _ = ran
    monkeypatch.setattr(C, "OUTPUT", out)
    monkeypatch.setattr(C, "RUNS", runs)
    return runs / "100K" / "seed20250"


# --------------------------------------------------------------- one fit only

def test_each_view_is_fitted_exactly_once(ran):
    """Four views, four fits of each tier, and the eval pass adds none.

    The alternative this rules out is a second fit to produce the per-row eval
    file. It would not reproduce the first — early stopping lands on a different
    iteration — so `results.json` and the eval files would describe two
    different models while reading as one run.
    """
    _, _, counts = ran
    assert counts == {"tier1": 4, "tier2": 4}


def test_the_eval_pass_reads_the_frozen_bundle(at):
    """Evidence that the eval file came from the artifact, not from memory.

    The bundle verifies its own sha256 on load, so an eval file that exists at
    all was scored by a model that reloaded byte identically.
    """
    import pandas as pd
    from t9v2 import bundle as BU
    for v in C.VIEWS:
        BU.load_bundle(at / "bundle" / v)          # raises if a hash moved
        d = pd.read_parquet(at / "eval" / ("%s.parquet" % v))
        assert len(d) > 0 and "p_win_at_recommended" in d.columns


# ------------------------------------------------------------ what is on disk

def test_a_finished_seed_is_complete(at):
    fp = C.current_fingerprint()
    assert C.why_incomplete("100K", 20250, fp) is None
    for f in C.artifacts("100K", 20250):
        assert f.exists(), f


def test_results_json_records_the_design_that_made_the_data(at):
    r = json.loads((at / "results.json").read_text(encoding="utf-8"))
    assert r["design_fingerprint"] == C.current_fingerprint()
    man = json.loads(C.master_path("100K", 20250).with_suffix(".manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["fingerprint"] == r["design_fingerprint"], \
        "the data and the results it produced must name the same design"


# --------------------------------------------------------- the staleness guard

def test_a_seed_from_a_superseded_design_is_not_complete(at):
    """The failure that does not announce itself.

    Nothing about these files is broken. Thirty of them, finished before a
    generator change, would be skipped as done and reported as the campaign.
    """
    p = at / "results.json"
    keep = p.read_text(encoding="utf-8")
    r = json.loads(keep)
    r["design_fingerprint"] = "0000000000000000"
    p.write_text(json.dumps(r), encoding="utf-8")
    try:
        why = C.why_incomplete("100K", 20250, C.current_fingerprint())
        assert why and why.startswith("design 0000000000000000, current is ")
    finally:
        p.write_text(keep, encoding="utf-8")


def test_results_predating_the_fingerprint_are_not_complete(at):
    """v2's thirty committed results files have no such key, and must not pass."""
    p = at / "results.json"
    keep = p.read_text(encoding="utf-8")
    r = json.loads(keep)
    del r["design_fingerprint"]
    p.write_text(json.dumps(r), encoding="utf-8")
    try:
        assert C.why_incomplete("100K", 20250, C.current_fingerprint()) \
            == "results.json predates design fingerprinting"
    finally:
        p.write_text(keep, encoding="utf-8")


def test_the_fingerprint_argument_cannot_be_forgotten():
    """No default. A check a caller can omit is a check that gets omitted."""
    with pytest.raises(TypeError):
        C.is_complete("100K", 20250)


# --------------------------------------------------------- a missing artifact

@pytest.mark.parametrize("rel", ["eval/C3.parquet", "eval/manifest.json",
                                 "bundle/C2/manifest.json"])
def test_a_missing_artifact_makes_the_seed_incomplete(at, rel):
    """Remove any one of them and the seed stops counting.

    This is the enforcement the prose rule did not provide: `run_arm1.py`
    removed the masters behind the `v2-corrected-1M` results and nothing said
    so. Retention is now a property of the completeness test rather than of
    whoever reads the document.
    """
    f = at / rel
    keep = f.read_bytes()
    f.unlink()
    try:
        why = C.why_incomplete("100K", 20250, C.current_fingerprint())
        assert why and Path(why).name == f.name, why
    finally:
        f.write_bytes(keep)
    assert C.why_incomplete("100K", 20250, C.current_fingerprint()) is None


def test_a_missing_master_makes_the_seed_incomplete(at):
    """The parquet is the artifact that was actually lost, so it is named here."""
    f = C.master_path("100K", 20250)
    held = f.with_suffix(".held")
    f.rename(held)
    try:
        why = C.why_incomplete("100K", 20250, C.current_fingerprint())
        assert why and "t9v2_100K_seed20250.parquet" in why
    finally:
        held.rename(f)


# --------------------------------------------------- the aggregator's refusal

def load_aggregator():
    # `results_report.py` since step 8; `stage6_aggregate.py` was one of the four
    # it replaced and is deleted, because leaving a second copy of the interval
    # arithmetic beside it is how the copies came to disagree.
    p = Path(C.__file__).resolve().parents[2] / "tools" / "results_report.py"
    spec = importlib.util.spec_from_file_location("results_report", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_aggregator_refuses_rather_than_overwrite_with_nothing(tmp_path, capsys):
    """Zero seeds is a stop, not a table of n/a.

    Unguarded, every mean is taken over an empty list, `fmt` renders nan as
    "n/a", and `docs/V2_Results.md` — a committed result — is replaced by a
    complete-looking document of dashes. Gate 6 does fail, but the file it was
    reporting on is already gone.
    """
    agg = load_aggregator()
    agg.RUNS = tmp_path / "runs"
    agg.ROOT = tmp_path                      # nothing may be written here
    agg.DOCS = tmp_path / "docs"
    agg.GATES = tmp_path / "docs" / "gates"
    assert agg.main([]) == 2
    assert "REFUSING TO WRITE" in capsys.readouterr().out
    assert not (tmp_path / "docs").exists(), "it wrote something anyway"


def test_the_aggregator_names_which_scales_are_empty():
    agg = load_aggregator()
    assert agg.empty_scales({"100K": [1], "1M": [], "10M": []}) == ["1M", "10M"]
    assert agg.empty_scales({s: [1] for s in agg.SCALES}) == []
