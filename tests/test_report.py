# -*- coding: utf-8 -*-
"""Step 8: one reporting script, thirty cells, and v2 left alone.

Four scripts held four copies of the confidence-interval arithmetic, two meanings
of `ev_ratio` and two labels for profit-over-wins. They are replaced by one and
DELETED rather than left beside it, because leaving them is what let those
divergences appear.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def load_report():
    p = ROOT / "tools" / "results_report.py"
    spec = importlib.util.spec_from_file_location("results_report", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def R():
    return load_report()


def fake(seed_offset=0, n=10):
    """Ten synthetic seeds, enough shape for the aggregator to chew on."""
    rng = np.random.default_rng(7 + seed_offset)
    out = []
    for _ in range(n):
        r = {}
        for i, v in enumerate(["C1", "C2", "C3", "C4"]):
            base = 40.0 + 6.0 * (i % 2) + rng.normal(0, 1)
            e = {"learned": {"profit": base, "value_captured": 0.19 + 0.01 * i,
                             "value_vs_oracle": 0.20 + 0.01 * i, "wins": 1400 + 10 * i,
                             "profit_per_1k_wins": base / 1.4},
                 "oracle": {"profit": 246.0, "value_captured": 0.977,
                            "value_vs_oracle": 1.0, "wins": 1450,
                            "profit_per_1k_wins": 170.0}}
            r[v] = {
                "economics": e, "economics_price": json.loads(json.dumps(e)),
                "ev": {"ratio": 0.44, "spearman": 0.08, "spearman_active": 0.1},
                "heads": {
                    "click": {"auc": 0.55 + 0.01 * i}, "install": {"auc": 0.59},
                    "payer": {"auc": 0.55}, "spend": {"crps": 12.0},
                    "win": {"auc": 0.81 + 0.001 * i},
                    "win_price": {"auc": 0.79 + 0.002 * i, "sigma": 0.9},
                    "price": {"rmse_log": 1.2 - 0.1 * i},
                },
                "sigma_sweep": {
                    "lo": {"sigma": 0.675, "auc": 0.79 + 0.002 * i,
                           "economics": {"profit": base * 0.98}},
                    "hi": {"sigma": 1.35, "auc": 0.79 + 0.002 * i,
                           "economics": {"profit": base * 1.02}},
                },
            }
        out.append(r)
    return out


# ------------------------------------------------------------ the arithmetic

def _code_only(path):
    """The file's CODE, with comments and docstrings removed.

    WHY THIS IS NOT PEDANTRY. The test below forbids a second copy of the
    interval formula, and it used to search the raw file text. That made a
    comment ABOUT the convention indistinguishable from a use of it, so
    tools/shap_aggregate.py failed the guard for a docstring explaining why it
    deliberately does NOT compute a 95 percent interval. A test that a file
    cannot pass by being right is a test that gets weakened rather than obeyed.

    Docstrings are dropped as well as comments, since a module docstring quoting
    the formula is the same false positive one indent further in.
    """
    import io
    import tokenize
    out, prev_was_definition = [], True
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev_was_definition:
                continue                      # a docstring, not a value
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENCODING):
                continue
            if tok.type == tokenize.NAME and tok.string in ("def", "class"):
                prev_was_definition = False
            elif tok.type == tokenize.OP and tok.string == ":":
                prev_was_definition = True
            else:
                prev_was_definition = tok.type == tokenize.ENCODING
            out.append(tok.string)
    return " ".join(out)


def test_there_is_exactly_one_confidence_interval_implementation():
    """Four copies is how two of them came to disagree.

    Tested as "no OTHER module computes an interval", which is the property that
    matters. Counting the literal inside `results_report.py` would not: it
    appears twice on the one line that forms both ends of the interval.
    """
    mine = ROOT / "tools" / "results_report.py"
    assert "def paired(" in mine.read_text(encoding="utf-8")
    others = []
    for sub in ("src", "tools"):
        for f in sorted((ROOT / sub).rglob("*.py")):
            if f == mine or "__pycache__" in f.parts:
                continue
            if "1.96" in _code_only(f):
                others.append(str(f.relative_to(ROOT)).replace("\\", "/"))
    assert not others, ("a second interval formula lives in %s; one of them will "
                        "drift" % ", ".join(others))


def test_the_four_superseded_scripts_are_gone():
    for name in ("stage6_aggregate.py", "merged_results_table.py",
                 "results_1m.py", "results_across_seeds.py"):
        assert not (ROOT / "tools" / name).exists(), \
            "%s still exists; it holds a second copy of the arithmetic" % name


def test_a_verdict_needs_both_a_clear_interval_and_every_seed_agreeing(R):
    """An interval clear of zero on a mean four seeds disagree with is a
    different claim from one all ten agree on, and only the count separates
    them."""
    assert R.verdict(R.paired([1.0] * 10)) == "POSITIVE"
    assert R.verdict(R.paired([-1.0] * 10)) == "NEGATIVE"
    assert R.verdict(R.paired([1, 1, 1, 1, 1, 1, -1, -1, 2, 3])) == "NULL"
    assert R.verdict(R.paired([0.001, -0.001] * 5)) == "NULL"
    assert R.verdict(R.paired([1.0])) == "n/a", "one seed forms no interval"


# ------------------------------------------------------------ the ratio floor

def test_the_ratio_floor_is_relative_and_would_not_blank_the_1m_headline(R):
    """Step 8e's open item, resolved 23 August 2026.

    The old floor was an absolute 1000, which worked only while profits were
    still in eCPM-sums. After the units fix C1's profit reads about 38 at 100K
    and 200 at 1M, so an absolute floor would silently blank BOTH — including
    the 1M headline — and the table would read n/a with nothing wrong.
    """
    rs = fake()
    fl = R.ratio_floor(rs)
    assert abs(fl - 2.46) < 0.01, "one percent of a 246 oracle"
    c1 = rs[0]["C1"]["economics"]["learned"]["profit"]
    assert c1 > fl, "a real 100K profit must clear the floor, not be blanked by it"
    assert 1000.0 > c1, "and it would NOT have cleared the old absolute 1000"


def test_the_floor_still_blanks_a_base_that_is_actually_noise(R):
    rs = fake()
    for r in rs:
        r["C1"]["economics"]["learned"]["profit"] = 0.01
    fl = R.ratio_floor(rs)
    f = R.pct(R.econ("C2", "profit"), R.econ("C1", "profit"), fl)
    assert not np.isfinite(f(rs[0])), "a base of 0.01 against a 246 oracle is noise"


# ------------------------------------------------------------- the 30 cells

def test_the_agreement_table_has_thirty_cells(R):
    cells, rows, sweep, fl = R.agreement(fake(), "1M")
    assert len(cells) == 30, sorted(cells)
    assert len(rows) == 30
    bidders = {c.split("|")[1] for c in cells}
    metrics = {c.split("|")[2] for c in cells}
    contrasts = {c.split("|")[0] for c in cells}
    assert bidders == {"clf", "aft"}
    assert metrics == {"profit", "value_captured", "value_vs_oracle", "wins", "auc_win"}
    assert contrasts == {"C2-C1", "C3-C1", "C4-C2"}


def test_the_sigma_sweep_rides_along(R):
    _, _, sweep, _ = R.agreement(fake(), "1M")
    assert set(sweep) == {"C2-C1|lo", "C2-C1|hi", "C3-C1|lo", "C3-C1|hi",
                          "C4-C2|lo", "C4-C2|hi"}
    for v in sweep.values():
        assert set(v) == {"profit", "auc_win"}


def test_every_profit_contrast_is_against_its_own_bidders_baseline(R):
    """The two heads win different impressions, so their levels are not the same
    quantity. v1 reported an AFT bidder showing higher profit in every cell and
    it was evidence of nothing."""
    src = (ROOT / "tools" / "results_report.py").read_text(encoding="utf-8")
    assert "econ(a, \"profit\", k), econ(b, \"profit\", k)" in src, \
        "a profit contrast must read the same `kind` on both sides"
    cells, _, _, _ = R.agreement(fake(), "1M")
    # clf and aft profits are equal in the fixture, so identical bases must give
    # identical contrasts; a crossed pairing would not
    assert cells["C3-C1|clf|profit"]["mean"] == cells["C3-C1|aft|profit"]["mean"]


# ------------------------------------------------------------- what it writes

def test_it_refuses_to_write_a_v2_filename(R, tmp_path):
    """v2 is published at tag v2.0.0 and must stay quotable. The writer
    overwrites unconditionally, so a collision would be silent."""
    for name in ("V2_Results.md", "V2_Results_Merged_CI.md"):
        with pytest.raises(RuntimeError, match="published"):
            R.write(tmp_path / name, ["x"])
    R.write(tmp_path / "V2.2_Results.md", ["ok"])       # the dotted name is fine


def test_it_refuses_when_a_requested_scale_has_no_seeds(R):
    assert R.empty_scales({"100K": [1], "1M": []}) == ["1M"]
    assert R.empty_scales({"100K": [1]}) == [], \
        "a scale that was not asked for must not trigger a refusal"


def test_spearman_appears_in_no_generated_v2_2_document():
    """8e: `ev_spearman` leaves every generated document and stays in
    results.json as a stored diagnostic. Its removal was started in v2 and left
    half-done, still appearing in V2_Results.md's main table."""
    for p in [ROOT / "docs" / "V2.2_Results.md",
              ROOT / "docs" / "V2.2_Results_10M_n10_CI.md"] + \
             list((ROOT / "docs" / "gates").glob("agreement_*.md")):
        if p.exists():
            assert "spearman" not in p.read_text(encoding="utf-8").lower(), p


def test_the_v2_results_files_are_still_on_disk_and_untouched():
    """Never regenerated, never deleted."""
    for name in ("V2_Results.md", "V2_Results_Merged_CI.md"):
        assert (ROOT / "docs" / name).exists(), name


# ---------------------------------------------------------------- the renames

def test_economics_reports_value_captured_and_not_ev_ratio():
    from t9v2.train import metrics as M
    r = M.economics(np.array([5.0, 5.0]), None, np.array([10.0, 10.0]),
                    np.array([1.0, 9.0]), np.zeros(2))
    assert "value_captured" in r and "ev_ratio" not in r
    assert "profit_per_1k_wins" in r


def test_profit_per_1k_wins_is_nan_when_nothing_was_won():
    from t9v2.train import metrics as M
    r = M.economics(np.array([1.0]), None, np.array([10.0]),
                    np.array([99.0]), np.array([99.0]))
    assert r["wins"] == 0 and not np.isfinite(r["profit_per_1k_wins"])


def test_profit_per_1k_wins_divides_by_wins_not_impressions():
    """The old name `profit CPM` hid its own denominator and invited comparison
    with `value_captured`, whose denominator is all rows."""
    from t9v2.train import metrics as M
    r = M.economics(np.array([5.0, 5.0, 5.0]), None, np.array([100.0] * 3),
                    np.array([1.0, 1.0, 99.0]), np.zeros(3))
    assert r["wins"] == 2
    assert abs(r["profit_per_1k_wins"] - r["profit"] / 2 * 1000) < 1e-9
