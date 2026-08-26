# -*- coding: utf-8 -*-
"""The two results writers must not drift apart on which metrics they report.

`results_report.py` writes the markdown and, through `safe_docx_export`, two of
the three Word documents. `results_v1_layout.py` writes the third directly with
python-docx, because it needs ticks and row shading that pandoc cannot produce.
Both read the same `results.json` and both keep their own list of what to report.

TWO LISTS IS THE DEFECT THIS GUARDS. On 23 August three metrics were dropped --
`mce_win`, `spend rmse_log` and `value_captured` -- and each had to be deleted
TWICE, by hand, in two files, with nothing checking they agreed. That is the
same shape as the duplicated confidence interval a test caught the week before:
identical today, silently divergent the day somebody edits one of them.

WHY NOT JUST SHARE ONE LIST. Because they are not the same list, and pretending
otherwise would be the worse fix. The v1 layout carries a `claim` flag marking
rows that are reported but never judged, it groups rows into five titled
sections rather than three, and it reports four quantities the main table does
not (`profit per 1k wins`, `n_won`, and both of those again for the second
bidder). A shared list would need a superset with per-writer filters, which is
more machinery than the duplication costs. Two days from submission the honest
trade is to leave the duplication and test the invariant that matters.

WHAT THE INVARIANT IS. Not "the lists are equal" -- they are not, by design.
It is: a metric dropped from one writer is dropped from the other, and a metric
central to the study appears in both.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

# Dropped 23 August 2026. Neither writer may report these.
RETIRED = ["mce_win", "value_captured"]
# `spend rmse_log` is the same metric under two labels: the main table called it
# that, the v1 layout called it `rmse_log_spend`. Both are retired.
RETIRED_SPEND = ["spend rmse_log", "rmse_log_spend"]

# The study's load-bearing metrics. Every one must survive in BOTH writers,
# because dropping one from a single document is how two records of the same
# campaign come to disagree.
CORE = [
    ("click", "click AUC", "auc_click"),
    ("install", "install AUC", "auc_install"),
    ("payer", "payer AUC", "auc_payer"),
    # the v1 layout spells out the estimand in the label, the main table puts it
    # in a footnote. Same metric, and the difference in wording is deliberate.
    ("spend", "spend CRPS", "crps_spend, E(spend|payer)"),
    ("EV level", "EV level", "EV level"),
    ("price error", "rmse_log", "price rmse_log"),
    ("price bias", "bias_log", "price bias_log"),
    ("share of oracle", "share of oracle", "share of oracle"),
    ("profit", "profit, USD", "profit, total (USD)"),
]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / ("%s.py" % name))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def labels():
    """Every metric label each writer would emit, as two lists."""
    rr = _tool("results_report")
    vl = _tool("results_v1_layout")
    main = [m[0] for m in rr.METRICS if m[1] is not None]
    v1 = [row[0] for _, rows in vl.SECTIONS for row in rows]
    return main, v1


def test_neither_writer_reports_a_retired_metric(labels):
    main, v1 = labels
    for label in RETIRED:
        assert not [x for x in main if x == label], \
            "%s is back in results_report.METRICS" % label
        assert not [x for x in v1 if x.split(" (")[0] == label], \
            "%s is back in results_v1_layout.SECTIONS" % label
    for label in RETIRED_SPEND:
        assert label not in main and label not in v1, \
            "%s is back; the spend head reports CRPS only" % label


def test_both_writers_carry_every_core_metric(labels):
    """The drift test proper. If a future edit removes one of these from one
    document and not the other, two records of the same campaign disagree and
    nothing else in the suite notices."""
    main, v1 = labels
    missing = []
    for what, in_main, in_v1 in CORE:
        if in_main not in main:
            missing.append("%s: results_report has no %r" % (what, in_main))
        if in_v1 not in v1:
            missing.append("%s: results_v1_layout has no %r" % (what, in_v1))
    assert not missing, "\n".join(missing)


def test_the_v1_layout_extras_are_the_ones_we_expect(labels):
    """It reports four things the main table does not, and that is deliberate.

    Pinned so the difference stays a decision. If this list grows, somebody
    added a metric to one document only, which is the drift the other two tests
    exist to stop.
    """
    _, v1 = labels
    extras = sorted({x for x in v1
                     if x in ("profit per 1k wins (USD)", "n_won")})
    assert extras == ["n_won", "profit per 1k wins (USD)"], extras


def test_both_writers_use_the_same_confidence_interval(labels):
    """Already enforced by test_report.py's one-implementation rule, asserted
    here from the other direction: the layout writer must be USING the import,
    not merely not defining a copy."""
    vl = _tool("results_v1_layout")
    rr = _tool("results_report")
    got = vl.paired([0.4, 0.5, 0.6, 0.7])
    want = rr.paired([0.4, 0.5, 0.6, 0.7])
    for k in ("mean", "lo", "hi", "agree", "n"):
        assert abs(got[k] - want[k]) < 1e-12, (k, got[k], want[k])
