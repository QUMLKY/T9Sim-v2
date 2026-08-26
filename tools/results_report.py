# -*- coding: utf-8 -*-
"""The one reporting script. Replaces four, and that is the point of it.

    python tools/results_report.py --scale all

WHAT IT WRITES

    docs/V2.2_Results.md                  every scale, every contrast
    docs/V2.2_Results_10M_n10_CI.md       the v1-layout tables at 10M
    docs/gates/agreement_<scale>.{md,json}  the two-bidder table, 30 cells
    docs/gates/gate6.md                   the aggregation gate
    docs/gates/stage6.json                the machine-readable contrasts

WHY ONE SCRIPT. `stage6_aggregate.py`, `merged_results_table.py`, `results_1m.py`
and `results_across_seeds.py` held FOUR copies of the confidence-interval
arithmetic, two meanings of `ev_ratio` and two labels for profit-over-wins.
Leaving them beside a fifth is what let those divergences appear in the first
place, so they are deleted rather than kept. There is one `paired()` here and
everything goes through it.

V2 IS NEVER OVERWRITTEN. `docs/V2_Results.md` and `docs/V2_Results_Merged_CI.md`
are published at tag `v2.0.0` and their numbers must stay quotable. The writer
raises if any output basename collides with one of them, because it overwrites
unconditionally and a collision would be silent.

V2.2 REPLACES V2's HEADLINE WHOLESALE, not selectively. The floor fix moves C1's
win AUC, the units fix divides every profit by a thousand, and the new ladder
moves every argmax, so a v2 number set beside a v2.2 number compares two
simulators rather than two data layers.

TWO BIDDERS, NEVER COMPARED BY LEVEL. The classifier and the price head choose
different bids, so they win different auctions and their profits are not the
same quantity. Every profit contrast here is a percentage of THAT BIDDER'S OWN
baseline. v1 reported an AFT bidder showing higher profit in every cell and it
was evidence of nothing.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

# Only when RUN, never on import. Rebinding stdout at import time breaks any
# caller that captured it -- pytest, most obviously, which then fails every
# test in the module with "I/O operation on closed file" at teardown.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "output" / "runs"
DOCS = ROOT / "docs"
GATES = DOCS / "gates"
SCALES = ["100K", "1M", "10M"]
VIEWS = ["C1", "C2", "C3", "C4"]
LAYERS = {"C1": "DSP only", "C2": "DSP + MMP", "C3": "DSP + SSP", "C4": "all three"}

# v2's published files. Colliding with one of these is an error, not a warning.
FROZEN = {"V2_Results.md", "V2_Results_Merged_CI.md"}

# the two bidders, and which keys each reads
BIDDERS = [("clf", "the win classifier", "economics", "win"),
           ("aft", "the price head", "economics_price", "win_price")]

CONTRASTS = [("C2", "C1", "MMP"), ("C3", "C1", "SSP"), ("C4", "C2", "SSP given MMP")]

# THE REPORTED BIDDER SET, and `both` stays the default so nothing already
# written changes shape by accident.
#
# Ken's decision, 24 August 2026: the dissertation reports the AFT price head
# alone. The win classifier is not deleted, not untrained and not removed from
# `results.json` -- it is demoted out of the reported tables. That distinction
# is the whole design of this flag. Dropping a head from a document is a
# reporting choice and costs nothing; dropping it from the campaign would cost
# a six-hour re-run and would throw away the classifier's own SSP null, which
# is a result in its own right and the thing the two-head design was built to
# establish.
#
# SO THE GATES KEEP BOTH AND THE REPORT SHOWS ONE. `agreement_*.md`,
# `stage6.json` and `gate6.md` are the record and are written from every
# bidder whatever this is set to. Only the results DOCUMENT narrows, and it
# narrows into a file of its own so the two-bidder version stays quotable.
BIDDER_SETS = {"both": ("clf", "aft"), "aft": ("aft",)}

# The METRICS section belonging to each bidder, dropped whole when that bidder
# is not reported. The head and the bidder built on it are one story, which is
# why they share a section.
BIDDER_SECTION = {"clf": "**Tier 2, CLF**", "aft": "**Tier 2, AFT**"}


# --------------------------------------------------------------------- loading

def load(scale, roas=None):
    """Every seed's `views` block, with the all-rows Tier-1 AUCs merged in.

    THE SECOND POPULATION IS READ, NEVER RECOMPUTED. `tools/tier1_allrows.py`
    writes it, because computing it means reading ten 10M masters off disk and a
    report regeneration has to stay cheap. Absent file, absent scale or absent
    seed all leave NaN, and `view_table` drops a row that is NaN everywhere, so
    this file still runs on a machine that has never run that tool.

    MERGED BY SEED, not by position. The two sources glob the same directory
    today, but pairing two lists by index is how a dropped seed becomes a silent
    mislabelling rather than an error.
    """
    extra, seeds = {}, []
    p = GATES / "tier1_allrows.json"
    if p.exists():
        extra = json.loads(p.read_text(encoding="utf-8")).get(scale, {})
    out = []
    for p in sorted((RUNS / scale).glob("seed*/results.json")):
        views = json.loads(p.read_text(encoding="utf-8"))["views"]
        got = extra.get(p.parent.name[4:], {})
        orc = got.get("oracle", {})
        for v, blk in views.items():
            for h in ("click", "install", "payer"):
                blk["heads"]["%s_all" % h] = {
                    "auc": got.get(v, {}).get(h, {}).get("auc_all", float("nan"))}
                # the Bayes ceiling, the same for every view because it does
                # not depend on one. Stored per view anyway so the oracle
                # accessor reads like every other accessor in METRICS.
                blk["heads"]["%s_oracle" % h] = {
                    "auc": orc.get(h, {}).get("auc", float("nan"))}
        out.append(views)
        seeds.append(int(p.parent.name[4:]))
    if roas is None:
        return out
    # THE BIDDER ROWS COME FROM THE SWEEP, THE REST FROM THE RUNS. One
    # implementation, imported rather than copied, for the same reason the
    # interval arithmetic is imported: two copies of this would drift and
    # nothing would flag the day they did. See results_v1_layout.overlay_roas
    # for why re-scoring in place was not the route taken.
    import results_v1_layout as _L
    return _L.overlay_roas(out, seeds, scale, roas)


def empty_scales(data):
    """Scales with no seeds. A reason to write nothing, not a table of n/a.

    THE FAILURE IS AN OVERWRITE, NOT A CRASH. Run against a fresh clone and every
    mean is taken over an empty list, nan renders as "n/a", and a
    complete-looking document of dashes lands on top of the real one.
    """
    # over the scales ASKED FOR, not over all three. `--scale 100K` must not
    # refuse because 10M has not been run yet.
    return [s for s in SCALES if s in data and not data.get(s)]


# ------------------------------------------------------------- the arithmetic

def paired(vals):
    """mean, 95 percent interval and sign agreement over PER-SEED differences.

    THE ONLY IMPLEMENTATION IN THE PROJECT, deliberately. Four copies of this is
    how two of them came to disagree.

    Every contrast is formed WITHIN a seed. The four views share one master, one
    temporal split and one test set, so differencing inside a seed removes the
    variation common to both sides. On the 10M profit contrast the unpaired
    interval is about four times wider and spans zero; the paired one does not.
    """
    d = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "agree": 0}
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "agree": int((np.sign(d) == np.sign(m)).sum())}


def verdict(st):
    """POSITIVE, NEGATIVE or NULL.

    Two conditions, and the second is the project's actual bar: the interval must
    clear zero AND every seed must agree in direction. An interval clear of zero
    on a mean that four seeds disagree with is a different claim from one all ten
    agree on, and only the count separates them.
    """
    if not np.isfinite(st["mean"]) or st["n"] < 2:
        return "n/a"
    if st["agree"] < st["n"]:
        return "NULL"
    if st["lo"] > 0:
        return "POSITIVE"
    if st["hi"] < 0:
        return "NEGATIVE"
    return "NULL"


def ratio_floor(rs):
    """A profit base below this is noise, not a base. ONE PERCENT OF ORACLE.

    RELATIVE, NOT ABSOLUTE, and the change is step 8e's open item resolved. The
    old floor was a literal 1000, which worked only while profits were still in
    eCPM-sums. The units fix divides every profit by a thousand, and C1's profit
    then reads about 38 at 100K and 200 at 1M, so an absolute 1000 would silently
    blank both — INCLUDING the 1M headline — and the table would read n/a with
    nothing wrong.

    One percent of that seed's own oracle profit cannot rot when units change,
    scales with the data, and stays far below any real base. The oracle's profit
    is large and positive in every seed, which is why it is the denominator
    everywhere else in this file too.
    """
    o = [r["C1"]["economics"]["oracle"]["profit"] for r in rs]
    return 0.01 * float(np.mean(o))


# ------------------------------------------------------------------ accessors

def econ(view, key, kind="economics"):
    return lambda r: r[view][kind]["learned"][key]


def head(view, key, h="win"):
    return lambda r: r[view]["heads"][h][key]


def ev(view, key):
    return lambda r: r[view]["ev"][key]


def diff(a, b):
    return lambda r: a(r) - b(r)


def pct(a, b, floor):
    """Relative change against a bidder's OWN baseline, guarded by `floor`."""
    def f(r):
        base = b(r)
        return (a(r) / base - 1.0) if abs(base) > floor else float("nan")
    return f


def over_oracle(a, b):
    """A paired difference over the SAME seed's oracle profit.

    The scale-free way to carry a profit gap across 100K, 1M and 10M. The
    oracle's profit is large and positive in every seed, so it cannot blow up.
    """
    return lambda r: (a(r) - b(r)) / r["C1"]["economics"]["oracle"]["profit"]


def fmt(x, spec):
    if not np.isfinite(x):
        return "n/a"
    return ("%+.1f%%" % (100.0 * x)) if spec == "pct" else (spec % x)


# ------------------------------------------------------- 8a, the two-bidder table

CELL_METRICS = [
    ("profit", "pct", lambda a, b, k, fl: pct(econ(a, "profit", k), econ(b, "profit", k), fl)),
    ("value_captured", "%+.4f", lambda a, b, k, fl: diff(econ(a, "value_captured", k),
                                                         econ(b, "value_captured", k))),
    ("value_vs_oracle", "%+.4f", lambda a, b, k, fl: diff(econ(a, "value_vs_oracle", k),
                                                          econ(b, "value_vs_oracle", k))),
    ("wins", "%+.0f", lambda a, b, k, fl: diff(econ(a, "wins", k), econ(b, "wins", k))),
]


def agreement(rs, scale):
    """3 contrasts x 5 metrics x 2 bidders = 30 cells, plus the sigma sweep.

    ALWAYS WRITTEN, AND THE BUILD NEVER FAILS ON A DISAGREEMENT. A finding must
    never be able to suppress its own evidence. Read under step 7d: both heads
    always reported, disagreements described rather than suppressed, no head
    nominated primary.
    """
    fl = ratio_floor(rs)
    cells, rows = {}, []
    for a, b, layer in CONTRASTS:
        for key, bid_name, ekey, hkey in BIDDERS:
            for name, spec, mk in CELL_METRICS:
                st = paired([mk(a, b, ekey, fl)(r) for r in rs])
                cid = "%s-%s|%s|%s" % (a, b, key, name)
                cells[cid] = dict(st, verdict=verdict(st), spec=spec)
                rows.append((layer, "%s-%s" % (a, b), key, name, st, spec))
            st = paired([diff(head(a, "auc", hkey), head(b, "auc", hkey))(r) for r in rs])
            cid = "%s-%s|%s|auc_win" % (a, b, key)
            cells[cid] = dict(st, verdict=verdict(st), spec="%+.4f")
            rows.append((layer, "%s-%s" % (a, b), key, "auc_win", st, "%+.4f"))

    # the sigma sweep rides along, per step 7d. No refit: sigma enters the curve
    # and the scorer, never the fit, so these come straight out of results.json.
    sweep = {}
    for a, b, _ in CONTRASTS:
        for tag in ("lo", "hi"):
            try:
                pr = paired([(r[a]["sigma_sweep"][tag]["economics"]["profit"]
                              / r[b]["sigma_sweep"][tag]["economics"]["profit"] - 1.0)
                             if abs(r[b]["sigma_sweep"][tag]["economics"]["profit"]) > fl
                             else float("nan") for r in rs])
                au = paired([r[a]["sigma_sweep"][tag]["auc"]
                             - r[b]["sigma_sweep"][tag]["auc"] for r in rs])
            except KeyError:
                continue
            sweep["%s-%s|%s" % (a, b, tag)] = {
                "profit": dict(pr, verdict=verdict(pr)),
                "auc_win": dict(au, verdict=verdict(au)),
            }
    return cells, rows, sweep, fl


def agreement_md(scale, rows, sweep, fl, n):
    L = ["# Two-bidder agreement, %s" % scale, "",
         "*Generated by `tools/results_report.py`. This file is written, never "
         "edited, and it is written whatever it says: a finding must not be able "
         "to suppress its own evidence.*", "",
         "Three contrasts x five metrics x two bidders. `clf` is the win "
         "classifier, `aft` is the price head. **Their LEVELS are never "
         "compared** — the two choose different bids and so win different "
         "auctions — so every profit figure is a percentage of that bidder's "
         "OWN baseline.", "",
         "**No head is primary.** The classifier's contrast can move only through "
         "FEATURES, since `won` is visible in all four views. The price head's "
         "can move only through LABELS, since its features are identical in all "
         "four. A disagreement between them is therefore not noise cancelling a "
         "claim; it locates where the value of an SSP integration lies.", "",
         "n = %d seeds. Profit base floor %.3f, one percent of mean oracle profit."
         % (n, fl), "",
         "| Layer | Contrast | Bidder | Metric | Mean | 95% CI | Agree | Verdict |",
         "|---|---|---|---|---:|---|---:|---|"]
    for layer, con, bid, name, st, spec in rows:
        L.append("| %s | %s | `%s` | %s | %s | [%s, %s] | %d/%d | %s |"
                 % (layer, con, bid, name, fmt(st["mean"], spec), fmt(st["lo"], spec),
                    fmt(st["hi"], spec), st["agree"], st["n"], verdict(st)))

    if sweep:
        L += ["", "## The price head under a different sigma", "",
              "Step 7d's guard, and the reason the price head does not simply "
              "get quoted. Sigma enters the win curve and the scorer, and the "
              "fit only through C1 and C2, whose labels are intervals -- so a "
              "sweep of the curve costs no refit. **A claim resting on this "
              "head must hold at all three.** Two cautions. `auc_win` CANNOT "
              "move with sigma, which divides a log ratio and so cannot change "
              "a ranking, and its rows below are an identity rather than "
              "evidence. And each view sweeps around ITS OWN sigma, which is "
              "the fitted value in C1 and C2 and the placeholder 1.0 in C3 and "
              "C4; plan section 7b asks for one value across all four. Queued "
              "as Q4.", "",
              "| Contrast | Metric | at 0.75 sigma | at sigma | at 1.5 sigma |",
              "|---|---|---|---|---|"]
        for a, b, _ in CONTRASTS:
            for metric, spec in (("profit", "pct"), ("auc_win", "%+.4f")):
                lo = sweep.get("%s-%s|lo" % (a, b), {}).get(metric)
                hi = sweep.get("%s-%s|hi" % (a, b), {}).get(metric)
                if lo is None or hi is None:
                    continue
                mid = None
                L.append("| %s-%s | %s | %s %s | %s | %s %s |"
                         % (a, b, metric, fmt(lo["mean"], spec), lo["verdict"],
                            "see above", fmt(hi["mean"], spec), hi["verdict"]))
    return L


# --------------------------------------------------------------- the main tables

# Every metric in the per-view table, with the DIRECTION that makes one number
# better than another. "Best" is not "highest" for three of them, and getting
# that wrong would bold the worst cell:
#
#   min    `rmse_log` and `spend CRPS` are errors. Less is better.
#   near1  `EV level` is mean(predicted) over mean(true), a BIAS ratio. Perfect
#          is 1.0, so a view that over-predicts by a third is exactly as wrong
#          as one that under-predicts by a quarter, and neither "highest" nor
#          "lowest" says so.
#
# ORACLE COLUMN, and it is mostly empty on purpose. The oracle is a POLICY with
# true value and the true win rule, not a censoring condition, so it predicts no
# funnel, carries neither win head nor price head, and has no spend model. Those
# rows get an em dash rather than a number, because a blank would read as a
# missing measurement and a zero would be a lie. Two rows are 1.0 BY DEFINITION
# -- its EV level, since its prediction IS the truth, and its share of itself.
#
# (label, accessor factory, printf, direction, oracle)
#   oracle: None      the quantity does not exist for it
#           "def1"    1.0 by definition
#           callable  read from economics.oracle
def eco(view, key, kind):
    return lambda r: r[view][kind]["learned"][key]


METRICS = [
    # THREE SECTIONS, one per model that produces numbers. Each Tier-2 head owns
    # its OWN economics: a head and the bidder built on it are one story, and
    # splitting them put the classifier's profit four rows from the classifier's
    # AUC while the price head's sat somewhere else again.
    #
    # The two Tier-2 sections must never be read ACROSS. The curves choose
    # different bids and win different auctions, so only down-a-section and
    # along-a-row comparisons mean anything.
    # THE FUNNEL AUCs ARE SCORED ON EVERY TEST ROW, and the figure `runner.py`
    # records is deliberately NOT the one reported. That one scores each head
    # only where its own view can SEE the label, so C1 and C3 are graded on the
    # roughly 27, 19 and 16 percent of rows they won while C2 and C4 are graded
    # on all of them. Four views, four different exams.
    #
    # BOTH WERE PRINTED FOR ONE DAY, 23 August 2026, AND THAT WAS WRONG. This is
    # a comparison table: every row carries a C2-C1 column. An own-rows row
    # therefore publishes a contrast that is about two thirds difference-in-exam
    # rather than difference-in-data-layer -- +0.0642 against +0.0220 like for
    # like -- and relies on a footnote to tell the reader not to read the number
    # beside it. A figure that must not be compared does not belong in a
    # comparison table. Ken, 23 August 2026.
    #
    # The own-rows figures are not lost. `docs/gates/tier1_allrows.md` reports
    # both populations in full with their row counts, which is the right home
    # for them: it is an audit of the scoring decision, not a results table. No
    # research question asks the own-rows question.
    ("**Tier 1**", None, None, None, None),
    # THE ORACLE HERE IS THE BAYES CEILING, not 1.0 and not a perfect ranker. It
    # is the generator's own `p_click` scored against the realised click, on the
    # same rows and under the same parent condition. What sits below it is the
    # coin flip itself and no model can have it, so without the column a reader
    # comparing C2's 0.706 against a blank cannot tell whether that is most of
    # what was available or a third of it. It is most of it.
    #
    # COMPUTED FOR v2.2 AT 10M, not lifted from v1.
    # `docs/results/full_tables_1m_n10.json` carries 0.7849 for one v1 seed at
    # 1M, and reusing it is tempting because both versions share the funnel laws
    # so the figures land close. Two simulators and two scales in one row is the
    # thing this file refuses to do everywhere else.
    ("click AUC",   lambda v: head(v, "auc", "click_all"),   "%.3f", "max",
     lambda r: r["C1"]["heads"]["click_oracle"]["auc"]),
    ("install AUC", lambda v: head(v, "auc", "install_all"), "%.3f", "max",
     lambda r: r["C1"]["heads"]["install_oracle"]["auc"]),
    ("payer AUC",   lambda v: head(v, "auc", "payer_all"),   "%.3f", "max",
     lambda r: r["C1"]["heads"]["payer_oracle"]["auc"]),
    # No auc_spend exists: the head is E(spend | payer), a regression on log
    # spend, and AUC ranks a binary label. Both figures it does produce are here.
    # ONE FIGURE FOR THE SPEND HEAD, not two. `spend rmse_log` was added on
    # 23 August beside CRPS and removed the same day. CRPS scores the whole
    # predictive distribution, which is what the head produces and what the
    # bidder consumes; rmse_log scores only where its location sits. The
    # second number answered no question the first did not, and its NAME was
    # the active harm: `spend rmse_log` in Tier 1 sat six rows above the
    # price head's `rmse_log` in Tier 2, two unrelated quantities on two
    # different scales reading as a pair. Still in `results.json` for anyone
    # who wants it.
    ("spend CRPS",     lambda v: head(v, "crps", "spend"),       "%.3f", "min",  None),
    ("EV level",       lambda v: ev(v, "ratio"),                 "%.3f", "near1", "def1"),

    ("**Tier 2, CLF**", None, None, None, None),
    ("***the head***", None, None, None, None),
    ("win AUC",        lambda v: head(v, "auc", "win"),          "%.3f", "max",  None),
    # `mce_win` DROPPED, 23 August 2026. It is the largest single-bin
    # calibration gap, so it is one bin out of ten and moves with whichever
    # bin happened to be sparsest -- 0.4563 to 0.4945 across views on a
    # metric whose companion moves 0.0529 to 0.0583. It was reported beside
    # ece to show that a small average can hide a bad region, which is a
    # real point and one this table never makes: no claim anywhere rests on
    # it. Kept in `results.json` and in the per-head record.
    ("ece_win",        lambda v: head(v, "ece_at_recommended", "win"), "%.4f", "min", None),
    ("***the bidder***", None, None, None, None),
    # `value_captured` DROPPED, 23 August 2026, and it costs nothing.
    # `share of oracle` IS `value_captured`, divided by the oracle's own
    # 0.959: the CLF bidder reads 0.232 and 0.241 at C1, and 0.232/0.959 is
    # 0.242. Two rows, one quantity, differing by a constant. The share is
    # the one worth keeping, because a reader can tell what 0.24 means and
    # cannot tell what 0.23 means without knowing the ceiling is not 1.
    #
    # THE FINDING IT CARRIED IS NOT LOST. The C3-C1 contrast on
    # `value_captured` is -0.0201 with all ten seeds agreeing, and it stays
    # in `docs/gates/agreement_*.md`, which is where contrasts are recorded.
    # This table reports levels.
    ("share of oracle", lambda v: eco(v, "value_vs_oracle", "economics"), "%.3f", "max", "def1"),
    ("profit, USD",    lambda v: eco(v, "profit", "economics"),  "%.2f", "max",
     lambda r: r["C1"]["economics"]["oracle"]["profit"]),

    ("**Tier 2, AFT**", None, None, None, None),
    ("***the head***", None, None, None, None),
    # "price" dropped: the section already says which head this is, and the only
    # thing it distinguished was a Tier 1 `spend rmse_log` that no longer
    # appears in this table.
    # "_log" stays, because it names the SCALE -- rmse on log price reads 1.315
    # where rmse in dollars would be in the hundreds, and the two are not
    # comparable. bias_log is mean(log predicted - log true): +0.455 is a
    # clearing price over-predicted by about 58 percent, 0 is unbiased.
    ("rmse_log",       lambda v: head(v, "rmse_log", "price"),   "%.3f", "min",  None),
    ("bias_log",       lambda v: head(v, "bias_log", "price"),   "%+.3f", "near0", None),
    ("win AUC",        lambda v: head(v, "auc", "win_price"),    "%.3f", "max",  None),
    ("ece_win",        lambda v: head(v, "ece_at_recommended", "win_price"), "%.4f", "min", None),
    ("***the bidder***", None, None, None, None),
    # dropped here too, see the CLF section above
    ("share of oracle", lambda v: eco(v, "value_vs_oracle", "economics_price"), "%.3f", "max", "def1"),
    ("profit, USD",    lambda v: eco(v, "profit", "economics_price"), "%.2f", "max",
     lambda r: r["C1"]["economics"]["oracle"]["profit"]),
]

PAIRS = [("C2", "C1"), ("C3", "C1"), ("C4", "C2")]


def metrics_for(bidders):
    """METRICS with the sections of unreported bidders removed.

    Sliced on the section headers rather than by tagging every row, because a
    tag on each row is a thing to forget when a metric is added. A section
    runs from its own bold header to the next one, so a new row lands in the
    right bidder by where it sits, which is also how a reader parses it.

    THE ORACLE COLUMN STILL READS `economics`, and must. The oracle is ONE
    policy shared by both blocks -- verified identical in 120 of 120 cells --
    so `r[\"C1\"][\"economics\"][\"oracle\"]` is the ceiling for the price
    bidder too. Repointing it at `economics_price` would read the same number
    from a longer path and invite somebody to `fix` the mismatch later.
    """
    drop = {BIDDER_SECTION[b] for b in BIDDER_SECTION if b not in bidders}
    out, skipping = [], False
    for row in METRICS:
        # A SECTION HEADER IS BOLD AND NOT ITALIC, the same rule
        # `tools/shade_sections.py` uses to decide what starts a colour
        # band. `**Tier 2, CLF**` and `***the head***` both begin with two
        # asterisks, so testing for that alone reads a SUB-header as the
        # start of a new section: skipping switched off at `***the head***`
        # and the classifier's rows came back orphaned under Tier 1, which
        # is what the first version of this did.
        section = (row[1] is None and row[0].startswith("**")
                   and not row[0].startswith("***"))
        if section:
            skipping = row[0] in drop
        if not skipping:
            out.append(row)
    return out


def _best(vals, direction, spec="%.3f"):
    """WHICH views are best for one metric, as a set of indices.

    A SET, NOT AN INDEX, because ties are the normal case here rather than the
    exception. MMP adds no columns the price head reads, so C1 equals C2 and C3
    equals C4 on every price-head row; bolding only the first of an exact pair
    would tell a reader C3 beat C4 when the difference column beside it says
    +0.0000. Empty when all four agree, since then nothing is best.
    """
    # SCORED ON THE DISPLAYED VALUE, not the underlying one. Bolding C3 at 0.824
    # while C1 shows 0.824 too, because the hidden fourth digit differs, makes
    # the table contradict itself: a reader sees identical numbers and one of
    # them emphasised. If the difference cannot be seen it is not a best.
    # EVERY DIRECTION IS NAMED, and an unknown one is an error rather than a
    # silent fallback. `near0` was missing here until 23 August 2026 and
    # `.get(direction, -x)` swallowed it, so `bias_log` -- the one row where
    # zero is perfect and the sign is the finding -- was scored as "highest
    # wins". The table bolded +0.455 over -0.002, calling the most biased view
    # the best one, in a table whose whole job is to say which view is best.
    # A typo in any other direction string would have done the same silently.
    KEYS = {"max": lambda x: -x,
            "min": lambda x: x,
            "near1": lambda x: abs(x - 1.0),
            "near0": abs}
    if direction not in KEYS:
        raise KeyError(
            "unknown direction %r in METRICS. Add it to `_best` rather than "
            "letting it default: an unrecognised direction used to fall "
            "through to 'highest wins' and bold the worst cell." % direction)
    key = KEYS[direction]
    shown = [float(spec % x) if np.isfinite(x) else np.nan for x in vals]
    scored = [(key(x) if np.isfinite(x) else np.inf) for x in shown]
    if len(set(scored)) < 2:
        return set()
    lo = min(scored)
    return {i for i, s in enumerate(scored) if s == lo}


def view_table(rs, bidders=("clf", "aft")):
    """Metrics down the side, views and contrasts across the top.

    TRANSPOSED from the v2 layout, 23 August 2026. With eleven metrics and two
    bidders the wide form ran to thirteen columns and read as a wall; this way
    each row is one quantity and the three differences sit beside the levels
    they come from, so a reader compares along a row rather than across a page.

    The best of C1 to C4 is bolded per row, by the direction in METRICS.
    """
    def g(f):
        vs = [f(r) for r in rs]
        # An all-NaN metric is EXPECTED, not broken: it is the all-rows row on a
        # machine that has not run `tier1_allrows.py`. The row is dropped below.
        # nanmean warns on an empty slice, and a warning printed by a reporting
        # script teaches a reader to ignore its warnings.
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return float(np.nanmean(vs))
    L = ["| Metric | C1 | C2 | C3 | C4 | oracle | C2-C1 | C3-C1 | C4-C2 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, acc, spec, direction, orc in metrics_for(bidders):
        if acc is None:                      # a section row, spanning nothing
            L.append("| %s | | | | | | | | |" % label)
            continue
        vals = [g(acc(v)) for v in VIEWS]
        if not np.isfinite(vals).any():
            # the all-rows rows before `tier1_allrows.py` has ever been run.
            # Dropping beats printing a row of "nan", which reads as a failure
            # rather than as a measurement not taken.
            continue
        star = _best(vals, direction, spec)
        cells = []
        for i, x in enumerate(vals):
            s = spec % x
            cells.append("**%s**" % s if i in star else s)
        # ONE MORE DIGIT ON THE DIFFERENCES than on the levels, and the two are
        # different kinds of quantity so the mixed precision is deliberate. The
        # levels are orientation and three decimals is the convention for them.
        # The differences ARE the contribution being claimed -- the price head's
        # SSP contrast is +0.0145 with an interval of width 0.002 -- and at three
        # decimals half of them collapse to +0.000 or the negative zero that
        # reads as a typo.
        dspec = "%.4f" if spec.endswith("f") and spec != "%.2f" else spec
        d = []
        for a, b in PAIRS:
            gap = vals[VIEWS.index(a)] - vals[VIEWS.index(b)]
            d.append(("%+" + dspec.lstrip("%")) % gap)
        if orc is None:
            ocell = "—"        # a literal em dash; an entity may not survive pandoc
        elif orc == "def1":
            ocell = spec % 1.0
        else:
            ocell = spec % g(orc)
        L.append("| %s | %s | %s | %s |"
                 % (label, " | ".join(cells), ocell, " | ".join(d)))

    sg = float(np.mean([r["C1"]["heads"]["win_price"]["sigma"] for r in rs]))
    sg3 = float(np.mean([r["C3"]["heads"]["win_price"]["sigma"] for r in rs]))
    L += ["",
          "C1 DSP only · C2 DSP + MMP · C3 DSP + SSP · C4 all three. "
          "**Bold = best of C1–C4.** Higher is better, except the error rows "
          "-- `spend CRPS`, `rmse_log`, `ece_win` "
          "-- where lower is, and the two bias rows: `EV level` is nearest 1.0 "
          "and `bias_log` nearest 0.", "",
          "**Oracle = the ceiling**, unreachable because it knows the competing "
          "bid. Dashed where it has no such quantity.", "",
          "**The funnel AUCs are scored on every test row**, against the "
          "uncensored master, which is how every economic number here is already "
          "computed. They are NOT the figures `results.json` records: those "
          "score each head only where its own view can see the label, so C1 and "
          "C3 are graded on the 27, 19 and 16 per cent of rows they won and C2 "
          "and C4 on all of them, and a contrast between two such figures is "
          "largely a difference in scoring population. The MMP click contrast "
          "reads +0.0642 that way and **+0.0220 scored like for like**, and the "
          "whole of the difference is C1 rising rather than C2 falling, since "
          "won rows carry about half the predictor spread. Both populations, "
          "with their row counts, are in `docs/gates/tier1_allrows.md`.", "",
          "**Price head sigma** is %.3f in C1 and C2, read off C3's residuals, "
          "and the placeholder %.3f in C3 and C4. Plan section 7b asks for one "
          "value across all four; the mismatch is queued as Q4. It cannot move "
          "any AUC, which sigma divides a log ratio to produce, and it does move "
          "the bid." % (sg, sg3)]
    return L


def scale_section(scale, rs, bidders=("clf", "aft")):
    fl = ratio_floor(rs)
    L = ["## %s, n = %d seeds" % (scale, len(rs)), ""]
    L += view_table(rs, bidders)
    # ONE normalisation of profit, not two. `profit / oracle` was the same
    # contrast divided by a different denominator, and its reason -- carrying a
    # profit gap across three scales whose levels differ tenfold -- went with the
    # other two scales. The oracle profit is in the table above for anyone who
    # wants the ratio. And no Layer column: it was one-to-one with Contrast.
    L += ["", "### What each layer is worth", "",
          "| Contrast | Bidder | Metric | Mean | 95% CI | Agree |",
          "|---|---|---|---:|---|---:|"]
    for a, b, layer in CONTRASTS:
        # the CONTRAST table narrows with the levels table. The gate files keep
        # every bidder; this document shows the ones it reports.
        for key, _, ekey, hkey in [b for b in BIDDERS if b[0] in bidders]:
            for nm, spec, f in [
                ("profit", "pct",
                 pct(econ(a, "profit", ekey), econ(b, "profit", ekey), fl)),
                ("value_captured", "%+.4f",
                 diff(econ(a, "value_captured", ekey), econ(b, "value_captured", ekey))),
                ("auc_win", "%+.4f",
                 diff(head(a, "auc", hkey), head(b, "auc", hkey))),
            ]:
                st = paired([f(r) for r in rs])
                L.append("| %s-%s | `%s` | %s | %s | [%s, %s] | %d/%d |"
                         % (a, b, key, nm, fmt(st["mean"], spec),
                            fmt(st["lo"], spec), fmt(st["hi"], spec),
                            st["agree"], st["n"]))
    L += ["", "### Price head, rmse_log", "",
          "| Contrast | Mean | 95% CI | Agree |", "|---|---:|---|---:|"]
    for a, b, _ in CONTRASTS:
        st = paired([diff(head(a, "rmse_log", "price"),
                          head(b, "rmse_log", "price"))(r) for r in rs])
        L.append("| %s-%s | %s | [%s, %s] | %d/%d |"
                 % (a, b, fmt(st["mean"], "%+.4f"), fmt(st["lo"], "%+.4f"),
                    fmt(st["hi"], "%+.4f"), st["agree"], st["n"]))
    return L[3:] if len(SCALES) == 1 else L


# ------------------------------------------------------------------- gate 6

def gate6(data, store):
    total = sum(len(v) for v in data.values())
    sweep = GATES / "direction_sweep.json"
    sw = json.loads(sweep.read_text(encoding="utf-8")) if sweep.exists() else []
    n_dir = sum(1 for r in sw if r["verdict"] == "PASS")
    ten = data.get("10M") or []
    p_c2 = float(np.mean([r["C2"]["heads"]["payer"]["auc"] for r in ten])) if ten else float("nan")
    p_c4 = float(np.mean([r["C4"]["heads"]["payer"]["auc"] for r in ten])) if ten else float("nan")
    agree_files = [GATES / ("agreement_%s.md" % s) for s in SCALES]
    conds = [
        (total == 30, "all 30 datasets present", "%d of 30" % total),
        (n_dir == 30, "every dataset passed its direction checks",
         "%d of 30" % n_dir),
        (bool(ten) and p_c2 > 0.53 and p_c4 > 0.53,
         "mean payer AUC above 0.53 in C2 and C4 at 10M",
         "C2 %.4f, C4 %.4f" % (p_c2, p_c4)),
        (all(f.exists() for f in agree_files),
         "the two-bidder agreement table exists at every scale",
         "%d of 3 written" % sum(1 for f in agree_files if f.exists())),
        (True, "both bidders reported, neither primary, disagreements described",
         "step 7d; the build never fails on a disagreement"),
        (True, "agreeing with v1 or v2 is not required to pass",
         "v2.2 replaces v2's headline wholesale"),
    ]
    return conds, all(ok for ok, _, _ in conds)


# ---------------------------------------------------------------------- writing

def write(path: Path, lines):
    """Write, refusing to touch a v2 filename.

    v2 is published at tag `v2.0.0` and its numbers must stay quotable. This
    writer overwrites unconditionally, so a basename collision would be silent.
    """
    if path.name in FROZEN:
        raise RuntimeError(
            "refusing to write %s: v2's results are published and must stay "
            "byte-identical. v2.2 writes its own files." % path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def require_allrows(data):
    """Refuse to write a results document with no funnel AUCs in it.

    THE FAILURE MODE THIS CLOSES was created on 23 August 2026 by a change that
    was otherwise right. The funnel rows used to be the figures `results.json`
    carries, with the all-rows figures added beside them, so a missing
    `tier1_allrows.json` cost three supplementary rows and left a complete
    table. The own-rows rows were then dropped, because a figure that must not
    be compared does not belong in a table whose every row carries a contrast
    column. Correct, and it made the remaining funnel rows depend entirely on a
    file this script does not produce.

    An absent row is invisible -- it leaves no gap -- so silently omitting click,
    install and payer AUC would publish a results table with no Tier-1 ranking
    metrics at all and nothing saying why. Raising is the only honest option:
    the document is the record, and a record missing a third of its evidence
    must not be written in the first place.
    """
    missing = []
    for scale, rs in data.items():
        if not rs:
            continue
        for h in ("click", "install", "payer"):
            vals = [r[v]["heads"]["%s_all" % h]["auc"]
                    for r in rs for v in VIEWS]
            if not any(np.isfinite(vals)):
                missing.append("%s %s" % (scale, h))
    if missing:
        raise SystemExit(
            "REFUSING TO WRITE. The funnel AUCs are scored on every test row and "
            "that pass has not run, so %s would be absent from the document with "
            "nothing to show it.\n\n"
            "  Run:  venv/Scripts/python.exe tools/make_report.py\n\n"
            "which orders `tools/tier1_allrows.py` before this script and skips "
            "it only when its output is newer than every results.json.\n"
            "Missing: %s" % ("they" if len(missing) > 1 else "it",
                             ", ".join(missing)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 10M ONLY BY DEFAULT (Ken, 23 August 2026). The campaign still RUNS all
    # thirty datasets and gate 6 still counts them, because the smaller scales
    # are a pipeline check; they are simply not reported as results.
    #
    # The argument for reporting three scales was that it defends the finding
    # against the "label efficiency that vanishes at scale" reading v2 gave
    # v1's SSP lift. That argument does not survive contact with why v2.2
    # exists: v1 and v2 had defects in design AND code -- the bid_density leak,
    # the label bug, the floor bug, the units bug -- so v2's explanation was
    # itself built on a broken chain and is not a live objection to anything
    # measured here.
    #
    # Nor does the finding need the help. At 10M the price head's SSP contrast
    # is +0.0145 over ten paired seeds, t = 26.7, and the WORST seed is +0.0107.
    # Sampling uncertainty was never the question. And 100K carries a genuinely
    # degenerate row: C1's payer head is scored on 25 rows and some seeds fall
    # back to an intercept, so its 0.500 is an unavailable head rather than a
    # measurement -- reporting it costs a paragraph to explain and buys nothing.
    ap.add_argument("--scale", default="10M", choices=SCALES + ["all"])
    ap.add_argument("--roas", type=float, default=None,
                    help="re-score the bidder rows to this ROAS target from "
                         "docs/roas_sweep_<scale>.json. The runs on disk are "
                         "not touched. Requires --bidders aft")
    ap.add_argument("--bidders", default="both", choices=sorted(BIDDER_SETS),
                    help="which bidders the RESULTS DOCUMENT reports. The "
                         "gate files always carry both")
    a = ap.parse_args(argv)
    want = SCALES if a.scale == "all" else [a.scale]
    bidders = BIDDER_SETS[a.bidders]
    # A separate file, never an overwrite. The two-bidder document is cited
    # and has to stay quotable, and a reader who finds one table where they
    # remember two must be able to see both.
    out_name = ("V2.2_Results.md" if a.bidders == "both"
                else "V2.2_Results_%s.md" % a.bidders.upper())

    if a.roas is not None and a.bidders != "aft":
        print("\nREFUSING: --roas needs --bidders aft. The sweep scores "
              "the price head\n  and the oracle only, so the classifier's "
              "rows cannot be re-scored.\n")
        return 2
    data = {s: load(s, roas=a.roas) for s in want}
    require_allrows(data)
    # gate 6 counts the whole campaign whatever is being reported
    everything = {s: (data[s] if s in data else load(s, roas=a.roas))
                  for s in SCALES}
    empty = empty_scales({s: data.get(s) for s in want})
    if empty:
        print("REFUSING TO WRITE: no seeds at %s." % ", ".join(empty))
        print("Aggregating nothing would replace a committed result with n/a.")
        print("Looked under %s." % RUNS)
        return 2

    md = ["# T9Sim v2.2 — results", "",
          "*Written by `tools/results_report.py`, never edited.*", "",
          "**v2.2 replaces v2's numbers wholesale.** The floor fix, the units fix "
          "and the wider ladder each move every figure, so a v2 number beside a "
          "v2.2 number compares two simulators, not two data layers.", "",
          "Contrasts are **paired within a seed**.", ""]
    if len(bidders) > 1:
        md += ["**Both bidders are reported and neither is primary**: the "
               "classifier's contrast moves only through features, the price "
               "head's only through labels. Their profit LEVELS are not "
               "comparable.", ""]
    else:
        md += ["**The price head alone is reported here**, by Ken's decision of "
               "24 August 2026. The win classifier is still trained, still in "
               "every `results.json` and still in the gate files: it is demoted "
               "out of the reported tables, not removed from the study. Its own "
               "SSP null stands and is a result. The two-bidder document is "
               "`V2.2_Results.md`.", ""]

    store = {}
    for s in want:
        rs = data[s]
        md += scale_section(s, rs, bidders)
        cells, rows, sweep, fl = agreement(rs, s)
        store[s] = cells
        ag = agreement_md(s, rows, sweep, fl, len(rs))
        write(GATES / ("agreement_%s.md" % s), ag)
        (GATES / ("agreement_%s.json" % s)).write_text(
            json.dumps({"cells": cells, "sigma_sweep": sweep,
                        "ratio_floor": fl, "n": len(rs)},
                       indent=1, default=float), encoding="utf-8")
        print("  agreement_%s: %d cells" % (s, len(cells)))
        md.append("")

    write(DOCS / out_name, md)

    # The 10M CI twin existed only to give 10M its own document while the main
    # one carried three scales. With the main one at 10M they would be the same
    # file, so it is written only when more than one scale is asked for.
    if len(want) > 1 and data.get("10M"):
        ci = ["# T9Sim v2.2 — 10M, n = 10, the v1-layout tables", "",
              "*Generated by `tools/results_report.py`. Written, never edited.*", ""]
        ci += view_table(data["10M"], bidders)
        ci += [""] + scale_section("10M", data["10M"])[3:]
        write(DOCS / ("V2.2_Results_10M_n10_CI.md" if a.bidders == "both"
                      else "V2.2_Results_10M_n10_CI_%s.md" % a.bidders.upper()), ci)

    conds, ok = gate6(everything, store)
    g = ["# Gate 6 report", "",
         "*Generated by `tools/results_report.py`. Written, never edited.*", "",
         "| Result | Condition | Measured |", "|---|---|---|"]
    for c_ok, cond, det in conds:
        g.append("| %s | %s | %s |" % ("PASS" if c_ok else "**FAIL**", cond, det))
    g += ["", "## Verdict", "", "**Gate 6: %s**" % ("PASS" if ok else "FAIL")]
    write(GATES / "gate6.md", g)
    (GATES / "stage6.json").write_text(json.dumps(store, indent=1, default=float),
                                       encoding="utf-8")

    print("wrote docs/%s and docs/gates/gate6.md" % out_name)
    print("gate 6: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
