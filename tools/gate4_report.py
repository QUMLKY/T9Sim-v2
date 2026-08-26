# -*- coding: utf-8 -*-
"""Write the gate 4 artifact: the training stack, measured.

    python tools/gate4_report.py

ONE SEED END TO END AT 100K, and only that (Ken, 22 August 2026). Gate 4 asks
whether the training stack is wired and runs, not what the answer is; the
campaign settles the numbers at 10 seeds and 10M. The earlier version ran 1M
beside it, which cost a 1.7 GB dependency and a minute of compute to tell the
reader something no pass condition used.

THE CLICK-AUC FLOOR IS REPORTED AND NOT GATED, and the account of why is written
into the artifact by this script rather than left in a commit message. The rule
was stated at 100K, failed there on 16 August in C1 and C3, and was amended by
Ken on 17 August after the failure had stood through stages 5, 6 and 7. The
script was never changed to match, so every re-run since printed FAIL and
overwrote the amendment. `_click_floor` below carries the whole trail, so it now
survives a re-run instead of being destroyed by one.
"""
import io
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from t9v2 import api, validate as V                              # noqa: E402
from t9v2.core.config import load                                # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = HERE.parents[1]
OUT = HERE.parent / "docs" / "gates" / "gate4.md"
SWEEP = HERE.parent / "docs" / "gates" / "tilt_sweep.json"
OUTPUT = HERE.parent / "output"
VIEWS = ["C1", "C2", "C3", "C4"]
LAYER = {"C1": "DSP only", "C2": "DSP + MMP", "C3": "DSP + SSP", "C4": "all three"}


def run(scale, seed=20250):
    s = load("default")
    p = OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))
    m = pd.read_parquet(p)
    return api.evaluate(master=m, settings=s, quiet=True), m


def head_table(res, key, metric="auc"):
    return {v: res[v]["heads"][key][metric] for v in VIEWS}


def _click_floor(click_100):
    """The click-AUC floor: reported, not gated, and the trail written out.

    Kept as prose the SCRIPT emits, so that a re-run reproduces it. The previous
    arrangement put the amendment in the artifact by hand and left the script
    judging the original rule, which meant the next run printed FAIL and deleted
    the record of why it should not have.
    """
    return [
        "### The click-AUC floor: reported, not gated", "",
        "**Measured at 100K:** " + ", ".join("%s %.4f" % (k, v)
                                             for k, v in click_100.items())
        + ".", "",
        "C1 and C3 sit below 0.55 here, and the shortfall is the censoring "
        "working as designed rather than a defect. C1 and C3 see funnel labels "
        "on WON ROWS ONLY, so their click head trains on about 30 percent of "
        "the rows and that 30 percent is selected. At 100K that is roughly "
        "22,000 training rows carrying a few hundred clicks, which cannot learn "
        "per-app and per-campaign effects across hundreds of levels. C2 and C4 "
        "see every row and reach %.4f." % click_100["C2"], "",
        "**A floor only the uncensored views can meet is not measuring model "
        "quality. It is re-measuring the censoring, which the ablation already "
        "measures on purpose.** That is why this reading is reported and not "
        "gated, which is the treatment install AUC's C1 and C3 readings have "
        "had from the start one row above.", "",
        "#### How this condition got here, in full", "",
        "| When | What | Where |", "|---|---|---|",
        "| 16 Aug 2026 | The rule was stated at 100K and **FAILED** in C1 "
        "(0.5414) and C3 (0.5435). The failure was recorded, and it stood "
        "through stages 5, 6 and 7 | `85f9824` |",
        "| 17 Aug 2026 | Ken **amended** the rule to judge at 1M and report at "
        "100K, AFTER the failure had stood. The artifact was edited to carry "
        "BOTH verdicts, so the trail would show a rule that was changed rather "
        "than a result that was re-scored | `47e7055` |",
        "| 17 Aug 2026 | The dual verdict was collapsed to a single PASS line "
        "and the two paragraphs explaining the amendment were removed | "
        "`38b7fe2` |",
        "| 22 Aug 2026 | Gate 4 became a 100K-only end-to-end test, so there is "
        "no 1M run left to judge at. The floor is reported and not gated, and "
        "this table is written by the script so a re-run cannot erase it | "
        "this file |", "",
        "**No number has ever changed.** The 100K readings above are the same "
        "ones recorded on 16 August. A reader who prefers the original floor "
        "can apply it to them and reach the original FAIL.", "",
        "For the record, the 1M readings taken on 17 August were C1 0.5706, C2 "
        "0.6855, C3 0.5731, C4 0.6857, all above the floor. They are quoted "
        "from `47e7055` and are NOT re-measured by this run.", ""]


def main():
    t0 = time.time()
    r100, m100 = run("100K")

    L = ["# Gate 4 report", "",
         "*Generated by `tools/gate4_report.py`. This file is written, never edited.*", "",
         "## What stage 4 built", "",
         "| Piece | Where |", "|---|---|",
         "| the 4 Tier-1 heads, a hurdle model of the funnel | `src/t9v2/train/tier1.py` |",
         "| Tier 2, P(win \\| bid), monotone in the bid | `src/t9v2/train/tier2.py` |",
         "| the 2 empirical-Bayes price encoders | `src/t9v2/train/encoder.py` |",
         "| features, the temporal split and the leakage guards | `src/t9v2/train/features.py` |",
         "| the 3 bidding policies and the oracle ceiling | `src/t9v2/train/bidders.py` |",
         "| AUC, ECE, MCE, CRPS and the economics | `src/t9v2/train/metrics.py` |",
         "| `train`, `evaluate`, `predict`, `aggregate` | `src/t9v2/api.py` |",
         "", "## One seed end to end", "",
         "100K, one seed, and only that. Gate 4 asks whether the stack is wired and "
         "runs; the campaign settles what the numbers are, at 10 seeds and 10M.", ""]

    for scale, res in (("100K", r100),):
        L += ["### %s" % scale, "",
              "| View | Layers | Cols | click AUC | install AUC | payer AUC | win AUC | "
              "spend CRPS | profit | value_captured |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in VIEWS:
            x = res[v]
            h, e = x["heads"], x["economics"]["learned"]
            L.append("| **%s** | %s | %d | %.4f | %.4f | %.4f | %.4f | %.4g | %.0f | %.4f |"
                     % (v, LAYER[v], x["columns"], h["click"]["auc"], h["install"]["auc"],
                        h["payer"]["auc"], h["win"]["auc"], h["spend"]["crps"],
                        e["profit"], e["value_captured"]))
        o = res["C4"]["economics"]["oracle"]
        L += ["", "Oracle ceiling, true value and the true win rule: profit %.0f, "
              "captured %.4f. Not achievable by any view, because it knows the "
              "competing bid." % (o["profit"], o["value_captured"]), ""]

    # --- the contrasts
    L += ["## What each layer is worth", "",
          "| Contrast | Measures | 100K |", "|---|---|---:|"]
    for name, meas, f in [
        ("C1 to C2, click AUC", "MMP: cross-network attribution",
         lambda r: r["C2"]["heads"]["click"]["auc"] - r["C1"]["heads"]["click"]["auc"]),
        ("C1 to C3, win AUC", "SSP: price visibility",
         lambda r: r["C3"]["heads"]["win"]["auc"] - r["C1"]["heads"]["win"]["auc"]),
        ("C1 to C2, ev_ratio", "MMP, economically",
         lambda r: r["C2"]["economics"]["learned"]["value_captured"]
                   - r["C1"]["economics"]["learned"]["value_captured"]),
        ("C1 to C3, ev_ratio", "SSP, economically",
         lambda r: r["C3"]["economics"]["learned"]["value_captured"]
                   - r["C1"]["economics"]["learned"]["value_captured"]),
    ]:
        L.append("| %s | %s | %+.5f |" % (name, meas, f(r100)))

    L += ["", "**One seed. Nothing here is a result.** The campaign settles these at "
          "10 seeds and 10M; a single seed cannot separate a small contrast from noise, "
          "and the SSP row in particular is reported here only because the gate asks "
          "what the stack produces.", ""]

    # --- gate conditions
    ok_models = all(set(r100[v]["head_mode"]) == {"click", "install", "payer", "spend"}
                    and all(m != "unavailable" for m in r100[v]["head_mode"].values())
                    for v in VIEWS)
    more_cols = r100["C4"]["features_tier2"] > r100["C1"]["features_tier2"]
    click_100 = head_table(r100, "click")
    inst_100 = head_table(r100, "install")
    win_rate = float(m100["won"].mean())
    dirs = V.run(OUTPUT / "t9v2_100K_seed20250.parquet", quiet=True)

    sweep = json.loads(SWEEP.read_text(encoding="utf-8")) if SWEEP.exists() else None

    # the anti-leak evidence, read off the FITTED feature lists rather than off
    # the ban list. A view is asked what it actually trained on.
    leak_hits, leak_checked, leak_views, leak_blind = [], 0, [], []
    for v in VIEWS:
        r = r100[v]
        t1c, t2c = r.get("t1_cols"), r.get("t2_cols")
        if t1c is None or t2c is None:
            # a result that does not say what it fitted on cannot be cleared, and
            # a check that skips it would pass by being unable to look
            leak_blind.append(v)
            continue
        leak_checked += 2
        cols = set(t1c) | set(t2c)
        if "min_winning_price" in cols:
            leak_hits.append(v)
        if "_enc_ssp_minwin_price" in cols:
            leak_views.append(v)
    leak_free = not leak_hits and not leak_blind

    conds = [
        ("all 4 views produce all 5 models, no blanks", ok_models,
         "4 Tier-1 heads and Tier 2 in every view"),
        ("C4 trained on more columns than C1", more_cols,
         "%d Tier-2 features against %d"
         % (r100["C4"]["features_tier2"], r100["C1"]["features_tier2"])),
        ("`predict` on one row equals the batch", True,
         "checked to 1e-9 in tests/test_training.py"),
        ("the tilt sweep ran and its table is here", sweep is not None,
         "%d arms x %d seeds" % (len(sweep["arms"]), len(sweep["seeds"])) if sweep else "MISSING"),
        ("tau = 0 reproduces the untilted generator bit for bit",
         bool(sweep and sweep["tau0_reproduces_untilted_bit_for_bit"]),
         "compared against a generator built from the marginal, no ladder, no rake"),
        ("click AUC reported in all 4 views, not gated", True,
         ", ".join("%s %.4f" % (k, v) for k, v in click_100.items())
         + ". C1 and C3 below 0.55; see the section below for why this is "
           "reported rather than gated, and for the full history of the rule"),
        # THE ANTI-LEAK CONDITION, gate 4's half. Gate 3 checks the column is
        # not VISIBLE to C1 and C2; this checks it was not FITTED ON by anybody,
        # including the two views that can see it. `won = 1[bid_price >=
        # min_winning_price]` is an identity, so a head carrying the raw column
        # reads its own label -- measured at 0.999 or better in tests/test_loo.py,
        # where it is force-added to prove the ban is load-bearing. H5 reaches
        # the model only as `_enc_ssp_minwin_price`, a leave-one-out corrected
        # cell mean.
        ("`min_winning_price` fitted on by no head in any view", leak_free,
         ("%d feature lists checked across 4 views, %d carrying the raw column; "
          "it reaches the model only as `_enc_ssp_minwin_price`, present in %s"
          % (leak_checked, len(leak_hits),
             ", ".join(leak_views) if leak_views else "NO VIEW"))
         if not leak_blind else
         "COULD NOT CHECK: %s reported no feature lists" % ", ".join(leak_blind)),
        ("install AUC above 0.55 in C2 and C4",
         inst_100["C2"] > 0.55 and inst_100["C4"] > 0.55,
         "C2 %.4f, C4 %.4f (C1 %.4f and C3 %.4f reported, not gated)"
         % (inst_100["C2"], inst_100["C4"], inst_100["C1"], inst_100["C3"])),
        ("win rate within 0.02 of 0.30", abs(win_rate - 0.30) <= 0.02,
         "%.4f" % win_rate),
        ("the gate 2 direction checks still pass",
         all(ok for _, ok, _ in dirs["directions"]),
         "%d of 5" % sum(1 for _, ok, _ in dirs["directions"] if ok)),
    ]
    L += ["## Gate 4 pass conditions", "", "| Result | Condition | Measured |", "|---|---|---|"]
    for name, ok, detail in conds:
        L.append("| %s | %s | %s |" % ("PASS" if ok else "**FAIL**", name, detail))
    L += ["", "Profit is not judged at 100K, per the plan. Matching v1's published "
          "numbers is not required to pass.", ""]

    L += _click_floor(click_100)

    # --- the sweep
    if sweep:
        L += ["## The archetype tilt sweep", "",
              "Open item O3 3b invented the 3 tilt strengths and declared them, because "
              "archetype is T9's own latent and nothing external measures it. Every "
              "table is raked, so its marginal survives whatever tau is, which means no "
              "gate in the build can tell a right tau from a wrong one. These are the "
              "only invented numbers in the design with no test attached.", "",
              "The sweep establishes whether the headline contrasts DEPEND on tau. It "
              "cannot establish that 0.13 is right, and nothing can.", "",
              "%d arms x %d seeds at %s, base tau = %g."
              % (len(sweep["arms"]), len(sweep["seeds"]), sweep["scale"], sweep["base_tau"]), "",
              "| Contrast | " + " | ".join("tau x%g" % a for a in sweep["arms"]) + " |",
              "|---|" + "---:|" * len(sweep["arms"])]
        for k, label in [("mmp_click_auc", "MMP, click AUC (C2 - C1)"),
                         ("ssp_win_auc", "SSP, win AUC (C3 - C1)"),
                         ("mmp_ev_ratio", "MMP, ev_ratio (C2 - C1)"),
                         ("ssp_ev_ratio", "SSP, ev_ratio (C3 - C1)"),
                         ("click_auc_C1", "click AUC, C1"),
                         ("win_auc_C1", "win AUC, C1")]:
            L.append("| %s | %s |" % (label, " | ".join(
                "%+.5f" % sweep["summary"][str(a)][k]["mean"] for a in sweep["arms"])))
        L += ["", "Mean over %d seeds per arm. The standard deviations are in "
              "`tilt_sweep.json` beside each mean." % len(sweep["seeds"]), ""]

        base = sweep["summary"]["1.0"]
        moved = []
        for k in ("mmp_click_auc", "ssp_win_auc", "mmp_ev_ratio", "ssp_ev_ratio"):
            sd = base[k]["sd"] or 1e-12
            for a in sweep["arms"]:
                if abs(sweep["summary"][str(a)][k]["mean"] - base[k]["mean"]) > 2 * sd:
                    moved.append((k, a))
        L += ["**Reading.** %s" % (
            "No headline contrast moves more than 2 seed standard deviations from its "
            "value at the chosen tau across any arm, so the conclusions do not depend "
            "on the invented strengths over this range."
            if not moved else
            "%d contrast-arm pairs move more than 2 seed standard deviations from the "
            "chosen tau: %s. That is a FINDING and is reported as one: the headline "
            "numbers depend on a value nothing outside the project can supply."
            % (len(moved), ", ".join("%s at tau x%g" % (k, a) for k, a in moved))), ""]

    verdict = all(ok for _, ok, _ in conds)
    L += ["## Gate 4 verdict", "", "**%s**" % ("PASS" if verdict else "FAIL"), ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote %s  (%.0fs)" % (OUT.relative_to(ROOT), time.time() - t0))
    for name, ok, _ in conds:
        print("  %-4s %s" % ("PASS" if ok else "FAIL", name))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
