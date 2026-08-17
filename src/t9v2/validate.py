# -*- coding: utf-8 -*-
"""The gate-2 validator: 6 calibration levels and 5 direction checks.

    python -m t9v2.validate output/t9v2_100K_seed20250.parquet

The two kinds are judged differently, and validation.yaml states the policy:
the 5 DIRECTION checks are ANDed into the run's pass or fail, while the 6
calibration LEVELS are reported with an in-band status and do not gate. A level
is a calibration outcome the solver aims at; a direction is a structural claim
about the generator, and a structural claim that fails is a defect.

Every direction check here is transcribed from what v1's validator ACTUALLY did,
not from v1's prose, which had drifted from the implementation in two places.
Both divergences are documented improvements and the notes say which.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .core.config import load, val

ARCH_ORDER = ["whale", "engaged_spender", "casual", "time_filler", "inactive"]
FORMAT_NAME = {1: "banner", 2: "interstitial", 3: "rewarded"}


# ---------------------------------------------------------------------------
# the 6 calibration levels
# ---------------------------------------------------------------------------

def levels(df):
    """The measured value of each calibration target, whatever it is."""
    clicks = int(df["click"].sum())
    installs = int(df["install"].sum())
    payers = int(df["is_payer"].sum())
    spend = df.loc[df["is_payer"] == 1, "ltv_value"].to_numpy()

    if spend.size:
        cut = np.quantile(spend, 0.95)
        whale_share = float(spend[spend >= cut].sum() / spend.sum())
        median_spend = float(np.median(spend))
    else:
        whale_share, median_spend = float("nan"), float("nan")

    return {
        "population_ctr": clicks / len(df),
        "click_to_install": installs / clicks if clicks else float("nan"),
        "install_to_payer": payers / installs if installs else float("nan"),
        "whale_concentration": whale_share,
        "median_payer_spend_usd": median_spend,
        "auction_win_rate": float(df["won"].mean()),
    }


# The 2 levels measured on PAYERS, and the payer count each needs to be worth
# reading. Established empirically on 16 August 2026 by measuring the SAME
# settings at three scales:
#
#     rows    payers   whale concentration   median payer spend
#       1M       405          0.4445               7.153
#       5M     2,043          0.6008               6.107
#      10M     4,159          0.5949               6.029
#
# The band is [0.55, 0.65] and the target 6.0, so the generator hits both from
# 5M rows onward. The 1M reading is not the generator missing; it is the top-5
# percent of 20 people. Below the threshold these report as NOT MEASURABLE
# rather than out of band, because calling an unmeasurable level a failure sends
# the next person to re-solve a constant that was already right.
PAYER_LEVELS = {"whale_concentration": 2000, "median_payer_spend_usd": 2000}


def check_levels(s, df):
    got = levels(df)
    payers = int(df["is_payer"].sum())
    out = []
    for name, spec in s.raw["calibration_targets"].items():
        lo, hi = spec["band"]
        v = got[name]
        need = PAYER_LEVELS.get(name)
        row = {"id": name, "value": v, "band": [lo, hi], "payers": payers,
               "needs_payers": need,
               "measurable": need is None or payers >= need,
               "in_band": bool(lo <= v <= hi) if v == v else False}
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# the 5 direction checks
# ---------------------------------------------------------------------------

def d_ltv_by_archetype_monotone(df):
    """mean e_ltv strictly decreasing down the archetype ladder.

    On the DETERMINISTIC e_ltv, not on realised payer spend. Realised medians
    need thousands of payers per archetype and tie or invert on noise at 100K.
    v1's prose said payer spend, which was wrong about v1's own code.
    """
    m = df.groupby("lu1_archetype", observed=True)["e_ltv"].mean()
    seq = [float(m.get(a, float("nan"))) for a in ARCH_ORDER]
    ok = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    return ok, "mean e_ltv " + " > ".join("%s %.4g" % (a, v)
                                          for a, v in zip(ARCH_ORDER, seq))


def d_relevance_lifts_ctr(df):
    """ctr(r above median) > ctr(r at or below median), r = lu6[ad_genre]."""
    cols = {g: "lu6_%s" % g for g in ("casual", "strategy", "rpg", "hypercasual")}
    r = np.zeros(len(df))
    for g, c in cols.items():
        m = (df["ad_genre"] == g).to_numpy()
        if m.any():
            r[m] = df.loc[m, c].to_numpy()
    hi = r > np.median(r)
    a = float(df["click"].to_numpy()[hi].mean())
    b = float(df["click"].to_numpy()[~hi].mean())
    return a > b, "ctr high-relevance %.5f vs low %.5f" % (a, b)


def d_format_ecpm_ordering(df):
    """mean winning_price over SOLD rows: rewarded > interstitial > banner."""
    sold = df[df["winning_price"].notna()]
    m = sold.groupby("slot_format", observed=True)["winning_price"].mean()
    b, i, r = (float(m.get(k, float("nan"))) for k in (1, 2, 3))
    return (r > i > b), "banner %.3f < interstitial %.3f < rewarded %.3f" % (b, i, r)


def d_adverse_selection(df):
    """mean log(ev_truth) on LOST rows > on WON rows, over ev_truth > 0.

    IN LOG SPACE. Raw means are whale-dominated and flip sign on an unlucky seed.
    v1's prose said raw means, which was wrong about v1's own code.

    This is the mechanism the whole study is about: we win the auctions the market
    valued least, so what the DSP observes is a biased sample of what was there.
    """
    pos = df["ev_truth"].to_numpy() > 0
    lg = np.log(df["ev_truth"].to_numpy()[pos])
    won = df["won"].to_numpy()[pos] == 1
    a, b = float(lg[~won].mean()), float(lg[won].mean())
    return a > b, "mean log ev lost %.4f > won %.4f (gap %.4f)" % (a, b, a - b)


def d_win_rises_with_bid_within_format(df):
    """Within EACH slot_format, win rate in the top bid quartile beats the bottom.

    Within format, not pooled. Pooling confounds bid level with format, because
    rewarded bids are the highest and also face the highest hurdles, so a pooled
    test can read backwards while every format is individually monotone.
    """
    parts, ok = [], True
    for f in (1, 2, 3):
        d = df[df["slot_format"] == f]
        if len(d) < 8:
            continue
        q1, q3 = np.quantile(d["bid_price"], [0.25, 0.75])
        top = float(d.loc[d["bid_price"] >= q3, "won"].mean())
        bot = float(d.loc[d["bid_price"] <= q1, "won"].mean())
        ok &= top > bot
        parts.append("%s %.3f>%.3f" % (FORMAT_NAME[f], top, bot))
    return ok, "; ".join(parts)


DIRECTION = [
    ("ltv_by_archetype_monotone", d_ltv_by_archetype_monotone),
    ("relevance_lifts_ctr", d_relevance_lifts_ctr),
    ("format_ecpm_ordering", d_format_ecpm_ordering),
    ("adverse_selection", d_adverse_selection),
    ("win_rises_with_bid_within_format", d_win_rises_with_bid_within_format),
]


# ---------------------------------------------------------------------------
# the exact identities
# ---------------------------------------------------------------------------

def identities(df):
    """The gate identities, which hold EXACTLY or the build is broken.

    No tolerance and no proportion-of-rows: one violating row is a failure. They
    are separated from the direction checks because a direction check is a claim
    about a distribution and these are claims about every single row.
    """
    ts = df["click_timestamp"].to_numpy()
    its = df["install_timestamp"].to_numpy()
    out = [
        ("install 0 wherever click 0",
         bool(((df["click"] == 0) & (df["install"] == 1)).sum() == 0)),
        ("is_payer 0 wherever install 0",
         bool(((df["install"] == 0) & (df["is_payer"] == 1)).sum() == 0)),
        ("ltv_value 0 wherever not payer",
         bool((df.loc[df["is_payer"] == 0, "ltv_value"] != 0).sum() == 0)),
        ("click_timestamp -1 exactly where no click",
         bool(((df["click"] == 0) == (ts == -1)).all())),
        ("install_timestamp -1 exactly where no install",
         bool(((df["install"] == 0) == (its == -1)).all())),
        ("winning_price is our own bid on won rows (first price)",
         bool(np.allclose(df.loc[df["won"] == 1, "winning_price"],
                          df.loc[df["won"] == 1, "bid_price"]))),
        ("winning_price is NaN exactly on unsold rows",
         bool((df["winning_price"].isna()
               == ((df["won"] == 0) & (df["lu7_competing_bid"] < df["floor_price"]))).all())),
        ("bid_density in 1..8",
         bool(df["bid_density"].between(1, 8).all())),
        ("ltv_7d = 0.40 x ltv_value and ltv_30d = 0.70 x ltv_value",
         bool(np.allclose(df["ltv_7d"], 0.40 * df["ltv_value"])
              and np.allclose(df["ltv_30d"], 0.70 * df["ltv_value"]))),
        ("ev_truth = p_click x p_install x p_payer x e_ltv",
         bool(np.allclose(df["ev_truth"],
                          df["p_click"] * df["p_install"] * df["p_payer"] * df["e_ltv"]))),
        ("inactive users carry ev_truth exactly 0",
         bool((df.loc[df["lu1_archetype"] == "inactive", "ev_truth"] == 0).all())),
    ]
    return out


# ---------------------------------------------------------------------------

def run(path, profile="default", quiet=False):
    import pandas as pd
    s = load(profile)
    df = pd.read_parquet(path)

    lv = check_levels(s, df)
    ids = identities(df)
    dirs = [(name, *fn(df)) for name, fn in DIRECTION]

    if not quiet:
        print("%s" % path)
        print("  %d rows, %d columns\n" % (len(df), len(df.columns)))
        print("  CALIBRATION LEVELS (reported, do not gate)")
        for r in lv:
            if not r["measurable"]:
                status = ("NOT MEASURABLE at this scale, %d payers against the %d "
                          "this level needs" % (r["payers"], r["needs_payers"]))
            else:
                status = "in band" if r["in_band"] else "OUT OF BAND"
            print("    %-24s %-12s band [%g, %g]  %s"
                  % (r["id"], "%.5g" % r["value"], r["band"][0], r["band"][1], status))
        print("\n  EXACT IDENTITIES (every row, no tolerance)")
        for name, ok in ids:
            print("    %-4s %s" % ("PASS" if ok else "FAIL", name))
        print("\n  DIRECTION CHECKS (ANDed into pass or fail)")
        for name, ok, detail in dirs:
            print("    %-4s %-34s %s" % ("PASS" if ok else "FAIL", name, detail))

    passed = all(ok for _, ok in ids) and all(ok for _, ok, _ in dirs)
    if not quiet:
        print("\n  RESULT: %s" % ("PASS" if passed else "FAIL"))
        bad = [r["id"] for r in lv if r["measurable"] and not r["in_band"]]
        unm = [r["id"] for r in lv if not r["measurable"]]
        if bad:
            print("  %d calibration level(s) out of band, reported not gated: %s"
                  % (len(bad), ", ".join(bad)))
        if unm:
            print("  %d level(s) not measurable at this scale, reported not failed: %s"
                  % (len(unm), ", ".join(unm)))
    return {"passed": passed, "levels": lv, "identities": ids,
            "directions": [(n, ok, d) for n, ok, d in dirs]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--profile", default="default")
    a = ap.parse_args(argv)
    return 0 if run(Path(a.path), a.profile)["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
