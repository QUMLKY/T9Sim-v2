# -*- coding: utf-8 -*-
"""Solve the calibration constants and write calibrated.yaml.

    python -m t9v2.autocal --scale 100K --seed 20250

A constant is SOLVED by generating, measuring the aggregate it aims at, moving
it, and repeating until the aggregate lands in its band. Nothing here is fitted
to data; each is a scalar bisection against a published target, which is why the
route is `auto-calibrated` and why the provenance of the answer is the target and
this procedure rather than the number.

Order is not arbitrary. The three funnel rates are separable, because each gates
the next: base_ctr sets who clicks, base_ir sets who installs GIVEN a click, and
base_payer sets who pays GIVEN an install. The LTV pair comes next, and k_global
comes last because the rival bids read a standardised value score whose spread
moves when the LTV constants move.

WHAT IS NOT SOLVED HERE, and no longer claims to be. The three archetype LTV
means aim at the same whale-concentration target lognormal_sigma aims at, so all
four against one scalar is four unknowns and one equation. sigma is the one the
solver takes; the three means are held at v1's values.

Until 16 August 2026 they were routed `auto-calibrated`, which claimed a solver
had produced them when none had, and the contract in calibrated.yaml listed them.
Both are corrected: the route is `inferred` and the contract is 6, so what the
solver promises and what it delivers are the same list. Solving all four would
need a per-archetype revenue-share target, and nothing publishes one, because
archetype is T9's own latent rather than an industry category.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from . import generate as G
from . import validate as V
from .core.config import CONFIG, load

CALIBRATED = CONFIG / "calibrated.yaml"

# key -> (settings path, measured level, search bracket, direction, rows)
# direction +1 means raising the constant raises the level.
#
# ROWS DIFFER BY CONSTANT, and that is the whole reason this column exists. The
# three funnel rates and the win rate are measured on events that happen on a
# large fraction of rows, so 100K gives thousands of them. The two LTV constants
# are measured on PAYERS, and a payer is about 4 in 10,000 rows: 100K rows holds
# roughly 40 payers, whose top 5 percent is two people. Solving a whale-
# concentration target against two people is solving against noise, and on
# 16 August 2026 it produced sigma = 5.29 against v1's 1.65, which then made
# e_ltv astronomical and the win-rate solve impossible to bracket.
# Each entry also names the SCALE it solves at, because a scale fixes the pool
# sizes and a constant should be solved on the configuration it ships in. The
# first run of this solver measured the LTV pair over 10M rows drawn from
# 1M-scale pools, 220,000 users, so every user appeared about 45 times instead of
# 4.5. It read whale concentration 0.595 while the shipped 10M dataset, with its
# own 2.2M-user pool, read 0.632. Both sit inside the [0.55, 0.65] band, but the
# gap is a third of the band's width for a reason that has nothing to do with the
# design, and it would have shown up as unexplained seed spread in the campaign.
#
# The funnel rates and the win rate stay at 1M scale: they are per-row rates,
# they are stable by 200K rows, and building a 2.2M-user pool for each of thirty
# measurements would cost far more than it could possibly buy.
PLAN = [
    ("funnel.base_ctr", "funnel.base_ctr.start.value", "population_ctr", (0.01, 5.0), +1, 200_000, "1M"),
    ("funnel.base_ir", "funnel.base_ir.start.value", "click_to_install", (0.05, 20.0), +1, 200_000, "1M"),
    ("funnel.base_payer", "funnel.base_payer.start.value", "install_to_payer", (0.005, 5.0), +1, 1_000_000, "1M"),
    ("ltv.lognormal_mu", "ltv.lognormal_mu.value", "median_payer_spend_usd", (-3.0, 6.0), +1, 10_000_000, "10M"),
    ("ltv.lognormal_sigma", "ltv.lognormal_sigma.value", "whale_concentration", (0.4, 3.5), +1, 10_000_000, "10M"),
    ("k_global", "auction.k_global.value", "auction_win_rate", (0.01, 50.0), +1, 200_000, "1M"),
]


def target_of(s, level):
    """The midpoint of the published band, or the stated reference if there is one.

    Aiming at the midpoint rather than at an edge leaves the solved value inside
    the band under seed noise, which matters because the gate re-measures on a
    different sample from the one the solve used.
    """
    spec = s.raw["calibration_targets"][level]
    if "reference" in spec:
        return float(spec["reference"]), tuple(spec["band"])
    lo, hi = spec["band"]
    return (lo + hi) / 2.0, (lo, hi)


def bisect(measure, lo, hi, target, direction, tol_rel=0.01, max_iter=18, log=None):
    """Plain bisection on a monotone scalar. Returns (x, achieved, iterations).

    The bracket is widened rather than trusted: if the target is not inside
    [f(lo), f(hi)] the search would converge to an endpoint and report success,
    so the endpoints are pushed out first and a failure to bracket is raised.
    """
    f_lo, f_hi = measure(lo), measure(hi)
    grow = 0
    while direction * (f_lo - target) > 0 and grow < 6:
        lo /= 4.0; f_lo = measure(lo); grow += 1
    while direction * (f_hi - target) < 0 and grow < 12:
        hi *= 4.0; f_hi = measure(hi); grow += 1
    if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
        raise RuntimeError("cannot bracket the target %.6g; the level spans "
                           "[%.6g, %.6g] over the search range" % (target, f_lo, f_hi))

    x = (lo + hi) / 2.0
    for i in range(1, max_iter + 1):
        x = (lo + hi) / 2.0
        got = measure(x)
        if log:
            log("      iter %2d  x=%.6g  level=%.6g  target=%.6g" % (i, x, got, target))
        if abs(got - target) <= tol_rel * abs(target):
            return x, got, i
        if direction * (got - target) < 0:
            lo = x
        else:
            hi = x
    return x, measure(x), max_iter


def solve(scale=None, seed=20250, rows=None, quiet=False, only=None):
    """Run the whole plan and return the solved constants with their evidence."""
    # calibrated=False: solve from the hand-written starts, never from the last
    # answer, so a re-solve is idempotent and one bad solve cannot poison the next
    s0 = load("default", calibrated=False)
    solved, overrides = {}, {}
    n_used = {}

    # A PARTIAL re-solve must start from the constants it is NOT re-solving.
    # In a full run `overrides` accumulates each answer as the plan proceeds, so
    # k_global, solved last, sees the solved LTV values. Skipping the first five
    # leaves that empty and k_global is then solved against the unsolved STARTS,
    # where lognormal_mu is 1.7918 rather than 0.146. That inflates e_ltv and our
    # bid, so it solved to 0.2907 instead of 1.747 and the shipped data would have
    # run at a 2 percent win rate. Seeding from the file is the whole reason
    # `--only` is safe to offer.
    if only:
        import yaml
        kept = (yaml.safe_load(CALIBRATED.read_text(encoding="utf-8")) or {})
        for path, entry in (kept.get("solved") or {}).items():
            if path not in {p for k, p, *_ in PLAN if k in only}:
                overrides[path] = entry["value"]

    def say(msg):
        if not quiet:
            print(msg)

    for key, path, level, bracket, direction, plan_rows, plan_scale in PLAN:
        # `only` re-solves a subset. The constants are not all coupled: the funnel
        # rates and the LTV pair are drawn independently of the market, so a change
        # to the rival-bid law moves k_global alone. Re-solving the other 5 against
        # unchanged laws would spend 30 minutes to reproduce the same answers with
        # fresh bisection noise, and would make calibrated.yaml differ for no reason.
        if only and key not in only:
            continue
        target, band = target_of(s0, level)
        start = _read(load("default", calibrated=False), path)
        n = rows or plan_rows
        sc = scale or plan_scale
        n_used[key] = (n, sc)

        def measure(x, _p=path, _l=level, _n=n, _sc=sc):
            # G.measure and not G.frame: at 10M rows a frame is several GB, and
            # the LTV constants have to be measured at 10M to be measured at all
            return G.measure(_sc, seed, _n, overrides={**overrides, _p: float(x)})[_l]

        say("  %-22s target %-10.6g band [%g, %g]  on %s rows at %s pools"
            % (key, target, band[0], band[1], format(n, ","), sc))
        x, got, iters = bisect(measure, bracket[0], bracket[1], target, direction,
                               log=say if not quiet else None)
        overrides[path] = float(x)
        solved[key] = {
            # the FULL settings path, because that is what the loader overlays.
            # Keying by the short name would put a float where a block belongs.
            "path": path, "level": level,
            "value": float(x), "achieved": float(got), "target": float(target),
            "band": [float(band[0]), float(band[1])],
            "in_band": bool(band[0] <= got <= band[1]),
            "start": float(start), "moved": float(x) - float(start),
            "iterations": int(iters),
        }
        say("    solved %.6g  (start %.6g, moved %+.6g, %d iterations, level %.6g %s)"
            % (x, start, x - start, iters, got,
               "in band" if solved[key]["in_band"] else "OUT OF BAND"))
    return solved, overrides, n_used


def _read(s, path):
    node = s.raw
    for k in path.split("."):
        node = node[k]
    return node


def write(solved, scale, seed, n_used):
    """Rewrite calibrated.yaml's `solved` block, leaving the contract alone.

    The file is machine-written and the header says so. The `contract` block
    below it is hand-written and states what the solver owes; it is preserved
    verbatim, because a solver that could rewrite its own contract would be
    marking its own homework.
    """
    import hashlib
    import yaml
    text = CALIBRATED.read_text(encoding="utf-8")
    marker = "\n# ---------------------------------------------------------------------------\n"
    tail = text[text.index(marker):] if marker in text else ""

    # a partial re-solve keeps the constants it did not touch, so `--only` cannot
    # silently drop the other five from the file
    kept = (yaml.safe_load(text) or {}).get("solved") or {}
    have = {v["path"] for v in solved.values()}
    carried = {k: v for k, v in kept.items() if k not in have}

    body = ["# calibrated.yaml, machine-written by the stage 2 solver. Do not hand-edit.",
            "# Regenerate with `python -m t9v2.autocal`.",
            "meta:",
            "  solved: '%s'" % dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "  scale: %s" % (scale or "per constant, see rows_per_measurement"),
            "  seed: %d" % seed,
            "  rows_per_measurement:",
            ] + ["    %s: %d rows at %s pools" % (k, v[0], v[1]) for k, v in n_used.items()] + [
            "  checksum: '%s'" % hashlib.sha256(
                repr(sorted((k, v["value"]) for k, v in solved.items())).encode()).hexdigest()[:16],
            "solved:"]
    for k, v in solved.items():
        body += ["  %s:" % v["path"],
                 "    value: %.10g" % v["value"],
                 "    note: 'SOLVED %s against %s. Target %.6g, achieved %.6g on %s rows at %s pools in %d "
                 "iterations. Started at %.6g and moved %+.6g. %s'"
                 % (k, v["level"], v["target"], v["achieved"], format(n_used[k][0], ","), n_used[k][1],
                    v["iterations"], v["start"], v["moved"],
                    "In band." if v["in_band"] else "OUT OF BAND, reported not hidden.")]
    for path, entry in carried.items():
        body += ["  %s:" % path,
                 "    value: %.10g" % entry["value"],
                 "    note: %s" % json.dumps(entry.get("note", "carried, not re-solved"))]
    CALIBRATED.write_text("\n".join(body) + "\n" + tail, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default=None, help="override every constant's own scale")
    ap.add_argument("--seed", type=int, default=20250)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma-separated constants to re-solve; the rest are carried")
    a = ap.parse_args(argv)

    print("solving against %s, seed %d" % (a.scale, a.seed))
    only = set(a.only.split(",")) if a.only else None
    solved, _, n_used = solve(a.scale, a.seed, a.rows, only=only)
    print("\n  %-22s %-14s %-12s %s" % ("constant", "solved", "achieved", "status"))
    for k, v in solved.items():
        print("  %-22s %-14.6g %-12.6g %s"
              % (k, v["value"], v["achieved"], "in band" if v["in_band"] else "OUT OF BAND"))
    if a.dry_run:
        print("\ndry run, calibrated.yaml not written")
        return 0
    write(solved, a.scale, a.seed, n_used)
    print("\nwrote %s" % CALIBRATED)
    print("NOT solved, and no longer claiming to be: the 3 archetype LTV means.\n"
          "They aim at the same whale-concentration target as lognormal_sigma, so all 4\n"
          "against one scalar is 4 unknowns and 1 equation. Held at v1's values, routed\n"
          "`inferred` since 16 Aug 2026, and absent from the contract, so what this\n"
          "solver promises is what it delivers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
