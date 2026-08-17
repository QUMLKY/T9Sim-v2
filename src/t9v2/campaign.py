# -*- coding: utf-8 -*-
"""The campaign driver: 30 datasets, 10 seeds each at 100K, 1M and 10M.

    python -m t9v2.campaign --scale 100K
    python -m t9v2.campaign --scale 10M --seeds 20250
    python -m t9v2.campaign --status

Every seed is generated, censored into 4 views, trained, evaluated and written to
its own folder. Steps 1 to 4 of the plan's campaign are this module; aggregation
is stage 6.

THE COUNT RULE: 30 datasets, nothing dropped. THE SEED LIST NEVER CHANGES: 20250
to 20259, fixed before the campaign starts. Generation is deterministic, so a
re-run gives byte-identical data, and swapping a seed until a check passes would
be selecting on the outcome.

Four guards make the unattended 10M run safe, and each is here because a 16 GB
laptop running for three hours unattended fails in a specific way:

  FRESH PROCESS PER SEED. The driver spawns a subprocess per seed and waits. A
  10M seed holds about 5 GB at its peak, and Python does not reliably return that
  to the OS, so a loop in one process climbs until it dies on seed 4. Only a
  process exit is a guarantee.

  SKIP ONLY WHAT IS COMPLETE. A seed counts as done when its `results.json` is
  present, PARSES, and holds all 4 views. Generation output alone is not enough:
  a seed killed during training leaves a valid parquet and no results, and
  skipping it would silently drop a dataset from a campaign whose whole claim is
  that it dropped none.

  DISK GUARD. A 10M parquet is about 1.7 GB. The driver refuses to start a seed
  unless free space exceeds 8 GB plus 2 GB for every 10M seed still to run, so it
  stops cleanly with a message rather than dying mid-write and leaving a truncated
  parquet that looks like a real one.

  LOGGING. Everything is appended to a per-scale log, so an unattended run that
  failed at 2 a.m. can still be read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
RUNS = OUTPUT / "runs"
SEEDS = [20250, 20251, 20252, 20253, 20254, 20255, 20256, 20257, 20258, 20259]
SCALES = ["100K", "1M", "10M"]
VIEWS = ["C1", "C2", "C3", "C4"]

FLOOR_GB = 8.0            # never go below this, whatever the scale
PER_10M_GB = 2.0          # reserve for each 10M seed still to run


def seed_dir(scale, seed):
    return RUNS / scale / ("seed%d" % seed)


def is_complete(scale, seed):
    """A seed is done only when its results parse and cover all 4 views.

    Deliberately strict. The failure this guards against is a seed killed during
    training: the parquet is on disk and looks finished, so a laxer check would
    skip it and the campaign would quietly be 29 datasets while reporting 30.
    """
    p = seed_dir(scale, seed) / "results.json"
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return all(v in r.get("views", {}) for v in VIEWS)


def free_gb(path=None):
    return shutil.disk_usage(str(path or ROOT)).free / 1e9


def disk_ok(scale, remaining_10m):
    need = FLOOR_GB + PER_10M_GB * (remaining_10m if scale == "10M" else 0)
    have = free_gb()
    return have >= need, have, need


def run_one(scale, seed, quiet=False):
    """Generate, train and evaluate ONE seed. Called in its own process."""
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd

    from . import generate as G
    from .core.config import load
    from .train.runner import run_seed as train_all

    t0 = time.time()
    d = seed_dir(scale, seed)
    d.mkdir(parents=True, exist_ok=True)
    s = load("default")

    path = OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))
    # reuse only what the CURRENT design produced. Keeping parquets saves the
    # generation time; reusing one from a superseded design would train on the
    # wrong data and report it as a result.
    current, why = G.parquet_is_current(path, s)
    if not current:
        if path.exists():
            print("  regenerating %s: %s" % (path.name, why), flush=True)
        G.generate(scale=scale, seed=seed, quiet=quiet)
    t_gen = time.time() - t0

    master = pd.read_parquet(path)
    views = train_all(master, s, seed=0, quiet=quiet)
    del master

    out = {
        "scale": scale, "seed": seed, "views": views,
        "rows": int(s.raw["scales"][scale]),
        "generated_seconds": round(t_gen, 1),
        "total_seconds": round(time.time() - t0, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (d / "results.json").write_text(json.dumps(out, indent=1, default=float),
                                   encoding="utf-8")
    return out


def campaign(scale, seeds=None, force=False, log=None):
    """Every seed at one scale, each in a fresh process."""
    seeds = seeds or SEEDS
    RUNS.mkdir(parents=True, exist_ok=True)
    logp = log or (RUNS / ("campaign_%s.log" % scale))
    logp.parent.mkdir(parents=True, exist_ok=True)

    def say(msg):
        line = "%s  %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with logp.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    say("campaign %s, %d seeds, free disk %.1f GB" % (scale, len(seeds), free_gb()))
    done, skipped, failed, t0 = [], [], [], time.time()

    for i, seed in enumerate(seeds):
        if is_complete(scale, seed) and not force:
            skipped.append(seed)
            say("  seed %d  already complete, skipped" % seed)
            continue
        ok, have, need = disk_ok(scale, len(seeds) - i)
        if not ok:
            say("  STOPPING: %.1f GB free, %.1f GB needed for the %d seeds left"
                % (have, need, len(seeds) - i))
            break

        t = time.time()
        # a fresh process per seed: only an exit reliably returns 5 GB to the OS
        r = subprocess.run([sys.executable, "-m", "t9v2.campaign",
                            "--one", "--scale", scale, "--seed", str(seed)],
                           cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0 or not is_complete(scale, seed):
            failed.append(seed)
            say("  seed %d  FAILED after %.0fs: %s"
                % (seed, time.time() - t, (r.stderr or "").strip()[-300:]))
            continue
        res = json.loads((seed_dir(scale, seed) / "results.json").read_text(encoding="utf-8"))
        v = res["views"]
        done.append(seed)
        say("  seed %d  %.0fs  click C1 %.4f C2 %.4f | win C1 %.4f C3 %.4f | "
            "profit C1 %.0f C2 %.0f"
            % (seed, res["total_seconds"],
               v["C1"]["heads"]["click"]["auc"], v["C2"]["heads"]["click"]["auc"],
               v["C1"]["heads"]["win"]["auc"], v["C3"]["heads"]["win"]["auc"],
               v["C1"]["economics"]["learned"]["profit"],
               v["C2"]["economics"]["learned"]["profit"]))

    say("%s: %d done, %d already complete, %d failed, %.0f min total"
        % (scale, len(done), len(skipped), len(failed), (time.time() - t0) / 60))
    if failed:
        say("  FAILED SEEDS, re-run rather than drop: %s" % failed)
    return {"done": done, "skipped": skipped, "failed": failed}


def status():
    """What the campaign has, against the 30 it owes."""
    print("%-6s %-9s %s" % ("scale", "complete", "seeds"))
    total = 0
    for sc in SCALES:
        have = [s for s in SEEDS if is_complete(sc, s)]
        total += len(have)
        miss = [s for s in SEEDS if s not in have]
        print("%-6s %d of %-6d %s" % (sc, len(have), len(SEEDS),
                                      ("missing " + str(miss)) if miss else "all"))
    print("\n%d of 30 datasets complete.  free disk %.1f GB" % (total, free_gb()))
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="100K", choices=SCALES)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated; default is all 10, and the list never changes")
    ap.add_argument("--force", action="store_true", help="re-run seeds already complete")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.status:
        status()
        return 0
    if a.one:
        r = run_one(a.scale, a.seed, quiet=True)
        print("seed %d done in %.0fs" % (a.seed, r["total_seconds"]))
        return 0
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    out = campaign(a.scale, seeds, force=a.force)
    return 1 if out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
