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

  SKIP ONLY WHAT IS COMPLETE, and complete means every artifact the seed owes,
  produced by the CURRENT design. `is_complete` is the enforcement point for
  both rules, and neither is a matter of prose: a seed killed during training
  leaves a valid parquet and no results, and a seed finished before a generator
  change leaves a full set of files that describe data the current code would
  never produce. Both read as done to any laxer check.

  DISK GUARD. A 10M seed is about 1.7 GB of parquet plus roughly 0.5 GB of
  bundles and eval files, so the driver refuses to start one unless free space
  exceeds 8 GB plus 2.5 GB for every 10M seed still to run. It stops cleanly
  with a message rather than dying mid-write and leaving a truncated parquet
  that looks like a real one.

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

# Reserve for each 10M seed still to run. RAISED FROM 2.0 WHEN THE SEED GREW.
# v2 wrote one 1.7 GB parquet per seed and 2.0 left about 0.3 GB of headroom.
# A v2.2 seed also writes four model bundles and four per-row eval files.
#
# MEASURED AT 1M, 23 August 2026, which is the scale that projects honestly:
#
#     master 179 MB    bundle 55 MB    eval 40 MB
#
# The three scale differently. The master and the eval files go with ROWS, so
# both are ten times larger at 10M. The bundle does not: a boosted model is
# bounded by its round count, and only the encoder cell tables grow, with the
# POOLS -- 500 apps at 1M against 1200 at 10M. So a 10M seed is about
#
#     1700 + 400 + 130 = 2.23 GB
#
# against this 2.5 GB reserve, leaving roughly 270 MB. An earlier version of
# this comment projected from 100K and got 2.0 GB, which was wrong: at 100K the
# bundles dominate the seed directory and the eval files are a rounding error,
# so the mix there says nothing about the mix at scale.
PER_10M_GB = 2.5


def seed_dir(scale, seed):
    return RUNS / scale / ("seed%d" % seed)


def master_path(scale, seed):
    return OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))


def current_fingerprint():
    """The design hash the code and settings ON DISK RIGHT NOW would produce.

    Imported inside the function rather than at module scope: the driver runs as
    a thin parent process around a subprocess per seed, and `generate` pulls in
    pandas, numpy and the whole generator to answer a question about hashes.
    """
    from . import generate as G
    from .core.config import load
    return G.fingerprint(load("default"))


# every file a finished seed owes. Named here rather than assembled at each call
# site so that adding an artifact to the run adds it to the completeness test in
# the same edit.
def artifacts(scale, seed):
    d = seed_dir(scale, seed)
    out = [master_path(scale, seed),
           master_path(scale, seed).with_suffix(".manifest.json"),
           d / "results.json",
           d / "eval" / "manifest.json"]
    for v in VIEWS:
        out.append(d / "bundle" / v / "manifest.json")
        out.append(d / "eval" / ("%s.parquet" % v))
    return out


def why_incomplete(scale, seed, fingerprint):
    """None if the seed is done, otherwise the first reason it is not.

    TWO FAILURES, ONE CHECK. A seed killed during training leaves a valid parquet
    and no results; a seed finished before a generator change leaves a complete
    set of files describing data the current code would never produce. The first
    would quietly make a 30-dataset campaign 29 while it reported 30. The second
    is how thirty pre-floor-fix seeds would read as complete, and it is the one
    that does not announce itself, because nothing about those files is broken.

    `fingerprint` is required and has no default. A default would make the
    staleness check something a caller can forget rather than something it has
    to answer for, and a forgotten check is the state this function replaces.
    """
    p = seed_dir(scale, seed) / "results.json"
    if not p.exists():
        return "no results.json"
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "results.json does not parse"
    missing = [v for v in VIEWS if v not in r.get("views", {})]
    if missing:
        return "results.json is missing %s" % ", ".join(missing)
    have = r.get("design_fingerprint")
    if have is None:
        return "results.json predates design fingerprinting"
    if have != fingerprint:
        return "design %s, current is %s" % (have, fingerprint)
    for f in artifacts(scale, seed):
        if not f.exists():
            try:
                name = str(f.relative_to(ROOT))
            except ValueError:
                name = str(f)
            return "no %s" % name
    return None


def is_complete(scale, seed, fingerprint):
    return why_incomplete(scale, seed, fingerprint) is None


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
    from .evalfile import write_eval
    from .train.runner import run_seed as train_all

    t0 = time.time()
    d = seed_dir(scale, seed)
    d.mkdir(parents=True, exist_ok=True)
    s = load("default")

    path = master_path(scale, seed)
    # reuse only what the CURRENT design produced. Keeping parquets saves the
    # generation time; reusing one from a superseded design would train on the
    # wrong data and report it as a result.
    current, why = G.parquet_is_current(path, s)
    if not current:
        if path.exists():
            print("  regenerating %s: %s" % (path.name, why), flush=True)
        G.generate(scale=scale, seed=seed, quiet=quiet)
    t_gen = time.time() - t0

    # ONE FIT, THREE OUTPUTS. The four views are fitted once; that same pass
    # freezes each bundle before it releases the model, and the eval pass then
    # LOADS those bundles rather than refitting. A second fit would not
    # reproduce the first — early stopping lands elsewhere — so the numbers in
    # results.json and the numbers in the eval files would quietly be from
    # different models.
    master = pd.read_parquet(path)
    views = train_all(master, s, seed=0, quiet=quiet, bundle_dir=d / "bundle")
    write_eval(master, d / "bundle", d / "eval", s, quiet=quiet)
    del master

    out = {
        "scale": scale, "seed": seed, "views": views,
        "rows": int(s.raw["scales"][scale]),
        # what design produced the data underneath these numbers. Without it a
        # results file cannot say which generator it belongs to, and the reuse
        # path has no way to tell a finished seed from a superseded one.
        "design_fingerprint": G.fingerprint(s),
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

    fp = current_fingerprint()
    say("campaign %s, %d seeds, design %s, free disk %.1f GB"
        % (scale, len(seeds), fp, free_gb()))
    done, skipped, failed, t0 = [], [], [], time.time()

    for i, seed in enumerate(seeds):
        why = why_incomplete(scale, seed, fp)
        if why is None and not force:
            skipped.append(seed)
            say("  seed %d  already complete, skipped" % seed)
            continue
        if why is not None and (seed_dir(scale, seed) / "results.json").exists():
            say("  seed %d  re-running: %s" % (seed, why))
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
        after = why_incomplete(scale, seed, fp)
        if r.returncode != 0 or after is not None:
            failed.append(seed)
            say("  seed %d  FAILED after %.0fs: %s"
                % (seed, time.time() - t,
                   (r.stderr or "").strip()[-300:] or after))
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
    """What the campaign has, against the 30 it owes.

    Prints the REASON a seed is short rather than only its number, because the
    two reasons need different work: "no eval/C3.parquet" is a re-run, and
    "design abc, current is def" means the generator moved under the whole
    scale and none of it can be reported beside the rest.
    """
    fp = current_fingerprint()
    print("design fingerprint %s\n" % fp)
    print("%-6s %-9s %s" % ("scale", "complete", "seeds"))
    total, reasons = 0, {}
    for sc in SCALES:
        have = []
        for s in SEEDS:
            why = why_incomplete(sc, s, fp)
            if why is None:
                have.append(s)
            else:
                reasons.setdefault(why, []).append("%s/%d" % (sc, s))
        total += len(have)
        miss = [s for s in SEEDS if s not in have]
        print("%-6s %d of %-6d %s" % (sc, len(have), len(SEEDS),
                                      ("missing " + str(miss)) if miss else "all"))
    if reasons:
        print()
        for why, who in sorted(reasons.items()):
            print("  %-44s %s" % (why, ", ".join(who)))
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
