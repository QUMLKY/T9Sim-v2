# -*- coding: utf-8 -*-
"""censor(master, view), returns one of the 4 views C1 to C4.

    python -m t9v2.censor output/t9v2_1M_seed20250.parquet

Reads the observability field in graph.yaml, so the views are settings rather
than code. Nothing about which column is hidden in which view is written here;
this module only applies what the settings declare, which is what lets the
censoring map be reviewed as a table instead of read out of an if-statement.

THE VIEWS ARE MASKS, NEVER COPIES. Same rows, same order, same temporal split in
all four. Only visibility differs. That is what makes the C1-to-C4 comparison an
ablation rather than four separate experiments: if the views held different rows,
any difference between them could be the rows rather than the data layer.

Three things can happen to a column:

    kept        every cell visible
    masked      the column is there, cells on lost rows become NA
    dropped     the column does not exist in this view at all

and the difference between the last two is the difference between C2 and C1 on
one hand, and C3 and C1 on the other. MMP adds ROWS to columns you already had.
SSP adds COLUMNS you never had.

NA and not zero, and not -1. A censored cell is a fact you were not told. `-1` on
E2 and F2 is a fact you WERE told, that the event did not happen. `0` on a label
is a measured negative. Collapsing any of the three into another would quietly
change what a model trained on the view is learning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.config import load

VIEWS = ["C1", "C2", "C3", "C4"]

# Masking puts NA into a column, and a numpy integer column cannot hold one.
# Converting to float would silently turn a count into a measurement and let
# `1.0000000001` into a label column, so the nullable integer types are used
# instead: the column stays integral and gains a real missing value.
NULLABLE = {"int8": "Int8", "int16": "Int16", "int32": "Int32", "int64": "Int64"}


def plan(settings, view):
    """What happens to each column in this view: kept, masked or dropped.

    Returned as data rather than applied directly, so the same decision can be
    printed, tested and checked without generating anything.
    """
    if view not in VIEWS:
        raise ValueError("view must be one of %s; got %r" % (VIEWS, view))
    keep, mask, drop = [], [], []
    for n in settings.emitting():
        obs = n["observability"]
        for c in n["columns"]:
            if obs == "latent":
                drop.append(c)
            elif obs[view] == "all":
                keep.append(c)
            elif obs[view] == "won":
                mask.append(c)
            elif obs[view] == "none":
                drop.append(c)
            else:
                raise ValueError("column %r has row rule %r in %s, which censor "
                                 "does not implement" % (c, obs[view], view))
    return {"keep": keep, "mask": mask, "drop": drop}


def censor(master, view, settings=None):
    """One view of the master. Column order is preserved from the master."""
    s = settings or load("default")
    p = plan(s, view)
    dropped = set(p["drop"])
    out = master[[c for c in master.columns if c not in dropped]].copy()

    if p["mask"]:
        if "won" not in out.columns:
            raise RuntimeError(
                "view %s masks %d column(s) on lost rows but does not carry `won`, "
                "so there is nothing to mask against. `won` is always visible per "
                "specification 1.5." % (view, len(p["mask"])))
        lost = out["won"].to_numpy() == 0
        for c in p["mask"]:
            dt = str(out[c].dtype)
            if dt in NULLABLE:
                out[c] = out[c].astype(NULLABLE[dt])
            out.loc[lost, c] = None
    return out


def all_views(master, settings=None):
    s = settings or load("default")
    return {v: censor(master, v, s) for v in VIEWS}


def summary(settings=None):
    """The censoring map as a table, one row per view. Reads the settings only."""
    s = settings or load("default")
    rows = []
    for v in VIEWS:
        p = plan(s, v)
        rows.append({"view": v, "kept": len(p["keep"]), "masked": len(p["mask"]),
                     "dropped": len(p["drop"]),
                     "columns": len(p["keep"]) + len(p["mask"])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("master", help="the master parquet from t9v2.generate")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-write", action="store_true",
                    help="report the map and the counts, write nothing")
    a = ap.parse_args(argv)

    import pandas as pd
    s = load("default")
    src = Path(a.master)
    out_dir = Path(a.out_dir) if a.out_dir else src.parent

    print("censoring map, from graph.yaml")
    print("  %-5s %7s %7s %7s %8s" % ("view", "kept", "masked", "dropped", "columns"))
    for r in summary(s):
        print("  %-5s %7d %7d %7d %8d"
              % (r["view"], r["kept"], r["masked"], r["dropped"], r["columns"]))

    if a.no_write:
        return 0

    master = pd.read_parquet(src)
    print("\n%s, %d rows" % (src.name, len(master)))
    for v in VIEWS:
        d = censor(master, v, s)
        hidden = int(d[plan(s, v)["mask"][0]].isna().sum()) if plan(s, v)["mask"] else 0
        path = out_dir / ("%s_%s.parquet" % (src.stem, v))
        d.to_parquet(path, index=False)
        print("  %-3s %2d columns, %8d cells hidden per masked column -> %s"
              % (v, len(d.columns), hidden, path.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
