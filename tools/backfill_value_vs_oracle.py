# -*- coding: utf-8 -*-
"""Add `value_vs_oracle` to results.json files written before it existed.

    python tools/backfill_value_vs_oracle.py --dry-run
    python tools/backfill_value_vs_oracle.py

The ratio is policy value over the oracle's value, and every results.json
already stores both, so this is arithmetic on what is on disk and not a re-run.
Nothing else is recomputed: a backfill that re-scored anything would be a new
result wearing an old seed's filename.

The `ev` block added at the same time is NOT backfillable here. It needs the
model's per-row predictions, which results.json does not keep, so seeds written
before it exists carry no `ev` until they are re-trained. That is stated rather
than papered over with a plausible-looking number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "output" / "runs"


def backfill(res):
    """Fill in the ratio for every view in one results dict. Returns n changed."""
    n = 0
    for view in res.get("views", {}).values():
        econ = view.get("economics") or {}
        ceiling = (econ.get("oracle") or {}).get("value")
        if not ceiling or ceiling <= 0:
            continue
        for pol in econ.values():
            if "value" in pol and "value_vs_oracle" not in pol:
                pol["value_vs_oracle"] = pol["value"] / ceiling
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    files = sorted(RUNS.glob("*/seed*/results.json"))
    if not files:
        print("no results under %s" % RUNS)
        return 1

    total, touched = 0, 0
    for p in files:
        res = json.loads(p.read_text(encoding="utf-8"))
        n = backfill(res)
        total += n
        if not n:
            continue
        touched += 1
        oracle = res["views"]["C1"]["economics"]["oracle"]
        print("%-6s %-9s %2d policies   oracle captured %.4f -> of oracle %.4f"
              % (res.get("scale"), "seed%s" % res.get("seed"), n,
                 oracle["value_captured"], oracle["value_vs_oracle"]))
        if not a.dry_run:
            p.write_text(json.dumps(res, indent=1, default=float),
                         encoding="utf-8")

    print("\n%d of %d results files %s, %d policy entries"
          % (touched, len(files), "would change" if a.dry_run else "updated",
             total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
