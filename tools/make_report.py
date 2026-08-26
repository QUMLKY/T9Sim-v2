# -*- coding: utf-8 -*-
"""Rebuild every results document, in order, with one command.

    python tools/make_report.py                 # everything
    python tools/make_report.py --check         # say what would run, write nothing
    python tools/make_report.py --no-docx       # markdown only, no Word needed
    python tools/make_report.py --force-allrows # recompute the all-rows pass anyway

WHY THIS EXISTS. Reproducing the report used to be five commands in a fixed
order, two of them needing a DIFFERENT PYTHON, and one flag that changes the
appearance of the finished document and is off by default. A chain like that is
reproducible in the sense that a person who remembers all of it can reproduce
it. The documents that came out of it were the record of a dissertation, so
"remembers all of it" was not good enough.

THE ORDER IS NOT COSMETIC, and one edge of it is a correctness problem rather
than an inconvenience:

    1. tier1_allrows.py    scores the funnel heads on every test row
    2. results_report.py   READS that output. Without it, three rows of the
                           results table silently disappear
    3. results_v1_layout.py  the v1-shaped contrast tables, written directly
    4. three docx exports through safe_docx_export

STALENESS IS THE DANGEROUS CASE, NOT ABSENCE. A missing `tier1_allrows.json`
costs three rows and `results_report.py` drops them rather than printing `nan`,
so the failure is visible. A STALE one is silent: re-run the campaign and
yesterday's AUCs are printed beside today's economics with nothing to notice. So
step 1 is not skipped because it is slow, it is skipped only when its output is
NEWER than every `results.json` it was computed from. Anything else recomputes.
`--force-allrows` recomputes regardless; `--check` reports the decision.

ONE INTERPRETER, since 24 August 2026. Everything runs in `t9v2/venv`, which
carries `pypandoc_binary` and `python-docx` beside xgboost and pandas. It took
two until then, and the reason was bad; the comment beside `PY` below records
what changed and why. `--no-docx` still stops after the markdown, because the
markdown is the record and the Word files are a rendering of it.

WHEN WORD BLOCKS A REBUILD there are two different refusals and only one is a
problem. A document OPEN in Word cannot be written at all, so close it. A
document the guard calls EDITED has moved since it was generated, which is also
what Word does merely by OPENING it to read. The guard cannot tell that from a
real edit and is right to refuse. Check `git diff` on the docx, and if nothing
real changed, re-run with `--force`.

WHAT IT WILL NOT DO. It does not train, generate or re-score anything. Every
number it writes comes from `output/runs/*/seed*/results.json`, which this script
never touches. If a number is wrong, this is not the thing that made it wrong.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # t9v2/
REPO = ROOT.parent                      # the worktree root
RUNS = ROOT / "output" / "runs"
GATES = ROOT / "docs" / "gates"

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# (markdown source, docx name at the repo root, extra flags)
DOCX = [
    (ROOT / "docs" / "V2.2_Results.md", "Training_Results v2.2.docx",
     ["--shade-sections"]),
    (ROOT / "docs" / "V2.2_Analysis.md", "V2.2 Analysis.docx", []),
    # The short form. Same findings, selected rather than summarised, written to
    # be read once before the dissertation edits rather than kept as a record.
    (ROOT / "docs" / "V2.2_Analysis_Brief.md", "V2.2 Analysis Brief.docx", []),
    # THE AFT-ONLY PAIR, Ken's decision of 24 August 2026. The two-bidder
    # documents above are kept and still built: they are cited, and a reader who
    # finds one table where they remember two has to be able to see both.
    (ROOT / "docs" / "V2.2_Results_AFT.md", "Training_Results v2.2 AFT.docx",
     ["--shade-sections"]),
    (ROOT / "docs" / "V2.2_Analysis_Brief_AFT.md", "V2.2 Analysis Brief AFT.docx", []),
]

# ONE INTERPRETER. `t9v2/venv` now carries `pypandoc_binary` and `python-docx`
# alongside xgboost and pandas, so every step of this chain runs in it.
#
# It used to take two, and the reason was bad. The docx export needs pypandoc,
# which lived only in v1's frozen `t9_sim/venv`, and this script hunted for an
# interpreter that could import it. The stated justification was that v2.2's
# environment should not grow a Word dependency -- which does not survive
# scrutiny, because installing into v2.2's venv does not touch v1's at all. Two
# pip installs on 24 August 2026 deleted the hunt, the fallback path and the
# class of confusion where a report is written by whichever python was first on
# PATH. `pypandoc_binary` bundles pandoc itself, so nothing outside the venv is
# needed either.
#
# Checked rather than assumed: numpy, pandas, pyarrow, scipy and xgboost are all
# at the versions they were before the install.
PY = ROOT / "venv" / "Scripts" / "python.exe"

# THE REPORTED TARGET IS READ FROM THE SETTINGS, NOT WRITTEN HERE, so this file
# cannot drift from config/training.yaml the way a second copy of the number
# would. The two AFT steps are passed it explicitly and re-score their bidder
# rows from docs/roas_sweep_<scale>.json; every other row comes from the runs.
#
# WITHOUT THIS THE CHAIN QUIETLY UNDID THE CHANGE. The thirty results.json hold
# economics at a target of 1.0, because re-scoring them in place needs about
# eight gigabytes a seed at 10M and an attempt at it was rolled back on 25
# August 2026. So a plain `make_report` run rebuilt both AFT documents at 1.0
# on top of the 3.0 ones, with nothing to notice: the tables look finished
# either way. The two `both` steps are NOT passed it, because the sweep does
# not score the win classifier and its rows cannot be re-scored.
def _reported_roas():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from t9v2.core.config import load as _load
    from t9v2.train import bidders as _B
    return float(_B.roas_target(_load("default")))


ROAS = _reported_roas()


def allrows_is_current(scale):
    """Is `tier1_allrows.json` newer than every results.json it summarises?

    mtime rather than a content hash, deliberately. The question is "was this
    computed after the data it describes", and a re-run of the campaign always
    rewrites results.json. A hash would be stronger and would also need the
    all-rows pass to record one, which it does not; this catches the case that
    actually happens.

    Returns (current, reason).
    """
    p = GATES / "tier1_allrows.json"
    if not p.exists():
        return False, "no tier1_allrows.json"
    import json
    try:
        have = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "tier1_allrows.json does not parse"
    if scale not in have:
        return False, "no %s section in tier1_allrows.json" % scale
    results = sorted((RUNS / scale).glob("seed*/results.json"))
    if not results:
        return False, "no %s seeds on disk" % scale
    newest = max(r.stat().st_mtime for r in results)
    if p.stat().st_mtime < newest:
        return False, "STALE: a results.json is newer than it"
    missing = [r.parent.name[4:] for r in results
               if r.parent.name[4:] not in have[scale]]
    if missing:
        return False, "missing seeds %s" % ", ".join(missing)
    return True, "current, %d seeds" % len(have[scale])


def run(cmd, label, check_only):
    print("  %-34s %s" % (label, "would run" if check_only else "running"))
    if check_only:
        return True
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode != 0:
        print("  !! %s FAILED, exit %d" % (label, r.returncode))
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="10M",
                    help="the scale the all-rows pass covers (default 10M, "
                         "which is what the dissertation reports)")
    ap.add_argument("--check", action="store_true",
                    help="report the plan and write nothing")
    ap.add_argument("--no-docx", action="store_true")
    ap.add_argument("--force-allrows", action="store_true")
    # PASSED THROUGH TO THE EXPORT GUARD, and worth knowing when to use.
    #
    # `safe_docx_export` refuses to overwrite a document whose bytes have moved
    # since it generated them, because that is what an edit in Word looks like.
    # Merely OPENING a docx in Word also moves them: Word rewrites parts of the
    # package on open, and the guard cannot tell that from a real edit. So a
    # document Ken has read, and not changed, blocks the next rebuild.
    #
    # The guard is right to refuse and this flag is the human saying "I know
    # there are no edits in there". It is not a default and must not become one:
    # the same refusal is the only thing standing between a hand-corrected
    # document and a script that overwrites it. Check first -- the guard takes a
    # backup either way, and `git diff` on the docx says whether anything real
    # changed.
    ap.add_argument("--force", action="store_true",
                    help="overwrite documents the guard reports as edited. Use "
                         "when Word merely opened them, never to discard edits")
    a = ap.parse_args(argv)

    if not PY.exists():
        sys.exit("no interpreter at %s" % PY)

    ok = True
    print("1. the all-rows pass")
    current, why = allrows_is_current(a.scale)
    if a.force_allrows:
        current, why = False, "--force-allrows"
    print("     %s" % why)
    if current:
        print("  %-34s skipped" % "tier1_allrows.py")
    else:
        ok &= run([PY, HERE / "tier1_allrows.py", "--scale", a.scale],
                  "tier1_allrows.py", a.check)

    print("2. the markdown, both bidder sets")
    ok &= run([PY, HERE / "results_report.py"], "results_report.py both", a.check)
    ok &= run([PY, HERE / "results_report.py", "--bidders", "aft",
               "--roas", "%g" % ROAS], "results_report.py aft", a.check)

    if a.no_docx:
        print("   --no-docx, stopping after the markdown")
        return 0 if ok else 1

    print("3. the v1-shaped docx")
    ok &= run([PY, HERE / "results_v1_layout.py"], "results_v1_layout.py both",
              a.check)
    ok &= run([PY, HERE / "results_v1_layout.py", "--bidders", "aft",
               "--roas", "%g" % ROAS], "results_v1_layout.py aft", a.check)

    print("4. the docx exports")
    for md, name, flags in DOCX:
        if not md.exists():
            print("  %-34s skipped, no %s" % (name, md.name))
            continue
        ok &= run([PY, REPO / "tools" / "safe_docx_export.py",
                   md.relative_to(REPO), name]
                  + flags + (["--force"] if a.force else []), name, a.check)

    print()
    if ok:
        print("report rebuild: OK")
        return 0
    print("report rebuild: SOMETHING FAILED, see above")
    print("  A document open in Word cannot be written: close it and re-run.")
    print("  A document the guard calls EDITED has moved since it was generated,")
    print("  which is also what Word does merely by opening it. Check `git diff`")
    print("  on the docx, and if nothing real changed, re-run with --force.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
