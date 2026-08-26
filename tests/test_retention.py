# -*- coding: utf-8 -*-
"""No code path deletes a run's artifacts, and the check is a walk of the source.

WHY A TEST AND NOT A RULE. The rule already existed in prose and did not hold:
`run_arm1.py` deleted the master parquets and destroyed the data behind the
`v2-corrected-1M` results, which then could not be reproduced or re-scored. A
sentence in a document is read by whoever happens to read it; this runs on every
commit.

WHAT IT ACTUALLY ENFORCES, in three parts.

  NO NEW DELETE CALL. Every `rmtree` / `remove` / `unlink` in `src/` and
  `tools/` is counted per file against a fixed budget with a written reason.
  Adding one anywhere fails here, including in a file that already has some, and
  the fix is to justify it in ALLOWED rather than to raise a number.

  NO OPT-IN FLAG. No `--clean`, `--purge`, `--delete`. A flag is how a deletion
  becomes routine: it exists for one bad afternoon, and afterwards it is in
  somebody's shell history.

  THE ARTIFACT LIST IS THE COMPLETENESS TEST. `campaign.artifacts` is the same
  list `why_incomplete` walks, so an artifact that can be deleted without the
  seed reading incomplete is one this project does not have.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# name -> (calls allowed, why). A file absent from this map may have none.
ALLOWED = {
    "src/t9v2/bundle.py": (
        3, "save_bundle builds in <name>.partial and swaps: it clears its own "
           "staging leftover, and the superseded bundle only once the "
           "replacement is whole on disk"),
    "tools/make_public.py": (
        2, "the publish target under release/ is rebuilt from scratch each time, "
           "and it is not a run artifact; .git is skipped explicitly"),
}

DELETERS = {"rmtree", "remove", "unlink", "rmdir", "removedirs"}
BANNED_FLAG_STEMS = ("delete", "clean", "purge", "wipe", "prune", "discard",
                     "erase", "reset")


def sources():
    out = []
    for sub in ("src", "tools"):
        for p in sorted((ROOT / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def delete_calls(tree):
    """Every call whose function name is one of DELETERS, as (name, lineno)."""
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in DELETERS:
            hits.append((name, n.lineno))
    return hits


@pytest.fixture(scope="module")
def parsed():
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in sources()]


def test_the_source_has_no_unaccounted_delete(parsed):
    found = {}
    for p, tree in parsed:
        hits = delete_calls(tree)
        if hits:
            found[rel(p)] = hits

    over = []
    for name, hits in sorted(found.items()):
        budget = ALLOWED.get(name, (0, "not allowed to delete anything"))[0]
        if len(hits) != budget:
            over.append("%s: %d delete call(s) at lines %s, budget %d"
                        % (name, len(hits), [ln for _, ln in hits], budget))
    assert not over, (
        "delete calls changed. Every one must be justified in ALLOWED, and "
        "raising a number is not a justification:\n  " + "\n  ".join(over))


def test_every_allowed_file_still_exists(parsed):
    """A budget for a file that has been renamed away is a hole, not a rule."""
    have = {rel(p) for p, _ in parsed}
    gone = sorted(set(ALLOWED) - have)
    assert not gone, "ALLOWED names files that no longer exist: %s" % gone


def test_nothing_deletes_under_output(parsed):
    """No delete call anywhere mentions `output` or `runs` in the same statement.

    A coarse check on purpose. It cannot resolve a path built three functions
    away, so it is a tripwire on the obvious form rather than a proof; the
    budget test above is what actually holds the line.
    """
    bad = []
    for p, tree in parsed:
        src = p.read_text(encoding="utf-8").splitlines()
        for name, ln in delete_calls(tree):
            line = src[ln - 1]
            if "output" in line.lower() or "runs" in line.lower():
                bad.append("%s:%d  %s" % (rel(p), ln, line.strip()))
    assert not bad, "a delete names an output path:\n  " + "\n  ".join(bad)


def test_there_is_no_opt_in_delete_flag(parsed):
    """No command-line switch offers to remove anything."""
    bad = []
    for p, tree in parsed:
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"):
                continue
            for a in n.args:
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                    continue
                stem = a.value.lstrip("-").replace("-", "_").lower()
                if any(b in stem for b in BANNED_FLAG_STEMS):
                    bad.append("%s:%d  %s" % (rel(p), n.lineno, a.value))
    assert not bad, ("a flag offers to delete:\n  " + "\n  ".join(bad)
                     + "\nNo delete flag exists; do not add one.")


def test_the_artifact_list_is_what_completeness_checks():
    """The retention rule and the skip rule read the same list.

    If they could differ, an artifact could be missing without the seed reading
    incomplete, which is exactly the state that lets a deletion go unnoticed.
    """
    from t9v2 import campaign as C
    names = [f.name for f in C.artifacts("100K", 20250)]
    assert "t9v2_100K_seed20250.parquet" in names
    assert "t9v2_100K_seed20250.manifest.json" in names
    assert names.count("results.json") == 1
    # the master's manifest carries the seed in its name; these 5 are the four
    # bundles and the eval pass's own
    assert names.count("manifest.json") == 5
    assert sorted(n for n in names if n.endswith(".parquet")) == [
        "C1.parquet", "C2.parquet", "C3.parquet", "C4.parquet",
        "t9v2_100K_seed20250.parquet"]
    assert len(names) == 12
