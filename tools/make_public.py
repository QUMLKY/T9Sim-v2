# -*- coding: utf-8 -*-
"""Assemble the publishable tree, design documents included, without copying
anything into the working tree.

    python tools/make_public.py                     # build it
    python tools/make_public.py --dry-run           # list what would be copied
    python tools/make_public.py --out ../somewhere  # a different destination

Default destination is `release/public_repo_v2/T9Sim-v2/`, OUTSIDE `t9v2/` and
outside anything v2 tracks.

WHY A SCRIPT AND NOT A COPY. The documents v2 was built from — the
specification, the plan, the symbol plan, the node register — live in the parent
repository's `docs/`, and they are still being edited. Copying them into `t9v2/`
would create a second copy that starts drifting the day it is made, and the drift
would be silent: a reviewer would read the copy believing it was current. The
copy therefore happens at PUBLISH time, into a staging tree that is rebuilt from
scratch each run, so what is published is whatever the source said at that
moment and there is never a second master.

WHY THE HISTORY IS NOT CARRIED. The working repository's history contains
personal files and can never be published. The staging tree is plain files with
no `.git`, so publishing is `git init` in the destination and a first commit.
This is how v1 was published and the reason has not changed.

FILENAMES ARE MATCHED BY STEM, NEVER HARDCODED. The `.docx` and `.xlsx` twins
live in the repository root and their names drift — spacing, capitalisation, a
"- Copy" suffix appearing beside the real one. Each is found by normalising both
names to lowercase alphanumerics and matching, with "copy" variants skipped, so a
rename upstream does not silently publish a stale twin or drop a file.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
REPO = V2.parent
DEFAULT_OUT = REPO / "release" / "public_repo_v2" / "T9Sim-v2"

# The design documents v2 is built FROM. Each markdown source, with the root
# twin Ken reads found by stem rather than by name.
# THE v2.2 SET, and the version suffixes are load-bearing. `T9Sim_Rebuild_Plan_v2.md`
# sat here until 26 August 2026, so the published tree carried v2's plan beside
# v2.2's specification and node register. Both files exist in docs/ and differ,
# and nothing would have failed: the copy succeeds either way and the reader
# finds a plan that does not describe what was built.
DESIGN = [
    "T9Sim_Specification_v2.2.md",
    "T9Sim_Rebuild_Plan_v2.2.md",
    "T9Sim_Symbol_Plan.md",
    "T9Sim_DGP_Node_Register_v2.2.md",
]

# SUPERSEDED, SO NOT PUBLISHED. Ken's instruction of 26 August 2026, and the
# list is wider than it first looks. v2.2 changed the generator, the bidder and
# the reported target, so a v2 results document beside a v2.2 one compares two
# simulators rather than two data layers. These are kept in the working
# repository, where they are the trail; they are not part of the release.
#
#   docs/Archive/            5 documents superseded during v2.2 itself,
#                            including the win classifier's results, retired
#                            from the report on 24 August
#   docs/V2_Results*.md      v2's results
#   docs/HANDOVER.md         its "what it found" is v2's SSP flat null and the
#                            SHAP redundancy reading, both dead. The rest of it
#                            is current and useful, so this is an exclusion
#                            pending a rewrite rather than a judgement on the
#                            document.
#
# Matched as path prefixes against git's own listing, so a file added to
# docs/Archive/ later is excluded without anyone remembering to add it here.
EXCLUDE = (
    "docs/Archive/",
    "docs/V2_Results.md",
    "docs/V2_Results_Merged_CI.md",
    "docs/HANDOVER.md",
)

# Never published, whatever else matches.
SKIP = re.compile(r"(^|[^a-z])copy([^a-z]|$)|~\$", re.I)


def norm(name):
    """A filename reduced to its comparable stem: lowercase alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def find_twin(md_name):
    """The .docx or .xlsx in the repo root that pairs with this markdown.

    Matched on the normalised stem so `T9Sim_DGP_Node_Register_v2.2.md` finds
    `T9Sim DGP Node Register.xlsx` whatever the spacing or capitalisation, and
    a `- Copy` beside the real file is skipped rather than published.
    """
    want = norm(md_name)
    hits = [p for p in list(REPO.glob("*.docx")) + list(REPO.glob("*.xlsx"))
            if norm(p.name) == want and not SKIP.search(p.name)]
    if len(hits) > 1:
        raise SystemExit("ambiguous twin for %s: %s"
                         % (md_name, [p.name for p in hits]))
    return hits[0] if hits else None


# Markdown image links in the design documents point at the working tree by
# ABSOLUTE WINDOWS PATH, e.g. C:/Users/.../Schema%20diagrams/T9Sim_Generator.png.
# That is fine where they are authored and useless once published: the reader
# gets a broken image and the publisher's directory layout. Found on 26 August
# 2026 in the specification, 3 links, in a tree that had never been built.
#
# REWRITTEN AT PUBLISH TIME, NEVER IN THE SOURCE, for the same reason the
# documents are copied at publish time rather than duplicated. The source keeps
# its absolute paths and stays editable where it lives; the published copy gets
# `figures/NAME.png` beside a `design/figures/` directory holding the file.
IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def rewrite_figures(text, repo):
    """Localise every image link. Returns (new_text, [source paths to copy])."""
    from urllib.parse import unquote
    wanted = []

    def sub(m):
        alt, target = m.group(1), m.group(2).strip()
        if target.startswith(("http://", "https://", "figures/")):
            return m.group(0)
        src = pathlib.Path(unquote(target.replace("\\", "/")))
        if not src.is_absolute():
            src = repo / src
        if not src.exists():
            # Left exactly as written. A link this cannot resolve is a fact
            # about the source document, and silently blanking it would hide
            # that a figure is missing rather than fix it.
            return m.group(0)
        wanted.append(src)
        return "![%s](figures/%s)" % (alt, src.name)

    return IMG.sub(sub, text), wanted


def tracked():
    """Every file v2 tracks in git, which is exactly what belongs in the repo.

    Reading from git rather than walking the tree means `output/` and anything
    else gitignored cannot leak into a publish by accident.
    """
    out = subprocess.run(["git", "ls-files"], cwd=str(V2),
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()

    if V2 in out.parents or out == V2:
        raise SystemExit("refusing to write inside t9v2/: %s" % out)

    files = [f for f in tracked()
             if not f.startswith("src/t9v2.egg-info/")
             and not f.startswith(EXCLUDE)]
    dropped = [f for f in tracked() if f.startswith(EXCLUDE)]
    plan = [(V2 / f, out / f) for f in files]

    missing = []
    for md in DESIGN:
        src = REPO / "docs" / md
        if not src.exists():
            missing.append(md)
            continue
        plan.append((src, out / "design" / md))
        twin = find_twin(md)
        if twin:
            plan.append((twin, out / "design" / twin.name))
        else:
            missing.append("%s (no docx/xlsx twin)" % md)

    print("%d files -> %s" % (len(plan), out))
    if dropped:
        print("  %d superseded documents NOT published:" % len(dropped))
        for f in dropped:
            print("    %s" % f)
    if a.dry_run:
        for s, d in plan:
            print("  %s" % d.relative_to(out))
    else:
        # Rebuilt from scratch, never merged into: a stale file left behind
        # would be published as though it were current. The CONTENTS go rather
        # than the directory itself, because on Windows anything holding a
        # handle on the folder — a shell parked in it, an editor, a virus
        # scanner — makes rmtree fail on the directory after it has already
        # emptied it, which leaves the tree half-deleted and the next run
        # unable to proceed. Clearing contents is idempotent and cannot get
        # stuck in that state.
        if out.exists():
            for child in out.iterdir():
                # NEVER touch .git. The destination is a real repository once it
                # has been published: it holds the remote, the branch and the
                # history of what was pushed. Deleting it silently unpublishes
                # the tree, and the damage is not obvious — the next `git add`
                # finds no repository HERE, walks UP to the parent, and stages
                # the parent's entire working tree instead. That happened on
                # 17 Aug 2026 and produced a 621-file commit in the working
                # repository, including a broken gitlink to v1's public repo.
                if child.name == ".git":
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        for s, d in plan:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

        # The design documents are POST-PROCESSED in the output tree rather
        # than transformed on the way in, so the copy above stays one plain
        # loop and the dry run still lists exactly what lands.
        figs, n_links = out / "design" / "figures", 0
        for md in DESIGN:
            dst = out / "design" / md
            if not dst.exists():
                continue
            text, wanted = rewrite_figures(
                dst.read_text(encoding="utf-8"), REPO)
            if not wanted:
                continue
            figs.mkdir(parents=True, exist_ok=True)
            for src_img in wanted:
                shutil.copy2(src_img, figs / src_img.name)
            dst.write_text(text, encoding="utf-8")
            n_links += len(wanted)
        if n_links:
            print("  %d figure links localised into design/figures/" % n_links)
        (out / "design" / "README.md").write_text(
            "# Design documents\n\n"
            "The documents T9Sim v2 was built FROM, copied at publish time by\n"
            "`tools/make_public.py` from the working repository, where they are\n"
            "still maintained. They are not edited here: this directory is\n"
            "rebuilt from scratch on every publish, so an edit made here is lost\n"
            "at the next one.\n\n"
            "| File | What it is |\n|---|---|\n"
            "| `T9Sim_Specification_v2.2.md` | the specification v2.2 is built from |\n"
            "| `T9Sim_Rebuild_Plan_v2.md` | the plan: 7 stages, the gates, what is deferred |\n"
            "| `T9Sim_Symbol_Plan.md` | the notation authority |\n"
            "| `T9Sim_DGP_Node_Register_v2.2.md` | the 79 nodes, one law and one type each |\n\n"
            "The `.docx` and `.xlsx` files are the same documents in the form\n"
            "they are read and reviewed in.\n", encoding="utf-8")

    if missing:
        print("\nNOT FOUND, so NOT published:")
        for m in missing:
            print("  %s" % m)
    print("\n%s. To publish:\n  cd %s && git init && git add -A && git commit"
          % ("dry run, nothing written" if a.dry_run else "built", out))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
