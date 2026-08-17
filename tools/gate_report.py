# -*- coding: utf-8 -*-
"""Write every gate artifact in one pass.

    python tools/gate_report.py [stage]        default stage 1

Replaces check_conformance.py and build_provenance.py, which each read the same
settings and each knew half the picture: the conformance report never carried the
undeclared keys, and neither wrote the gap log the plan requires.

Writes 3 documents:

  docs/gates/gate1.md         does the build match the design
  docs/provenance.md          every {value, route, source} in the settings
  docs/SPEC_GAPS.md           where the design documents were silent

It READS config and never writes it. Building config is `build_graph.py`'s job,
and a reporter that can change what it reports on is worth nothing.

SPEC_GAPS.md has 2 kinds of entry. The machine-detectable ones are regenerated
between the AUTO markers on every run. Everything outside those markers is
hand-written and is never touched, so a note like "the spec does not say what
shrinkage to use, so v1's pipeline.py:82 was read" survives regeneration.
"""
import io
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from t9v2.core import config                                      # noqa: E402
import report_holes                                               # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = HERE.parents[1]
DOCS = HERE.parent / "docs"
REGISTER = ROOT / "docs" / "T9Sim_DGP_Node_Register.md"

AUTO_START = "<!-- AUTO:START -->"
AUTO_END = "<!-- AUTO:END -->"

ROUTES = ["quoted", "auto-calibrated", "inferred", "specified"]

# leaves that are structure rather than calibration, so they need no provenance
STRUCTURAL = report_holes.STRUCTURAL


# ---------------------------------------------------------------- the register
def register_rows():
    """The active node rows, dead families excluded, keyed by their own headings.

    Read by heading and never by position. On 16 August 2026 the Plate and Stream
    columns were inserted after Role and every positional index below them moved,
    which this reporter did not notice: it went on reading index 5 for Type, found
    Stream there, and reported all 4 law counts as zero against a settings file
    that was correct. A heading lookup cannot fail that way.
    """
    rows, fam, hdr = [], None, None
    for line in REGISTER.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^### 3\.(\d+)\s+(.+?)(?:\s+—|$)", line)
        if m:
            fam, hdr = m.group(2).strip(), None
            continue
        if line.startswith("#"):
            fam = None
        if fam is None or not line.startswith("|") or re.match(r"^\|[\s\-|:]+\|$", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c[0] == "#":
            hdr = c
            continue
        if not c[0] or hdr is None or fam.lower().startswith("dead"):
            continue
        rows.append(dict(zip(hdr, c)))
    return rows


def register_law_counts(rows):
    """The register's own T1..T4 split, so the report COMPARES it rather than
    printing OK."""
    got = Counter()
    for r in rows:
        m = re.match(r"T[1-4]", r.get("Type", ""))
        if m:
            got[m.group(0)] += 1
    return got


def register_emitting(rows):
    """Rows whose Column cell is not an em dash. Counted, not hardcoded."""
    return sum(1 for r in rows
               if r.get("Column", "") and r["Column"][0] not in "—-")


# ------------------------------------------------------------------- provenance
def leaves(node, path=""):
    if isinstance(node, dict):
        if "route" in node and ("value" in node or "band" in node):
            yield (path, node.get("value", node.get("band")),
                   node.get("route", ""), str(node.get("source", "")))
            return
        for k, v in node.items():
            sub = "%s.%s" % (path, k) if path else str(k)
            if v == "HOLE":
                yield (sub, "HOLE", "", "")
            else:
                yield from leaves(v, sub)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, "%s[%d]" % (path, i))


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt(v):
    return ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)


# ---------------------------------------------------------------- the documents
def write_conformance(s, rows, found, stage):
    reg_laws = register_law_counts(rows)
    got_laws = Counter(n["law"] for n in s.nodes)
    cols = [c for n in s.emitting() for c in n["columns"]]

    counts = [
        ("nodes", len(s.nodes), len(rows)),
        ("emitting rows", len(s.emitting()), register_emitting(rows)),
        ("columns", len(cols), len(s.raw["column_order"])),
    ]
    for t in ("T1", "T2", "T3", "T4"):
        counts.append(("law %s" % t, got_laws.get(t, 0), reg_laws.get(t, 0)))

    L = ["# Gate %s conformance report" % stage, "",
         "*Generated by `tools/gate_report.py`. Regenerate after any settings or "
         "register change. This file is written, never edited.*", "",
         "## Counts", "",
         "Settings against the node register. Every row is compared; none is assumed.", "",
         "| What | Settings | Register | |", "|---|---:|---:|---|"]
    for name, got, want in counts:
        L.append("| %s | %d | %d | %s |"
                 % (name, got, want, "OK" if got == want else "**MISMATCH**"))

    L += ["", "## Startup checks", "",
          "Each check ran against the loaded settings. A failure raises and no "
          "report is written, so a report that exists is a report whose checks passed.", "",
          "| Check | What it holds | Result |", "|---|---|---|"]
    for name, _fn in config.CHECKS:
        L.append("| %s | %s | PASS |" % (name, CHECK_WHAT.get(name, "")))
    L += ["| V3 | every deterministic law names a function that exists | "
          "**deferred to stage 2**, it needs the generator |",
          "| V11 | no key is set twice at the same layer | PASS, during the merge |", ""]

    if s.lookup_only:
        L += ["## Auxiliaries reached by lookup", "",
              "These emit no column and no law names them as a parent. The "
              "specification's scope rule says a pool quantity enters auction scope "
              "through a lookup, so this is expected. Listed because a genuinely "
              "dead node looks identical.", "",
              ", ".join("`%s`" % x for x in s.lookup_only), ""]

    unconf = [(n["id"], n["_v5_unconfirmed"]) for n in s.nodes if n.get("_v5_unconfirmed")]
    if unconf:
        L += ["## V5, parents not confirmed in the law text", "",
              "The register writes laws in prose, so a parent can be real and still "
              "not appear by name. These re-check in stage 2 when the laws bind to "
              "functions.", "",
              "| Node | Parents not found in its law text |", "|---|---|"]
        L += ["| `%s` | %s |" % (i, ", ".join("`%s`" % p for p in ps)) for i, ps in unconf]
        L.append("")

    L += ["## Specification gaps", "",
          "Summary only. The full list, with the hand-written entries, is "
          "`docs/SPEC_GAPS.md`.", "",
          "| Kind | Count |", "|---|---:|",
          "| holes, no document supplies the value | %d |" % len(found["holes"]),
          "| unsourced, a value with no route or source | %d |" % len(found["unsourced"]),
          "| undeclared, a register key with no home in the config | %d |"
          % len(found["undeclared"]), ""]

    # A bold verdict line, so all seven gate files answer their question the
    # same way. Gate 1 passes when the counts reconcile and no value is missing
    # or unsourced; `undeclared` keys are reported and do not gate, being
    # register names with no home in the config rather than absent values.
    ok = (not [n for n, g, w in counts if g != w]
          and not found["holes"] and not found["unsourced"])
    L += ["## Gate 1 verdict", "",
          "**Gate 1: %s**" % ("PASS" if ok else "FAIL"), "",
          "%d undeclared register key%s reported, not gating."
          % (len(found["undeclared"]),
             "" if len(found["undeclared"]) == 1 else "s"), ""]

    p = DOCS / "gates" / "gate1.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    bad = [n for n, g, w in counts if g != w]
    return p, bad


CHECK_WHAT = {
    "V1": "every parent exists and the graph is acyclic",
    "V2": "one entry per node, no duplicates",
    "V4": "the emitting rows are exactly the frozen 55 columns",
    "V5": "every parent is read by its child's law",
    "V6": "every probability table's rows sum to 1",
    "V7": "every calibration path resolves",
    "V8": "every node's rng stream is declared",
    "V9": "every declared stream is named by some node",
    "V10": "every emitting node states its 4-view observability",
    "checksums": "the 6 calibration CSVs match their recorded hashes",
}


def write_provenance(s):
    rows, counts = [], Counter()
    for name in ["market.yaml", "tables.yaml", "training.yaml", "validation.yaml",
                 "profiles.yaml", "calibrated.yaml"]:
        doc = (config.CONFIG / name)
        if not doc.exists():
            continue
        import yaml
        for path, value, route, source in leaves(yaml.safe_load(
                doc.read_text(encoding="utf-8")) or {}):
            counts["HOLE" if value == "HOLE" else (route or "(none)")] += 1
            rows.append((name, path, value, route, source))

    L = ["# v2 provenance register", "",
         "*Generated by `tools/gate_report.py` from `config/*.yaml`. Do not hand-edit.*", "",
         "v2 stores provenance inside the settings themselves, as `{value, route, source}` "
         "on every calibratable leaf. This file is a readable view of that, nothing more. "
         "To change a source, edit the settings file and re-run.", "",
         "Routes: **quoted** an outside source supplies the number · **auto-calibrated** a "
         "solver output, whose real provenance is its target and solver · **inferred** a "
         "documented design judgement with no outside source · **specified** taken from a "
         "named T9 design document.", "",
         "| Route | Count |", "|---|---:|"]
    for r in ROUTES + ["(none)", "HOLE"]:
        if counts.get(r):
            L.append("| %s | %d |" % (r, counts[r]))
    L += ["| **total leaves** | **%d** |" % len(rows), ""]

    # The calibration CSVs are inputs with provenance too, and they are not yaml
    # leaves, so nothing else in this report would ever mention them.
    import json
    man = HERE.parent / "calibration" / "CHECKSUMS.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        L += ["## Calibration inputs, carried not computed", "",
              "The %d empirical shape files v2 inherits rather than rebuilds, since they cannot be "
              "recreated on this machine. Copied %s from `%s` and hashed; the loader re-checks every "
              "hash on every run and stops, naming the file, on a mismatch. Route is `quoted`: each "
              "is a measured distribution from the iPinYou log, not a design choice."
              % (len(m.get("files", {})), m.get("copied", "?"), m.get("source", "?")), "",
              "| File | Bytes | sha256 |", "|---|---:|---|"]
        for name, meta in m.get("files", {}).items():
            L.append("| `%s` | %d | `%s…` |" % (name, meta["bytes"], meta["sha256"][:16]))
        L.append("")

    L += ["## Every sourced value", "",
          "| File | Setting | Value | Route | Source |", "|---|---|---|---|---|"]
    for name, path, value, route, source in rows:
        if value == "HOLE":
            continue
        L.append("| `%s` | `%s` | %s | %s | %s |"
                 % (name, esc(path), esc(fmt(value)), route or "—", esc(source) or "—"))

    p = DOCS / "provenance.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p, counts


def write_spec_gaps(found, stage):
    """Regenerate only between the AUTO markers. Hand-written entries survive."""
    p = DOCS / "SPEC_GAPS.md"
    auto = ["%s" % AUTO_START,
            "",
            "*This block is regenerated by `tools/gate_report.py`. Do not edit inside the "
            "markers. Add hand-written entries below the block, where they are safe.*",
            "",
            "**Last written at gate %s.**" % stage,
            "",
            "### Holes, the values no document supplies", ""]
    if found["holes"]:
        auto += ["| File | Setting |", "|---|---|"]
        auto += ["| `%s` | `%s` |" % (f, k) for f, k in found["holes"]]
    else:
        auto.append("None.")
    auto += ["", "### Unsourced, a value carrying no route or source", ""]
    if found["unsourced"]:
        auto += ["| File | Setting |", "|---|---|"]
        auto += ["| `%s` | `%s` |" % (f, k) for f, k in found["unsourced"]]
    else:
        auto.append("None.")
    auto += ["", "### Undeclared, a key the node register names with no home in the config", "",
             "Each is either a genuine gap, meaning a node whose law has no parameters yet, "
             "or only a v1 name this config renamed. Both need a verdict, and the second "
             "kind should be reworded in the register so it stops being reported.", ""]
    if found["undeclared"]:
        auto += ["| Key |", "|---|"] + ["| `%s` |" % k for k in found["undeclared"]]
    else:
        auto.append("None.")
    auto += ["", AUTO_END]
    block = "\n".join(auto)

    if p.exists():
        text = p.read_text(encoding="utf-8")
        if AUTO_START in text and AUTO_END in text:
            head = text[:text.index(AUTO_START)]
            tail = text[text.index(AUTO_END) + len(AUTO_END):]
            p.write_text(head + block + tail, encoding="utf-8")
            return p, "updated"
        p.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        return p, "markers added"

    p.write_text(HEADER + block + FOOTER, encoding="utf-8")
    return p, "created"


HEADER = """# Specification gaps

**What this is.** Every place the design documents did not answer a question the build had to
answer. A gap is not a defect in the build, it is a silence in the specification, and a silence
never fails a check, which is why it needs its own log.

**Why it is not only a gate 7 document.** A gap found at stage 4 has to be visible at gate 4,
while amending the specification is still cheap. By gate 7 the data is generated and the silence
has already been resolved by whatever guess was made. So this file is rewritten at every gate.

**Two kinds of entry.** The block below is generated: holes, unsourced values and undeclared keys
are all machine-detectable. Everything after it is hand-written, because a silence that was
resolved by reading v1's code can only be recorded by whoever read it.

"""

FOOTER = """

---

## Hand-written entries

*Add below. Each entry: what the documents did not say, what was done instead, and what would
close it. These survive regeneration.*

### G1. The training half has no provenance register (15 Aug 2026)

**The silence.** The generator half has `t9_sim/calibration/provenance_table.md`, 50 rows of
parameter, value, route and source. The scoring half has no equivalent in any document.

**What was done.** v1's code and `docs/v1/T9Sim_Config_Reference.md` were read to recover 8
values: the historical-price switch state, the encoder shrinkage, the bid ladder, 3 XGBoost
parameter sets, and the attribution sample count and seed. All are routed `inferred` with a v1
citation. Evidence trail in `docs/T9Sim_Hole_Provenance.md`.

**What would close it.** A training-side provenance table in the specification, so v2's scoring
settings stop being recoverable only from v1.

### G2. The specification never states the archetype tilt strengths (15 Aug 2026)

**The silence.** Open item O1 wired archetype into os, device type and day of week. No document
says how strong those tilts should be, and no outside dataset can say, because archetype is T9's
own latent.

**What was done.** Invented and declared under O3 3b, as one dial per table on a shared ladder,
with tau = 0 reproducing a parentless draw exactly. v1's own `bn_cpts.yaml` was read afterwards
and disagreed on the day-of-week sign, which corrected the decision.

**What would close it.** Nothing external can. The values stay `inferred` and the datasheet
declares them.

### G3. Two live SHAP paths disagree on sample size (15 Aug 2026)

**The silence.** No document states the attribution sample count. v1's code has 2 answers,
`pipeline.py:774` at 3000 and `scripts/v10_shap.py:82` at 4000.

**What was done.** Took 4000, the path behind the reported v10 figures.

**What would close it.** The specification stating the count, and a note on which path produced
the published SHAP figures.
"""


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    s = config.load("default")
    rows = register_rows()
    found = report_holes.detect()

    p1, bad = write_conformance(s, rows, found, stage)
    p2, counts = write_provenance(s)
    p3, how = write_spec_gaps(found, stage)

    print("gate %s" % stage)
    print("  %s" % p1.relative_to(ROOT))
    print("      %d nodes, %d columns, mismatches: %s"
          % (len(s.nodes), sum(len(n["columns"]) for n in s.emitting()),
             ", ".join(bad) if bad else "none"))
    print("  %s" % p2.relative_to(ROOT))
    print("      %d leaves %s" % (sum(counts.values()), dict(counts)))
    print("  %s   (%s)" % (p3.relative_to(ROOT), how))
    print("      %d holes, %d unsourced, %d undeclared"
          % (len(found["holes"]), len(found["unsourced"]), len(found["undeclared"])))
    if bad:
        sys.exit("MISMATCH in: %s" % ", ".join(bad))


if __name__ == "__main__":
    main()
