# -*- coding: utf-8 -*-
"""List every settings value the design documents did not supply.

    python tools/report_holes.py

A HOLE is a value v2 could not fill from the documents. v2 never reads v1's
settings, so a hole is recorded rather than quietly copied. This is the list
gate 1 rules on, and it belongs in the node register's section 5.3.

Three classes, not one. The original tool counted only the first, which meant a
value could hide from the audit simply by never having a key written for it. The
gate-1 provenance sweep found roughly 20 constants doing exactly that.

  1. HOLE       the key exists and its value is the string HOLE
  2. UNSOURCED  the key exists and carries a value, but no {route, source}
  3. UNDECLARED a settings key the node register NAMES that this config lacks

Class 3 is a review list, not a failure. The register cites v1's key names, so a
miss can mean a genuine gap or only a rename. Each one is a question, and a
question asked is the point.
"""
import io
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
REGISTER = ROOT.parent / "docs" / "T9Sim_DGP_Node_Register.md"

# Keys that are structure or derived arithmetic rather than calibration, so they
# need no provenance of their own. `implies` and `swing` restate what a tilt dial
# already produces, so their provenance is the dial's.
STRUCTURAL = re.compile(
    r"^(plates|streams|column_order|nodes|meta\.note|split_days(\..*)?"
    r"|contract(\..*)?"                                # calibrated: the solver's output NAMES, no values
    # calibrated.yaml's run record: when the solve ran, on what, and how big a
    # sample each constant was measured on. It describes the solve, so its
    # provenance is the solve; the solved VALUES carry theirs on the target block
    # they overlay in market.yaml.
    r"|meta\.(solved|scale|seed|checksum|rows_per_measurement)(\..*)?"
    r"|\w+\.(seeds|scales|block_rows)(\..*)?"          # profiles: run config, not calibration
    r"|calibration\.(measures|heads)(\..*)?"           # which metrics on which heads, a scope not a value
    r"|\w+\.order(\..*)?"                              # the position order of a vector, a label not a value
    r"|meta\.tilt_sensitivity_sweep|meta\.archetype_ladder\..*"
    r"|repro_tolerance\..*"
    r"|tables\.[\w]+\.(kind|parents|raked|build_inputs|values|base_hour_marginal)(\..*)?"
    r"|tables\.[\w]+\.tilt\.(level|ladder|ladder_sign|implies|swing)(\..*)?"
    r"|tables\.[\w]+\.archetype_tilts\.mu_kappa(\..*)?"
    r"|.*\.(note|target|checks|gate_policy|circularity_warning|reference|pinned|spacing"
    r"|ceiling|id|assert)(\..*)?)$")

# Register citations that are v1 file paths, code, templates or prose, not keys.
NOT_A_KEY = re.compile(
    r"\.(py|yaml|csv|md|json)\b|^[A-Z]{1,3}\d|^\d|py:|[()<>{}]|\*"
    r"|^(design|none|as above|inferred|quoted|specified|PLACEHOLDER)", re.I)

# A settings key cited by its bare name, with no dot. Until 16 August 2026 these
# were skipped outright, which made a whole class of gap unreportable: n_apps,
# campaign_share, ad_exchanges, slot_format_shares and slot_quality were all named
# by the register, all absent from this config, and all invisible here, so the
# audit printed 0 undeclared while the generator had no pool sizes to run with.
# Bare tokens are now considered, filtered against the things that look like keys
# but are not: node ids, emitted column names, and single words of prose.
BARE_KEY = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")     # snake_case, 2+ parts


def walk(obj, path=""):
    """Yield (path, node) for every mapping and scalar in the tree."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, ("%s.%s" % (path, k)).lstrip("."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, "%s[%d]" % (path, i))
    else:
        yield path, obj


def scan(doc):
    """Return (holes, unsourced) for one settings document."""
    holes, unsourced, sourced_under = [], [], []
    for path, node in walk(doc):
        if node == "HOLE":
            holes.append(path)
        elif isinstance(node, dict) and "route" in node:
            sourced_under.append(path)
    for path, node in walk(doc):
        if isinstance(node, dict) or isinstance(node, list) or node is None:
            continue
        # a list index is not a separate setting, so judge the list as a whole
        if path in holes or STRUCTURAL.match(re.sub(r"\[\d+\]", "", path)):
            continue
        # a leaf is sourced if it IS a {value, route, source} field, or sits under one
        if path.rsplit(".", 1)[-1] in ("value", "route", "source"):
            continue
        if any(path == s or path.startswith(s + ".") for s in sourced_under):
            continue
        unsourced.append(path)
    return holes, unsourced


def register_keys(nodes=()):
    """Settings keys the node register names in backticks, deduped.

    `nodes` is every node id and every emitted column name. Both are written in
    backticks throughout the register and neither is a settings key, so they are
    excluded rather than reported as gaps.
    """
    if not REGISTER.exists():
        return set()
    skip = {n.lower() for n in nodes}
    out = set()
    for tok in re.findall(r"`([^`]+)`", REGISTER.read_text(encoding="utf-8")):
        tok = tok.strip()
        if " " in tok or NOT_A_KEY.search(tok):
            continue
        base = tok.split("[")[0]
        if "." not in base and not (BARE_KEY.match(base) and base.lower() not in skip):
            continue
        out.add(base)
    return out


def config_paths(docs):
    seen = set()
    for doc in docs.values():
        for path, _ in walk(doc):
            if path:
                seen.add(path)
                # also register the dotted tail, so `device.os_split` matches
                parts = path.split(".")
                for i in range(len(parts)):
                    seen.add(".".join(parts[i:]))
    return seen


def detect():
    """The three gap classes, as data. `gate_report.py` calls this so the gate
    artifacts and this console tool can never report different findings.

    Returns {holes, unsourced, undeclared}, each a list. holes and unsourced are
    (file, path) pairs; undeclared is a list of register key names.
    """
    docs, holes, unsourced = {}, [], []
    for f in sorted(CONFIG.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        docs[f.name] = doc
        if f.name == "graph.yaml":
            # generated from the node register, pure structure, no calibration leaves
            continue
        h, u = scan(doc)
        holes += [(f.name, p) for p in h]
        unsourced += [(f.name, p) for p in u]
    known = config_paths(docs)
    g = docs.get("graph.yaml") or {}
    nodes = [n["id"] for n in g.get("nodes", [])]
    nodes += [c for n in g.get("nodes", []) for c in (n.get("columns") or [])]
    return {"holes": holes, "unsourced": unsourced,
            "undeclared": sorted(k for k in register_keys(nodes) if k not in known)}


def main():
    found = detect()
    all_holes, all_unsourced = found["holes"], found["unsourced"]
    undeclared = found["undeclared"]
    by_file = {}
    for kind, rows in (("HOLE      ", all_holes), ("unsourced ", all_unsourced)):
        for f, p in rows:
            by_file.setdefault(f, []).append((kind, p))
    for f in sorted(CONFIG.glob("*.yaml")):
        if f.name == "graph.yaml":
            print("%-18s  generated from the register, not scanned" % f.name)
            continue
        rows = by_file.get(f.name, [])
        nh = sum(1 for k, _ in rows if k.startswith("HOLE"))
        print("%-18s %2d holes  %2d unsourced" % (f.name, nh, len(rows) - nh))
        for kind, p in rows:
            print("     %s %s" % (kind, p))

    print("\n%d holes. Each needs a verdict at gate 1." % len(all_holes))
    if all_unsourced:
        print("%d values carry no {route, source}. Add one or they are invisible to the audit."
              % len(all_unsourced))
    if undeclared:
        print("\n%d settings keys the node register names that this config has no home for."
              % len(undeclared))
        print("Review each: a genuine gap, or only a v1 name this config renamed.")
        for k in undeclared:
            print("     %s" % k)
    return len(all_holes)


if __name__ == "__main__":
    # only when run directly: wrapping at import time would re-wrap a stream a
    # caller had already wrapped, and closing the first wrapper closes the buffer
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
