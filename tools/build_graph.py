# -*- coding: utf-8 -*-
"""Generate config/graph.yaml from the node register.

    python tools/build_graph.py

The register (docs/T9Sim_DGP_Node_Register_v2.2.md) is the design input and stays
the
source. This turns its 79 rows into the settings file the loader reads, so the
two cannot drift by hand-editing. Re-run after amending the register.

Everything this script decides is a mapping rule stated below, not a judgement:
plate comes from the register's family, law from its Type column, kind from
whether the row emits a column, and observability from the censoring map in
specification 1.5. Nothing is invented per row.
"""
import io, json, re, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
# REPOINTED AT THE v2.2 FORKS, 22 August 2026. The v2 documents have no H5,
# so a builder left pointing at them would write a graph without
# `min_winning_price` and nothing in the output would say so.
REG = ROOT / "docs" / "T9Sim_DGP_Node_Register_v2.2.md"
SPEC = ROOT / "docs" / "T9Sim_Specification_v2.2.md"
OUT = Path(__file__).resolve().parents[1] / "config" / "graph.yaml"

# --- plate and stream are READ from the register, never inferred -------------
# Until 16 August 2026 this file derived both from the family heading through 2
# lookup tables. That inference gave B1 an app plate while its parent app_i sat
# at auction plate, a backwards edge no check could see because nothing read
# plate. Both are now stated per node in the register and this file only
# validates the vocabulary, so a wrong value is a register edit, not a silent
# guess. Specification 2.2's factorisation products are the independent check.
PLATES = {"user", "app", "campaign", "advertiser", "rival", "auction"}
STREAMS = {"user_pool", "app_pool", "campaign_pool", "rival_pool",
           "join", "context", "truth", "market", "funnel"}

# --- observability: specification 1.5, the C1-C4 censoring map ---------------
# Families A, B, C, D plus H1, H2, H3 are visible on every row in every view.
# E, F, G are won-rows-only in C1 and C3, all rows in C2 and C4.
# H4, H5 and H9 are SSP-only, so absent in C1 and C2 and present in C3 and C4.
ALL_VIEWS = {"C1": "all", "C2": "all", "C3": "all", "C4": "all"}
FUNNEL_VIEWS = {"C1": "won", "C2": "all", "C3": "won", "C4": "all"}
SSP_VIEWS = {"C1": "none", "C2": "none", "C3": "all", "C4": "all"}
#
# THE ONE EDIT IN v2.2 THAT FAILS SILENTLY, and H5 is why it is called out.
# `observability()` tests, in order: no columns, HIDDEN_NODES, a role starting
# "latent" or "estimand", this set, a role starting "outcome", then everything
# else falls through to ALL_VIEWS. H5's register role reads "observable
# (SSP-visible, C3 and C4 only)" and matches NONE of the earlier tests, because
# the role text is never parsed for the words "SSP-visible".
#
# So omitting H5 here does not raise. The tool succeeds, writes a well-formed
# graph.yaml, and hands `min_winning_price` to all four views while
# `winning_price` stays correctly SSP-only. C1 and C2 then hold the exact
# hurdle, `won = 1[bid >= min_winning_price]` becomes an identity they can
# read, and the SSP contrast collapses to zero with no number anywhere looking
# wrong. tests/test_graph.py asserts the membership rather than trusting this.
SSP_ONLY_NODES = {"H4", "H5", "H9"}
HIDDEN_NODES = {"user_id"}                        # master lineage only

# H3 `won` carries role `outcome` in the register, and the rule below would put
# any outcome on the funnel's won-rows-only schedule. Specification 1.5 lists it
# under "Always visible: A, B, C, D, H1, H2, H3" instead, and the reason is both
# real and structural. Real: every DSP is told whether its own bid won, whatever
# data layers it buys; that is the bid response, not attribution. Structural:
# Tier 2 learns P(win | bid) FROM this column, so hiding it on lost rows would
# leave C1 and C3 with no losses to learn from, and those are the 2 views the
# SSP result is measured across. Found on 16 August 2026, at the start of stage 3.
ALWAYS_VISIBLE_OUTCOMES = {"H3"}


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def read_register():
    fam, hdr, rows, out = None, None, [], []
    for line in REG.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^### 3\.(\d+)\s+(.+?)(?:\s+—|$)", line)
        if m:
            if fam and rows:
                out.append((fam, rows))
            fam, hdr, rows = m.group(2).strip(), None, []
            continue
        if fam is None or not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-\|:]+\|$", line):
            continue
        c = cells(line)
        if hdr is None:
            hdr = c
        else:
            rows.append(c)
    if fam and rows:
        out.append((fam, rows))
    return [(f, r) for f, r in out if not f.lower().startswith("dead")]


def parse_parents(text):
    """The register writes parents as prose. Keep only bare node names.

    Some rows append clauses that are not parents at all, as H2 does with
    'dials shade, sigma_explore; solved k_global'. A dial and a solved constant
    are settings, not variables, so anything after such a marker is dropped.
    Startup check V1 catches whatever slips through, since it fails on a parent
    that is not a node.
    """
    # a parenthetical usually qualifies, but 'B2 (through k_cpa, ..., eltv_b2)'
    # names real parents inside it, so keep node-shaped tokens from that form
    inner = " ".join(re.findall(r"\(through ([^)]*)\)", text))
    t = re.split(r";\s*(?:dials|solved)\b", text)[0]    # drop dial/solved clauses
    t = re.sub(r"\(.*?\)", "", t) + (", " + inner if inner else "")
    if t.strip().lower().startswith("none"):
        return []
    parts = [p.strip(" `") for p in re.split(r"[,;]", t)]
    keep = []
    for p in parts:
        p = p.split(" via ")[0].strip()
        if not p or p.lower() in ("none", "and", ""):
            continue
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_\[\]\(\)]*", p)
        if m:
            keep.append(m.group(0))
    return keep


def parse_columns(text):
    """'—' means the row emits nothing. Expand the 2 abbreviated multi-column rows."""
    t = text.strip()
    # the register writes '—' for no column, sometimes with a note after it,
    # as in '— (dropped at join)'. Anything starting with a dash emits nothing.
    if not t or t[0] in "—-":
        return []
    if "lu6_casual" in t:
        return ["lu6_casual", "lu6_strategy", "lu6_rpg", "lu6_hypercasual"]
    if "la2_whale" in t:
        return ["la2_whale", "la2_engaged_spender", "la2_casual",
                "la2_time_filler", "la2_inactive"]
    return [c.strip(" `") for c in t.split(",") if c.strip()]


def observability(node, role, columns):
    if not columns:
        return "latent"
    if node in HIDDEN_NODES:
        return "latent"
    if role.startswith(("latent", "estimand")):
        return "latent"
    if node in SSP_ONLY_NODES:
        return dict(SSP_VIEWS)
    if role.startswith("outcome") and node not in ALWAYS_VISIBLE_OUTCOMES:
        return dict(FUNNEL_VIEWS)
    return dict(ALL_VIEWS)


def yaml_scalar(v):
    if isinstance(v, str) and (v == "" or re.search(r"[:#\{\}\[\],&*?|<>=!%@`]", v)):
        return '"%s"' % v.replace('"', '\\"')
    return v


def frozen_columns():
    """The 56 in deposit order, from specification 1.7. Startup check V4 compares
    the emitting rows against this, and the writer emits in this order so a
    re-run is byte-identical."""
    text = SPEC.read_text(encoding="utf-8")
    # 56 since H5, 22 August 2026. The count is PARSED out of a literal
    # Specification sentence rather than counted, so a disagreement raises here
    # rather than passing quietly, which is the behaviour wanted.
    raw = re.search(r"56 columns\):\s*(.+?)\.\n", text, re.S).group(1)
    raw = raw.replace("\n", " ")
    raw = re.sub(r"(\w+)_\{([^}]+)\}",
                 lambda m: ", ".join("%s_%s" % (m.group(1), x.strip())
                                     for x in m.group(2).split(",")), raw)
    return [c.strip() for c in raw.split(",") if c.strip()]


def main():
    fams = read_register()
    entries, seen = [], set()
    for fam, rows in fams:
        for r in rows:
            num, node, colspec, role = r[0], r[1], r[2], r[3]
            plate, stream, parspec, typ = r[4].strip(), r[5].strip(), r[6], r[7]
            node = node.split()[0].strip(" `")
            if node in seen:
                sys.exit("duplicate node id %s" % node)
            seen.add(node)
            if plate not in PLATES:
                sys.exit("%s: plate %r is not one of %s" % (node, plate, sorted(PLATES)))
            if stream not in STREAMS:
                sys.exit("%s: stream %r is not one of %s" % (node, stream, sorted(STREAMS)))
            cols = parse_columns(colspec)
            entries.append({
                "id": node, "n": num, "family": fam, "plate": plate,
                "role": role.split(",")[0].split("(")[0].strip(),
                "parents": parse_parents(parspec),
                "edge_group": stream,
                "law": typ.split()[0],
                "law_note": typ,
                "kind": "column" if cols else "internal",
                "columns": cols,
                "observability": observability(node, role, cols),
                "law_plain": r[8] if len(r) > 8 else "",
            })

    # a parent token that is not a node is a settings key, as k_cpa is. Drop it
    # and print it, so nothing goes missing silently. Startup check V1 stays the
    # runtime guard against a genuine typo in the register.
    ids = {e["id"] for e in entries}
    dropped = {}
    for e in entries:
        keep = [p for p in e["parents"] if p in ids]
        for p in e["parents"]:
            if p not in ids:
                dropped.setdefault(p, []).append(e["id"])
        e["parents"] = keep

    lines = [
        "# graph.yaml, the 79 generator variables.",
        "# GENERATED by tools/build_graph.py from docs/T9Sim_DGP_Node_Register_v2.2.md.",
        "# Edit the register, then re-run the builder. Do not hand-edit this file.",
        "",
        "plates: [user, app, campaign, advertiser, rival, auction]",
        "",
        "streams: [%s]" % ", ".join(sorted(STREAMS)),
        "",
        "# The frozen 56, in deposit order, read from specification 1.7.",
        "column_order:",
    ] + ["  - %s" % c for c in frozen_columns()] + [
        "",
        "nodes:",
    ]
    for e in entries:
        lines.append("  - id: %s" % e["id"])
        lines.append("    plate: %s" % e["plate"])
        lines.append("    law: %s" % e["law"])
        if e["law_note"] != e["law"]:
            lines.append("    law_note: %s" % yaml_scalar(e["law_note"]))
        lines.append("    edge_group: %s" % e["edge_group"])
        lines.append("    parents: [%s]" % ", ".join(e["parents"]))
        lines.append("    family: %s" % e["family"])
        lines.append("    role: %s" % e["role"])
        lines.append("    kind: %s" % e["kind"])
        lines.append("    columns: [%s]" % ", ".join(e["columns"]))
        if isinstance(e["observability"], str):
            lines.append("    observability: %s" % e["observability"])
        else:
            lines.append("    observability:")
            for k in ("C1", "C2", "C3", "C4"):
                lines.append("      %s: %s" % (k, e["observability"][k]))
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    ncol = sum(len(e["columns"]) for e in entries)
    print("wrote %s" % OUT)
    print("  %d nodes, %d emitting, %d columns" %
          (len(entries), sum(1 for e in entries if e["columns"]), ncol))
    from collections import Counter
    if dropped:
        print("  dropped parent tokens that are not nodes, settings keys not variables:")
        for p, ns in sorted(dropped.items()):
            print("    %-12s named by %s" % (p, ", ".join(ns)))
    print("  laws  :", dict(Counter(e["law"] for e in entries)))
    print("  plates:", dict(Counter(e["plate"] for e in entries)))


if __name__ == "__main__":
    main()
