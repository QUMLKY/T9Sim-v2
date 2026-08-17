# -*- coding: utf-8 -*-
"""Settings loader and the startup checks.

Reads the 7 files in config/, merges them, and runs V1 to V11 before any data
is made. Every failure names the variable, the file and the key, because a
check that only says something is wrong costs more time than it saves.

V3 is not here. It verifies that every deterministic law names a function that
exists, and those functions arrive with the generator in stage 2.

A value the design documents did not supply is the string HOLE. The loader does
not treat that as an error, it collects them, because deciding them is gate 1's
job and not the loader's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"
CALIBRATION = ROOT / "calibration"

DOMAIN_FILES = ["graph.yaml", "tables.yaml", "market.yaml", "training.yaml",
                "validation.yaml"]
LAWS = {"T1", "T2", "T3", "T4"}
ROW_RULES = {"all", "won", "sold", "none"}
VIEWS = ["C1", "C2", "C3", "C4"]
HOLE = "HOLE"


class SettingsError(Exception):
    """Raised when a startup check fails. The message names file and key."""


def val(node):
    """The number inside a `{value, route, source}` leaf, or the node itself.

    Provenance is stored ON the value rather than beside it, which is what makes
    an unsourced constant impossible to write by accident. The cost is that every
    read has to step past the wrapper, and some leaves are bare because they are
    structure rather than calibration. This accepts both so a caller never has to
    know which it is holding.
    """
    if isinstance(node, dict) and "value" in node and "route" in node:
        return node["value"]
    return node


@dataclass
class Settings:
    raw: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)
    by_id: dict = field(default_factory=dict)
    profile: str = "default"
    holes: list = field(default_factory=list)
    lookup_only: list = field(default_factory=list)

    def node(self, node_id):
        return self.by_id[node_id]

    def emitting(self):
        return [n for n in self.nodes if n.get("columns")]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _read(name):
    p = CONFIG / name
    if not p.exists():
        raise SettingsError("missing settings file: config/%s" % name)
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise SettingsError("config/%s does not parse: %s" % (name, e))


def _find_holes(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _find_holes(v, ("%s.%s" % (path, k)).lstrip("."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_holes(v, "%s[%d]" % (path, i))
    elif obj == HOLE:
        yield path


def _merge(base, over, file_name, seen, path=""):
    """Key-by-key merge. V11 fails on a key set twice at the same layer."""
    for k, v in (over or {}).items():
        here = ("%s.%s" % (path, k)).lstrip(".")
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v, file_name, seen, here)
        else:
            if here in seen:
                raise SettingsError(
                    "V11 duplicate key. '%s' is set in both %s and %s"
                    % (here, seen[here], file_name))
            seen[here] = file_name
            base[k] = v
    return base


def _apply_solved(raw, solved):
    """Overlay the solver's answers, whose keys are full dotted settings paths.

    Not `_merge`. A solved entry is keyed `funnel.base_ctr.start.value`, and
    merging that dict would create a top-level key with a dot in its name rather
    than reaching the leaf it names, so the answers would be loaded and silently
    ignored. That is exactly what happened before 16 August 2026, and it is
    invisible from the outside: the file is written, the run succeeds, and the
    generator uses the unsolved starting values throughout.
    """
    for path, entry in solved.items():
        node, parts = raw, path.split(".")
        for k in parts[:-1]:
            if k not in node:
                raise SettingsError(
                    "calibrated.yaml solves '%s', but the settings have no '%s'"
                    % (path, ".".join(parts[:parts.index(k) + 1])))
            node = node[k]
        node[parts[-1]] = entry["value"] if isinstance(entry, dict) else entry


def load(profile="default", check=True, calibrated=True):
    """Merge the settings layers. `calibrated=False` omits the solver's output.

    The solver itself loads with calibrated=False, so it always starts from the
    hand-written value in market.yaml rather than from its own previous answer.
    Starting from the last answer would make a re-solve drift a little each time
    and, worse, would let one bad solve poison every solve after it: a wrong
    sigma raises e_ltv, which raises our bid, which makes the win-rate solve
    unable to bracket. That is not hypothetical; it is what happened on the first
    run of this solver on 16 August 2026.
    """
    raw, seen = {}, {}
    for f in DOMAIN_FILES:
        _merge(raw, _read(f), f, seen)

    profiles = _read("profiles.yaml")
    if profile not in profiles:
        raise SettingsError("profiles.yaml has no profile named '%s'. It has: %s"
                            % (profile, ", ".join(sorted(profiles))))
    _merge(raw, profiles[profile], "profiles.yaml:%s" % profile, {})

    if calibrated:
        _apply_solved(raw, _read("calibrated.yaml").get("solved") or {})

    s = Settings(raw=raw, nodes=raw.get("nodes") or [], profile=profile)
    s.by_id = {n["id"]: n for n in s.nodes}
    s.holes = sorted(_find_holes(raw))
    if check:
        run_checks(s)
    return s


# --------------------------------------------------------------------------
# the startup checks
# --------------------------------------------------------------------------

def _fail(check, msg):
    raise SettingsError("%s failed. %s" % (check, msg))


def v1_acyclic_and_parents_exist(s):
    for n in s.nodes:
        for p in n.get("parents") or []:
            if p not in s.by_id:
                _fail("V1", "node '%s' in graph.yaml names parent '%s', which is "
                            "not a node" % (n["id"], p))
    colour = {}

    def visit(nid, stack):
        state = colour.get(nid)
        if state == "done":
            return
        if state == "open":
            cycle = " -> ".join(stack[stack.index(nid):] + [nid])
            _fail("V1", "graph.yaml has a cycle: %s" % cycle)
        colour[nid] = "open"
        for p in s.by_id[nid].get("parents") or []:
            visit(p, stack + [nid])
        colour[nid] = "done"

    for n in s.nodes:
        visit(n["id"], [])


def v2_one_entry_each(s):
    seen = set()
    for n in s.nodes:
        if n["id"] in seen:
            _fail("V2", "graph.yaml declares node '%s' more than once" % n["id"])
        seen.add(n["id"])
    referenced = {p for n in s.nodes for p in (n.get("parents") or [])}
    # An auxiliary reaches its consumer through a lookup rather than a parent
    # edge, which is the specification's scope rule and what keeps the graph
    # acyclic. So an auxiliary with no parent edge is expected, not an orphan.
    # It is still recorded, because one that is genuinely dead looks the same.
    orphans, lookups = [], []
    for n in s.nodes:
        if n.get("columns") or n["id"] in referenced:
            continue
        (lookups if n.get("role") == "auxiliary" else orphans).append(n["id"])
    s.lookup_only = sorted(lookups)
    if orphans:
        _fail("V2", "these nodes emit no column and no law reads them, so they "
                    "are unused: %s" % ", ".join(sorted(orphans)))


def v4_frozen_columns(s):
    order = s.raw.get("column_order") or []
    emitted = [c for n in s.nodes for c in (n.get("columns") or [])]
    if not order:
        _fail("V4", "graph.yaml has no column_order, so there is nothing to "
                    "check the emitting rows against")
    extra, missing = set(emitted) - set(order), set(order) - set(emitted)
    if extra or missing:
        _fail("V4", "the emitting rows do not match column_order. extra=%s "
                    "missing=%s" % (sorted(extra) or "none", sorted(missing) or "none"))
    if len(emitted) != len(order):
        _fail("V4", "%d columns emitted against %d in column_order"
                    % (len(emitted), len(order)))


def v5_parents_are_read(s):
    """A declared parent that no law mentions is a lie in the declaration.
    Until the laws exist this can only check the register's own law text."""
    for n in s.nodes:
        parents = n.get("parents") or []
        text = "%s %s" % (n.get("law_plain", ""), n.get("law_note", ""))
        if not text.strip():
            continue
        for p in parents:
            if p not in text and p.lower() not in text.lower():
                # the register writes laws in prose, so this is a warning-grade
                # check until stage 2 binds real functions. Recorded, not fatal.
                n.setdefault("_v5_unconfirmed", []).append(p)


# The carried-over calibration files store proportions at 6 decimal places, so a
# row of 10 cities accumulates rounding of order 1e-5. A tolerance tighter than
# the source data's own precision would fail on arithmetic rather than on error.
# Rows are renormalised at load, so the drawn distribution sums exactly.
TABLE_TOL = 1e-4


def _renormalise(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total else dist


def v6_tables(s):
    tables = (s.raw.get("tables") or {})
    named = {n.get("table") for n in s.nodes if n.get("table")}
    for t in named:
        if t not in tables:
            _fail("V6", "a node names table '%s', which tables.yaml does not have" % t)
    for name, spec in tables.items():
        kind, values = spec.get("kind", ""), spec.get("values")
        if values == HOLE or values is None:
            continue
        if kind.startswith("conditional"):
            for row, dist in values.items():
                total = sum(dist.values())
                if abs(total - 1.0) > TABLE_TOL:
                    _fail("V6", "tables.yaml, table '%s' row '%s' sums to %.6f, "
                                "not 1 within %g" % (name, row, total, TABLE_TOL))
                values[row] = _renormalise(dist)
        elif kind.startswith("joint"):
            total = sum(v for row in values.values() for v in row.values())
            if abs(total - 1.0) > TABLE_TOL:
                _fail("V6", "tables.yaml, joint table '%s' sums to %.6f, not 1 "
                            "within %g" % (name, total, TABLE_TOL))


def v7_paths_resolve(s):
    for name, spec in (s.raw.get("tables") or {}).items():
        src = spec.get("source")
        if not src or not isinstance(src, str):
            continue
        rel = src.split(",")[0].strip()
        if rel.startswith("calibration/") and not (ROOT / rel).exists():
            _fail("V7", "tables.yaml, table '%s' points at '%s', which does "
                        "not exist" % (name, rel))


def v8_edge_group_declared(s):
    declared = set(s.raw.get("streams") or [])
    if not declared:
        _fail("V8", "graph.yaml declares no streams")
    for n in s.nodes:
        g = n.get("edge_group")
        if g not in declared:
            _fail("V8", "node '%s' draws from stream '%s', which is not in the "
                        "declared set %s" % (n["id"], g, sorted(declared)))


def v9_streams_named(s):
    for n in s.nodes:
        g = n.get("edge_group")
        if isinstance(g, int) or (isinstance(g, str) and g.isdigit()):
            _fail("V9", "node '%s' uses a positional stream '%s'. Streams are "
                        "named so adding a variable shifts nothing" % (n["id"], g))


def v10_observability(s):
    for n in s.nodes:
        obs = n.get("observability")
        if obs == "latent":
            continue
        if not isinstance(obs, dict):
            _fail("V10", "node '%s' has observability '%s', which is neither "
                         "'latent' nor a per-view map" % (n["id"], obs))
        if sorted(obs) != VIEWS:
            _fail("V10", "node '%s' covers views %s, not all 4"
                         % (n["id"], sorted(obs)))
        for view, rule in obs.items():
            if rule not in ROW_RULES:
                _fail("V10", "node '%s', view %s has row rule '%s'. Legal rules "
                             "are %s" % (n["id"], view, rule, sorted(ROW_RULES)))


def v_calibration_checksums(s):
    """Not numbered. The 6 carried-over files are hashed at load, and a
    mismatch stops the run naming the file."""
    man = CALIBRATION / "CHECKSUMS.json"
    if not man.exists():
        _fail("checksums", "calibration/CHECKSUMS.json is missing, so the "
                           "carried-over files cannot be verified")
    rec = json.loads(man.read_text(encoding="utf-8"))["files"]
    for name, want in rec.items():
        p = CALIBRATION / name
        if not p.exists():
            _fail("checksums", "calibration/%s is recorded but missing" % name)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want["sha256"]:
            _fail("checksums", "calibration/%s has changed since it was copied. "
                               "recorded %s, found %s"
                               % (name, want["sha256"][:16], got[:16]))


CHECKS = [
    ("V1", v1_acyclic_and_parents_exist),
    ("V2", v2_one_entry_each),
    ("V4", v4_frozen_columns),
    ("V5", v5_parents_are_read),
    ("V6", v6_tables),
    ("V7", v7_paths_resolve),
    ("V8", v8_edge_group_declared),
    ("V9", v9_streams_named),
    ("V10", v10_observability),
    ("checksums", v_calibration_checksums),
]


def run_checks(s):
    """V11 already ran during the merge, since it can only see the layers."""
    for name, fn in CHECKS:
        fn(s)
    return True


def report(profile="default"):
    s = load(profile)
    print("settings load OK, profile '%s'" % profile)
    print("  %d nodes, %d emitting, %d columns"
          % (len(s.nodes), len(s.emitting()),
             sum(len(n["columns"]) for n in s.emitting())))
    print("  checks passed: V1, V2, V4 to V11 and the calibration checksums")
    print("  V3 waits for the generator in stage 2")
    print("  %d holes, listed in the node register section 5.3" % len(s.holes))
    if s.lookup_only:
        print("  %d auxiliaries reach their consumer by lookup, not a parent edge: %s"
              % (len(s.lookup_only), ", ".join(s.lookup_only)))
    unconf = [n["id"] for n in s.nodes if n.get("_v5_unconfirmed")]
    if unconf:
        print("  V5 could not confirm parents in the law text for %d nodes: %s"
              % (len(unconf), ", ".join(unconf[:8]) + (" ..." if len(unconf) > 8 else "")))
    return s


if __name__ == "__main__":
    report()
