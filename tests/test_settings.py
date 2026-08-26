# -*- coding: utf-8 -*-
"""Stage 1 tests: the settings load and every startup check holds."""
import pytest
from t9v2.core import config


@pytest.fixture(scope="module")
def s():
    return config.load("default")


def test_loads(s):
    assert s.profile == "default"


def test_node_count(s):
    """79 since H5 `min_winning_price` landed, 22 August 2026."""
    assert len(s.nodes) == 79


def test_emitting_rows_sum_to_the_frozen_56(s):
    cols = [c for n in s.emitting() for c in n["columns"]]
    assert len(cols) == 56
    assert set(cols) == set(s.raw["column_order"])


def test_law_counts_match_the_register(s):
    """Read the expected split from the register's own type-count table rather
    than hardcoding it. A hardcoded expectation silently keeps testing an old
    design: this test passed on 18/9/17/34 after C4 was retyped to 17/9/17/35."""
    import re
    from collections import Counter
    from pathlib import Path

    # TWO LOCATIONS, and the second is not a convenience. In the working
    # repository the register lives one level ABOVE this package, in the parent
    # repo's docs/. In a published clone there is no parent repo: `parents[2]`
    # resolves outside the checkout entirely, and this test died with
    # FileNotFoundError on `<somewhere>/docs/T9Sim_DGP_Node_Register.md` when it
    # was first run from a clean copy. `tools/make_public.py` puts the register
    # in `design/`, so a clone looks there. A test that reaches outside its own
    # repository passes for the author and fails for everybody else.
    here = Path(__file__).resolve()
    # THE v2.2 FORK, and the v2 one is not a fallback. Reading v2 here would
    # compare the built graph against a register that has no H5, and the
    # mismatch would look like a graph defect rather than a stale path.
    candidates = [here.parents[1] / "design" / "T9Sim_DGP_Node_Register_v2.2.md",
                  here.parents[2] / "docs" / "T9Sim_DGP_Node_Register_v2.2.md"]
    found = [p for p in candidates if p.exists()]
    assert found, ("the node register is in neither location: %s"
                   % [str(p) for p in candidates])
    reg = found[0]
    want = {}
    for line in reg.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\|\s*(T[1-4])[^|]*\|\s*(\d+)\s*\|", line)
        if m:
            want[m.group(1)] = int(m.group(2))
    assert len(want) == 4, "could not read the 4 type counts from the register"
    assert dict(Counter(n["law"] for n in s.nodes)) == want


def test_every_parent_is_a_node(s):
    ids = set(s.by_id)
    for n in s.nodes:
        for p in n.get("parents") or []:
            assert p in ids, "%s names parent %s" % (n["id"], p)


def test_observability_covers_four_views(s):
    for n in s.nodes:
        obs = n["observability"]
        if obs != "latent":
            assert sorted(obs) == ["C1", "C2", "C3", "C4"]
            assert all(v in config.ROW_RULES for v in obs.values())


def test_no_v2_module_imports_v1():
    """The import test, section 2's single assertion."""
    import ast
    from pathlib import Path
    root = Path(config.__file__).resolve().parents[2]
    bad = []
    for f in list(root.rglob("*.py")):
        if "venv" in f.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "t9sim":
                        bad.append("%s:%d" % (f, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "t9sim":
                    bad.append("%s:%d" % (f, node.lineno))
    assert not bad, "v2 files importing v1: %s" % bad


def test_calibration_files_match_their_checksums(s):
    config.v_calibration_checksums(s)


def test_a_bad_parent_is_caught():
    """The checks must actually fail, not just pass on good input."""
    s = config.load("default", check=False)
    s.nodes[0]["parents"] = ["not_a_node"]
    s.by_id = {n["id"]: n for n in s.nodes}
    with pytest.raises(config.SettingsError, match="V1"):
        config.v1_acyclic_and_parents_exist(s)


def test_a_cycle_is_caught():
    s = config.load("default", check=False)
    a, b = s.nodes[0]["id"], s.nodes[1]["id"]
    s.by_id[a]["parents"] = [b]
    s.by_id[b]["parents"] = [a]
    with pytest.raises(config.SettingsError, match="cycle"):
        config.v1_acyclic_and_parents_exist(s)
