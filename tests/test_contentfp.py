# -*- coding: utf-8 -*-
"""The content fingerprint must survive the writer and catch the data.

These are the two properties the whole point rests on. If the first fails the
fingerprint is no better than a sha256 and another machine cannot use it. If the
second fails it is worse than useless, because it would agree while the data
differed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from t9v2.contentfp import fingerprint_file, fingerprint_frame


def frame(n=500, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "i": r.integers(0, 100, n).astype("int64"),
        "f": r.normal(size=n),
        "s": pd.Categorical(r.choice(["a", "bb", "ccc"], n)),
        "b": r.random(n) > 0.5,
        "nullable": pd.array(np.where(r.random(n) > 0.3, r.integers(0, 9, n),
                                      None), dtype="Int64"),
    })


def test_frame_and_file_agree(tmp_path):
    d = frame()
    p = tmp_path / "a.parquet"
    d.to_parquet(p)
    assert fingerprint_frame(d) == fingerprint_file(p)


def test_survives_a_different_writer(tmp_path):
    """The property that makes it portable, and the reason it exists."""
    d = frame()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    d.to_parquet(a, compression="snappy", row_group_size=500)
    d.to_parquet(b, compression="gzip", row_group_size=17)

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert sha(a) != sha(b), "the writers must actually differ for this to test anything"
    assert fingerprint_file(a) == fingerprint_file(b)


def test_chunk_size_does_not_matter():
    """The property the writer-independence rests on, tested directly.

    Two bugs hid here and both made the fingerprint count CHUNKS rather than
    rows: emitting the type tag per chunk, and interleaving each chunk's null
    mask with its values so the byte order followed the chunk boundaries. Both
    only show when the boundaries differ, which is why this is asserted on the
    feed rather than left to the file-level test to notice.
    """
    from t9v2.contentfp import _combine, _feed, _new

    d = frame(300, seed=7)

    def run(size):
        hs = {}
        for c in d.columns:
            hs[str(c)] = _new(str(c))
            for i in range(0, len(d), size):
                _feed(hs[str(c)], d[c].iloc[i:i + size])
        return _combine(hs)

    assert run(300) == run(17) == run(1) == fingerprint_frame(d)


def test_column_order_does_not_matter():
    d = frame()
    assert fingerprint_frame(d) == fingerprint_frame(d[list(reversed(d.columns))])


def test_row_order_does_matter():
    """Generation order is deterministic, so a reordering IS a difference."""
    d = frame()
    assert fingerprint_frame(d) != fingerprint_frame(d.iloc[::-1])


@pytest.mark.parametrize("col,delta", [("f", 1e-12), ("i", 1)])
def test_one_changed_value_moves_it(col, delta):
    d = frame()
    e = d.copy()
    e.loc[e.index[3], col] = d[col].iloc[3] + delta
    assert fingerprint_frame(d) != fingerprint_frame(e)


def test_null_and_zero_do_not_collide():
    """A null must not hash as the value it is filled with internally."""
    a = pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")})
    b = pd.DataFrame({"x": pd.array([1, 0, 3], dtype="Int64")})
    assert fingerprint_frame(a) != fingerprint_frame(b)


def test_stable_across_calls():
    d = frame()
    assert fingerprint_frame(d) == fingerprint_frame(d)
