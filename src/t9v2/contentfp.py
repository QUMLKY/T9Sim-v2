# -*- coding: utf-8 -*-
"""A content fingerprint: a hash of the DATA, not of the file that holds it.

    python -m t9v2.contentfp output/t9v2_1M_seed20250.parquet

WHY THIS EXISTS, AND WHY A SHA256 OF THE FILE IS NOT ENOUGH.

The checksum manifest hashes parquet bytes, which answers "is this file intact".
It cannot answer "did your machine produce the same data as mine", because a
parquet file embeds things that have nothing to do with the data: the writer's
version, compression choices, row-group layout, and padding. Two runs producing
identical numbers on different `pyarrow` builds give different file bytes and so
different sha256s, and a reviewer comparing them would conclude the data differed
when it did not.

This hashes the values instead. Same numbers in the same order gives the same
fingerprint whatever wrote the file, so it is the check another user can actually
run against a published value. v1 reached the same conclusion and shipped the
same idea in `t9sim/fingerprint.py`.

WHAT IT DOES NOT REUSE, AND WHY. v1 built its fingerprint on
`pandas.util.hash_pandas_object`. That works, but it pins the fingerprint to a
pandas internal: the function is not a documented stable format, so a pandas
upgrade may silently change every fingerprint ever published, and there would be
no way to tell that from the data having changed. This writes the byte encoding
out explicitly instead — fixed dtypes, fixed byte order, an explicit null
sentinel — so the fingerprint is a property of the numbers and of this file, and
nothing else can move it.

THE ENCODING, stated because a fingerprint whose rule is not written down cannot
be re-implemented by anyone checking it:

  columns are taken in SORTED name order, so column order in the file is
  irrelevant; each column contributes its name as utf-8, then a type tag, then
  its values;
  integers as little-endian int64; booleans as int64 0/1; floats as
  little-endian float64 raw IEEE-754 bits, so no rounding is applied and no
  precision is invented;
  strings and categoricals as utf-8 with a 0x1F separator, on the CATEGORY
  VALUES rather than the integer codes, since codes depend on discovery order;
  missing values as an explicit 0x00 marker in a parallel null mask, so a null
  and a zero never collide;
  rows in file order, which is deterministic because generation is.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHEME = "t9v2-contentfp-1"
CHUNK = 1 << 20          # rows per hash update, so a 10M column never lands whole in RAM
SEP = b"\x1f"
NULL = b"\x00"


class _Col:
    """The running state for one column: a hasher per STREAM, plus its type tag.

    THREE SEPARATE STREAMS, and this is the whole subtlety of the file. The
    obvious encoding writes a chunk's null mask followed by that chunk's values,
    then the next chunk's mask, and so on. That INTERLEAVES two things whose
    interleaving depends on where the chunk boundaries fell, and chunk boundaries
    follow the file's row groups, so a frame written with 500-row groups and the
    same frame written with 17-row groups hashed differently. Every column
    differed. Keeping the mask and the values in separate hashers makes each one
    a pure concatenation, which is what chunking is allowed to be.

    The type tag is held once, not written per chunk, for the same class of
    reason: emitting it per chunk makes the fingerprint count chunks.
    """

    __slots__ = ("mask", "vals", "tag", "n")

    def __init__(self, name):
        self.mask = hashlib.blake2b(digest_size=16, person=b"t9v2-cfp")
        self.vals = hashlib.blake2b(digest_size=16, person=b"t9v2-cfp")
        self.mask.update(name.encode("utf-8"))
        self.vals.update(name.encode("utf-8"))
        self.tag = b""
        self.n = 0

    def digest(self):
        h = hashlib.blake2b(digest_size=16, person=b"t9v2-cfp")
        h.update(self.tag)
        h.update(str(self.n).encode("ascii"))
        h.update(self.mask.digest())
        h.update(self.vals.digest())
        return h.digest()


def _feed(c, s):
    """One chunk of a column into its streams, in the encoding documented above."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)

    na = s.isna().to_numpy()
    c.n += len(na)
    # ONE BYTE PER ROW, not np.packbits, which pads to a byte boundary on every
    # call and so would also make the result depend on chunk size.
    c.mask.update(na.astype(np.uint8).tobytes())

    if pd.api.types.is_bool_dtype(s.dtype):
        c.tag = b"bool"
        v = s.fillna(False).to_numpy(dtype=bool).astype(np.int64)
        c.vals.update(np.ascontiguousarray(v, dtype="<i8").tobytes())
    elif pd.api.types.is_integer_dtype(s.dtype):
        c.tag = b"int"
        v = s.fillna(0).to_numpy(dtype="int64")
        c.vals.update(np.ascontiguousarray(v, dtype="<i8").tobytes())
    elif pd.api.types.is_float_dtype(s.dtype):
        c.tag = b"float"
        # raw IEEE-754 bits: exact, no rounding, no precision invented
        v = np.where(na, 0.0, s.to_numpy(dtype="float64"))
        c.vals.update(np.ascontiguousarray(v, dtype="<f8").tobytes())
    else:
        c.tag = b"str"
        v = s.to_numpy(dtype=object)
        # a TRAILING separator per value, not SEP.join: join puts no separator
        # between the last value of one chunk and the first of the next.
        c.vals.update(b"".join((NULL if n else str(x).encode("utf-8")) + SEP
                               for x, n in zip(v, na)))


def _new(name):
    return _Col(name)


def _combine(digests):
    """One fingerprint from the per-column digests, in sorted name order.

    Per-column digests rather than one running hash over everything, because it
    lets the file be read ONCE. A single sequential hash would have to take all
    of column A before any of column B, which means either holding the whole
    frame or re-reading the file per column: at 55 columns that is 55 passes over
    1.8 GB. Feeding row-group batches into 55 independent hashers and combining
    at the end is one pass and the same guarantee.
    """
    h = hashlib.blake2b(digest_size=8, person=b"t9v2-cfp")
    h.update(SCHEME.encode("utf-8"))
    for name in sorted(digests):
        h.update(name.encode("utf-8"))
        h.update(digests[name].digest())
    return h.hexdigest()


def fingerprint_frame(df):
    """The content fingerprint of one frame, as 16 hex characters."""
    hs = {}
    for c in df.columns:
        name = str(c)
        hs[name] = _new(name)
        col = df[c]
        for i in range(0, len(col), CHUNK):
            _feed(hs[name], col.iloc[i:i + CHUNK])
    return _combine(hs)


def fingerprint_file(path):
    """Read a parquet ONCE, in row groups, and fingerprint its contents.

    Batched so a 10M master never needs the ~5 GB a whole frame would take.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    hs = {str(n): _new(str(n)) for n in pf.schema_arrow.names}
    for batch in pf.iter_batches(batch_size=CHUNK):
        d = batch.to_pandas()
        for c in d.columns:
            name = str(c)
            _feed(hs[name], d[c])
    return _combine(hs)


def main(argv=None):
    args = list(argv or sys.argv[1:])
    if not args:
        print(__doc__.splitlines()[2].strip())
        return 1
    for a in args:
        print("%-34s %s" % (Path(a).name, fingerprint_file(a)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
