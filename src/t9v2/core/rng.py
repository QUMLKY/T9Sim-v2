# -*- coding: utf-8 -*-
"""Named random streams.

Every node names the stream it draws from, `edge_group` in graph.yaml, and a
stream is seeded from the run seed and the HASH OF ITS NAME rather than from a
counter. That is the whole point of the design: adding a variable to the funnel
shifts the funnel stream and leaves the user pool bit for bit identical, so a new
node cannot silently redraw the entire dataset. A positional scheme cannot
promise that, which is why check V9 fails on a numeric stream id.

The hash is blake2b and not Python's `hash()`, which is salted per process and
would make a run irreproducible across two invocations of the same script.

Three levels of key, each narrowing the one above:

    stream(seed, "user_pool")                  the pool streams, drawn once
    stream(seed, "funnel", block=7)            per row-block, so blocks are
                                               independent and a run can resume
    stream(seed, "market", block=7, sub="b_k") a named sub-draw inside a stream,
                                               so reordering two draws inside one
                                               block does not disturb the other

`sub` matters more than it looks. Without it, adding a draw to the market stream
would shift every later draw in that block, and the rival bids would change
because the floor gained a variate. With it, each law owns its own substream.
"""
from __future__ import annotations

import hashlib

import numpy as np

# bumped only if the derivation itself changes, which invalidates every dataset
# generated before the change. It is recorded in the run manifest for that reason.
SCHEME = "t9v2-rng-1"


def _key(name, block, sub):
    parts = [SCHEME, name]
    if block is not None:
        parts.append("block=%d" % block)
    if sub is not None:
        parts.append("sub=%s" % sub)
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def stream(seed, name, block=None, sub=None):
    """A Generator for one named stream, independent of every other name.

    Independence is the guarantee, not merely the intent: SeedSequence spawns
    from the pair (seed, name-hash), so two different names cannot correlate
    however close their text is.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("a stream needs a name; got %r" % (name,))
    return np.random.default_rng(np.random.SeedSequence([int(seed), _key(name, block, sub)]))


def block_bounds(n_rows, block_rows):
    """(start, stop, index) per block. The last block is short, never padded."""
    if block_rows <= 0:
        raise ValueError("block_rows must be positive; got %r" % (block_rows,))
    out, start, i = [], 0, 0
    while start < n_rows:
        stop = min(start + block_rows, n_rows)
        out.append((start, stop, i))
        start, i = stop, i + 1
    return out


def choice_p(rng, n, probs):
    """n draws from a categorical, returned as indices.

    Takes the pmf as a plain sequence and renormalises, because the calibration
    files store 6 decimal places and a row of 20 cities is short of 1 by about
    1e-5. numpy raises on that rather than absorbing it.
    """
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        raise ValueError("a categorical law was given a pmf summing to %r" % total)
    return rng.choice(len(p), size=n, p=p / total)


def resample_pmf(rng, n, values, probs):
    """n draws from an empirical pmf, returned as values rather than indices.

    The floor and paying price shapes are drawn this way: the iPinYou file is a
    (value, proportion) table and the law resamples it, so the zero atom carries
    its measured mass rather than a fitted one.
    """
    v = np.asarray(values, dtype=np.float64)
    return v[choice_p(rng, n, probs)]
