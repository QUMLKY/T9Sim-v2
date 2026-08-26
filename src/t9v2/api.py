# -*- coding: utf-8 -*-
"""The 6 public functions, and the one-row `predict` the plan gates on.

    generate(profile, scale, seed)   writes the master parquet
    censor(master, view)             one of the 4 views
    train(view)                      fits the model stack on that view
    evaluate(profile, scale, seed)   one seed end to end
    predict(view, row) -> ev, pwin   scores a single row
    aggregate(profile, scale)        the results table from the per-seed files

`predict` exists to be checked against the batch path, not because anything in
the campaign calls it one row at a time. Gate 4 requires that scoring one row
returns what the vectorised batch returns for that same row, which is the only
way to know a bidder lifted out of this code would behave the way the reported
numbers say it does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import censor as _censor
from . import generate as _generate
from .core.config import load
from .train import bidders as B
from .train import encoder as E
from .train import features as F
from .train.runner import run_seed, train_view

OUTPUT = Path(__file__).resolve().parents[2] / "output"


def generate(profile="default", scale="100K", seed=20250, **kw):
    return _generate.generate(scale=scale, seed=seed, profile=profile, **kw)


def censor(master, view, settings=None):
    return _censor.censor(master, view, settings)


class Trained:
    """A fitted stack for one view, with a single-row scorer.

    Holds the encoders as well as the models. A bidder that shipped without them
    would score a row against a global mean and quietly lose the entire SSP
    channel, since that channel IS the encoder.
    """

    def __init__(self, view, fitted, settings):
        self.view, self.settings = view, settings
        self.t1, self.t2 = fitted["tier1"], fitted["tier2"]
        self.enc = fitted["encoders"]
        self.t1_cols, self.t2_cols = fitted["t1_cols"], fitted["t2_cols"]

    def _prep(self, rows):
        d = F.prepare(rows.copy(), self.settings)
        E.apply(self.enc, d)
        return d

    def score(self, rows):
        """ev and p(win) for a frame of view rows. The batch path."""
        d = self._prep(rows)
        return self.t1.predict(d)["ev"], self.t2.predict(d)

    def predict(self, row):
        """One row. Must agree with `score` on the same row, and gate 4 checks it."""
        import pandas as pd
        d = row.to_frame().T if isinstance(row, pd.Series) else row.iloc[[0]]
        ev, pw = self.score(d)
        return float(ev[0]), float(pw[0])

    def bid(self, rows):
        """The profit-maximising bid over the shared ladder, and whether to bid.

        Returns `(bid, placed)`. The second is the ROAS gate's answer and is not
        folded into the first: `bid` stays the ungated argmax on every row, so a
        caller sees both what the bidder would pay and whether it was willing to
        pay it. A declined row cannot be signalled by a zero bid, because 14
        percent of rows have a floor of exactly zero and a zero bid clears the
        hurdle on them.
        """
        prices = B.ladder(self.settings)
        d = self._prep(rows)
        ev = self.t1.predict(d)["ev"]
        b, _, _, placed = B.choose(B.to_price_unit(ev),
                                   self.t2.win_curve(d, prices), prices,
                                   B.roas_target(self.settings))
        return b, placed


def train(master, view, settings=None, seed=0):
    s = settings or load("default")
    _, fitted = train_view(master, view, s, seed)
    return Trained(view, fitted, s)


def evaluate(profile="default", scale="100K", seed=20250, master=None,
             settings=None, quiet=False):
    """One seed end to end: generate if needed, then all 4 views."""
    import pandas as pd
    s = settings or load(profile)
    if master is None:
        path = OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))
        if not path.exists():
            _generate.generate(scale=scale, seed=seed, profile=profile, quiet=True)
        master = pd.read_parquet(path)
    return run_seed(master, s, seed=0, quiet=quiet)


def aggregate(results):
    """The cross-view table from a list of per-seed result dicts."""
    views = list(results[0].keys())
    out = {}
    for v in views:
        rows = [r[v] for r in results]
        out[v] = {
            "seeds": len(rows),
            "click_auc": float(np.mean([r["heads"]["click"]["auc"] for r in rows])),
            "install_auc": float(np.mean([r["heads"]["install"]["auc"] for r in rows])),
            "win_auc": float(np.mean([r["heads"]["win"]["auc"] for r in rows])),
            "spend_crps": float(np.mean([r["heads"]["spend"]["crps"] for r in rows])),
            "profit": float(np.mean([r["economics"]["learned"]["profit"] for r in rows])),
            "value_captured": float(np.mean([r["economics"]["learned"]["value_captured"] for r in rows])),
        }
    return out


def save(results, path):
    Path(path).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# the 3 console commands declared in pyproject.toml
# ---------------------------------------------------------------------------

def main_generate(argv=None):
    """t9v2-generate: write one master parquet."""
    from . import generate as _g
    return _g.main(argv)


def main_run(argv=None):
    """t9v2-run: one seed end to end, all 4 views, written to its run folder."""
    from . import campaign as _c
    import argparse
    ap = argparse.ArgumentParser(description=main_run.__doc__)
    ap.add_argument("--scale", default="100K", choices=_c.SCALES)
    ap.add_argument("--seed", type=int, default=20250)
    a = ap.parse_args(argv)
    r = _c.run_one(a.scale, a.seed)
    v = r["views"]
    print("%s seed %d, %.0fs" % (a.scale, a.seed, r["total_seconds"]))
    print("  %-4s %9s %9s %9s %11s" % ("view", "clickAUC", "winAUC", "value_captured", "profit"))
    for k in _c.VIEWS:
        h, e = v[k]["heads"], v[k]["economics"]["learned"]
        print("  %-4s %9.4f %9.4f %9.4f %11.0f"
              % (k, h["click"]["auc"], h["win"]["auc"], e["value_captured"], e["profit"]))
    return 0


def main_campaign(argv=None):
    """t9v2-campaign: every seed at one scale, each in a fresh process."""
    from . import campaign as _c
    return _c.main(argv)
