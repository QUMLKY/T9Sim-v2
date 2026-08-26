# -*- coding: utf-8 -*-
"""The frozen predictor: everything needed to score a row, saved to disk.

WHY THIS EXISTS. v2 trained models and threw them away. A model lived inside the
process that fitted it, so every new metric cost a full retrain, and a retrain
does not reproduce the model: `n_jobs=0` uses every core and `hist` is
deterministic for a fixed thread count, not across machines. A published model
can only be kept by being saved. Adding one metric to a 10M seed cost four hours
and produced a DIFFERENT model to the one whose numbers were published.

WHAT IS SAVED, per view:

    <root>/manifest.json          schema, sha256 per file, feature lists, modes
    <root>/tier1/<head>.ubj       the four funnel heads, XGBoost native
    <root>/tier2/win.ubj          the win classifier
    <root>/encoders/<name>.parquet  the empirical-Bayes cell tables

The encoders are as load-bearing as the models. A head's feature list names
`_enc_ssp_price`, and without the cell table the column cannot be rebuilt, so a
bundle without its encoders can be loaded and cannot score anything.

WHAT IS DELIBERATELY NOT SAVED. No training data, no test split, no per-row
predictions: those are the eval file's job and are written by a separate pass
that LOADS this bundle. That split is the point. It puts the frozen artifact on
the critical path, so the numbers reported come from the file that shipped and
the two cannot silently diverge.

THREE SILENT FAILURES THIS GUARDS AGAINST, all seen in practice:

    spend_scale lost      the spend head is lognormal and its scale is needed to
                          take the mean, exp(mu + s^2/2). Absent, the ev is
                          wrong by a factor that looks plausible.
    spend_scale None -> 0 an UNAVAILABLE head has no scale, and `None` is not
                          `0.0`. `Tier1.predict` reads `self.scale or 0.0` and
                          the CRPS path reads `t1.scale or 1.0`, so a zero
                          written where None belongs changes both. At 1,000 rows
                          there are no payers and this path is the normal one.
    best_iteration lost   early stopping picks a tree count; predict with the
                          full forest instead and every number moves.

Any of these produces plausible wrong numbers rather than an error, so each is
checked at save and raised on at load rather than defaulted.

ON best_iteration, MEASURED RATHER THAN ASSUMED. XGBoost 3.x carries it inside
the model file: save and reload a fitted estimator and `best_iteration` comes
back, with predictions identical to the bit. It is also a read-only property, so
it CANNOT be restored from the manifest. The manifest records it anyway and load
VERIFIES it, which is the useful half: if a future XGBoost stops persisting it,
this raises instead of quietly predicting with the full forest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "t9v2-bundle-2"

# "-2" since step 7, 23 August 2026: the bundle gained `tier2/price.ubj` and a
# `model.price` block. A "-1" bundle has no price head and cannot be scored by
# this build, so the version is refused rather than tolerated -- reading one
# would silently produce a run with one bidder where the results file claims two.


class BundleError(RuntimeError):
    """A bundle that cannot be trusted. Never a warning: the numbers look fine."""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(name, x):
    """A number that must be real. None is allowed and preserved; NaN is not."""
    if x is None:
        return None
    v = float(x)
    if not np.isfinite(v):
        raise BundleError("%s is %r, which cannot be saved as a number" % (name, x))
    return v


def _best_iteration(model):
    """Early stopping's tree count, or None when it did not stop early."""
    for attr in ("best_iteration", "best_ntree_limit"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    return None


# --------------------------------------------------------------------- encoders

def _save_encoders(enc, root: Path) -> dict:
    """Each encoder's cell table, its parent table and its root, as parquet.

    Written as a frame rather than pickled so a bundle can be read by anything
    that reads parquet, and so a diff of two bundles is a diff of numbers.
    """
    root.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name, e in enc.items():
        # sum and count travel WITH the eb value, because leave-one-out needs
        # the sufficient statistics: an EB mean cannot be un-averaged from the
        # mean alone. A bundle without them can score but cannot re-fit, and
        # `transform_train` refuses rather than quietly returning the in-fold
        # value.
        cell = e.cell.reset_index()
        cell.columns = list(e.keys) + ["eb"]
        cell["sum"] = e.cell_sum.to_numpy()
        cell["count"] = e.cell_count.to_numpy()
        par = e.parent.reset_index()
        par.columns = list(par.columns[:-1]) + ["eb"]
        par["sum"] = e.par_sum.to_numpy()
        par["count"] = e.par_count.to_numpy()
        cell.to_parquet(root / ("%s.cell.parquet" % name), index=False)
        par.to_parquet(root / ("%s.parent.parquet" % name), index=False)
        extra = {}
        for nm, (series, r) in (e.extra or {}).items():
            ex = series.reset_index()
            ex.columns = list(e.keys) + ["eb"]
            ex.to_parquet(root / ("%s.extra.%s.parquet" % (name, nm)), index=False)
            extra[nm] = {"root": _finite("%s.extra.%s.root" % (name, nm), r)}
        rec[name] = {"keys": list(e.keys), "shrink": float(e.shrink),
                     "root": _finite("%s.root" % name, e.root), "extra": extra,
                     "value_col": e.value_col, "mask_kind": e.mask_kind,
                     "cells": int(len(cell)), "parents": int(len(par))}
    return rec


def _check_best_iteration(what, model, rec):
    """The model file carries it; this only checks it came back.

    It is a read-only property, so it cannot be restored from the manifest. If a
    future XGBoost stops persisting it the model would silently predict with the
    full forest rather than the tree count early stopping chose, which moves
    every number and raises nothing. So: compare, and refuse.
    """
    want = rec.get("best_iteration")
    got = _best_iteration(model)
    if want != got:
        raise BundleError(
            "%s: best_iteration was %r when saved and is %r after loading. The "
            "model file no longer carries it, so predictions would use the full "
            "forest instead of the %r trees early stopping chose."
            % (what, want, got, want))


def _load_encoders(rec, root: Path):
    from .train.encoder import PriceEncoder
    out = {}
    for name, r in rec.items():
        e = PriceEncoder(r["keys"], r["shrink"], name,
                         r.get("value_col"), r.get("mask_kind"))
        cell = pd.read_parquet(root / ("%s.cell.parquet" % name))
        par = pd.read_parquet(root / ("%s.parent.parquet" % name))
        ci = cell.set_index(list(r["keys"]))
        pkeys = [c for c in par.columns if c not in ("eb", "sum", "count")]
        pi = par.set_index(pkeys)
        e.cell, e.parent = ci["eb"], pi["eb"]
        e.cell_sum, e.cell_count = ci["sum"], ci["count"]
        e.par_sum, e.par_count = pi["sum"], pi["count"]
        e.root = float(r["root"])
        e.extra = {}
        for nm, ex in (r.get("extra") or {}).items():
            f = pd.read_parquet(root / ("%s.extra.%s.parquet" % (name, nm)))
            e.extra[nm] = (f.set_index(list(r["keys"]))["eb"], float(ex["root"]))
        out[name] = e
    return out


# ------------------------------------------------------------------------ save

def save_bundle(fitted, settings, path, meta=None):
    """Freeze one view's fitted pieces. `fitted` is train_view's second return.

    BUILT ASIDE AND SWAPPED IN, never cleared first. A bundle is a run artifact
    under `output/`, and clear-then-write opens a window in which neither the old
    bundle nor a whole new one exists: a save that dies partway — out of disk at
    10M, a killed process — destroys the artifact it was replacing and leaves
    behind a directory that still looks like a bundle. Writing into
    `<name>.partial` and renaming means the worst case leaves the previous
    bundle standing.

    The only things this function removes are its own staging directory and the
    superseded bundle, and the second only after the replacement is whole.
    """
    final = Path(path)
    root = final.with_name(final.name + ".partial")
    if root.exists():
        shutil.rmtree(root)            # this function's own leftover, nothing else
    (root / "tier1").mkdir(parents=True)
    (root / "tier2").mkdir(parents=True)

    t1, t2 = fitted["tier1"], fitted["tier2"]
    files, heads = {}, {}

    for head, h in t1.models.items():
        rec = {"mode": h.mode, "n": int(h.n),
               "const": _finite("tier1.%s.const" % head, h.const)}
        if h.mode == "model":
            f = root / "tier1" / ("%s.ubj" % head)
            h.model.save_model(str(f))
            rec["file"] = str(f.relative_to(root)).replace("\\", "/")
            rec["best_iteration"] = _best_iteration(h.model)
            files[rec["file"]] = _sha256(f)
        heads[head] = rec

    f = root / "tier2" / "win.ubj"
    t2.model.save_model(str(f))
    win = {"mode": "model", "file": "tier2/win.ubj",
           "best_iteration": _best_iteration(t2.model), "n": int(t2.n_train)}
    files["tier2/win.ubj"] = _sha256(f)

    # THE PRICE HEAD, step 6. A native Booster rather than a sklearn wrapper,
    # because interval-censored AFT needs label_lower_bound / label_upper_bound
    # on a DMatrix and the wrapper has no path to them. `sigma` is saved beside
    # it and is not decoration: for C1 and C2 it shaped the likelihood that
    # produced these trees, so a bundle that lost it could not say what it fitted.
    ph = fitted.get("price")
    price = None
    if ph is not None:
        pf = root / "tier2" / "price.ubj"
        ph.model.save_model(str(pf))
        price = {"mode": "model", "file": "tier2/price.ubj",
                 "sigma": _finite("price.sigma", ph.sigma),
                 "exact_labels": bool(ph.exact),
                 "cols": list(ph.cols), "n": int(ph.n_train)}
        files["tier2/price.ubj"] = _sha256(pf)

    manifest = {
        "schema": SCHEMA,
        "model": {
            "heads": heads,
            "win": win,
            "price": price,
            # None survives as None. An unavailable spend head has no scale, and
            # writing 0.0 here changes both the ev and the CRPS silently.
            "spend_scale": _finite("spend_scale", t1.scale),
            "encoders": _save_encoders(fitted["encoders"], root / "encoders"),
        },
        "contract": {"t1_cols": list(fitted["t1_cols"]),
                     "t2_cols": list(fitted["t2_cols"]),
                     "price_cols": list(fitted.get("price_cols") or [])},
        "files": files,
        "meta": dict(meta or {}),
    }
    for name in manifest["model"]["encoders"]:
        for suffix in ("cell", "parent"):
            rel = "encoders/%s.%s.parquet" % (name, suffix)
            files[rel] = _sha256(root / rel)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    # the swap. Everything above is on disk and hashed before anything that was
    # already there is touched, so a failure anywhere above this line leaves the
    # previous bundle exactly as it was.
    old = final.with_name(final.name + ".superseded")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    root.rename(final)
    if old.exists():
        shutil.rmtree(old)
    return final


# ------------------------------------------------------------------------ load

def load_bundle(path):
    """Rebuild the fitted pieces. Raises rather than defaulting on anything absent."""
    import xgboost as xgb
    from .train.tier1 import Head, Tier1
    from .train.tier2 import Tier2

    root = Path(path)
    mf = root / "manifest.json"
    if not mf.exists():
        raise BundleError("no manifest at %s" % mf)
    m = json.loads(mf.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA:
        raise BundleError("bundle schema %r, this build reads %r"
                          % (m.get("schema"), SCHEMA))
    for rel, want in m["files"].items():
        got = _sha256(root / rel)
        if got != want:
            raise BundleError("%s has changed since it was saved" % rel)

    mm = m["model"]
    if "spend_scale" not in mm:
        raise BundleError("spend_scale absent; it is needed to take the mean of "
                          "the lognormal spend head and must not be defaulted")

    t1 = Tier1()
    t1.cols = list(m["contract"]["t1_cols"])
    t1.scale = mm["spend_scale"]          # None stays None
    for head, rec in mm["heads"].items():
        model = None
        if rec["mode"] == "model":
            model = xgb.XGBRegressor() if head == "spend" else xgb.XGBClassifier()
            model.load_model(str(root / rec["file"]))
            _check_best_iteration("tier1.%s" % head, model, rec)
        h = Head(head, rec["mode"], model=model, const=rec["const"],
                 scale=(t1.scale if head == "spend" else None), n=rec["n"])
        t1.models[head] = h
        t1.mode[head] = rec["mode"]
        t1.n_train[head] = rec["n"]

    t2 = Tier2()
    t2.cols = list(m["contract"]["t2_cols"])
    t2.n_train = mm["win"]["n"]
    t2.model = xgb.XGBClassifier()
    t2.model.load_model(str(root / mm["win"]["file"]))
    _check_best_iteration("tier2.win", t2.model, mm["win"])

    # the price head. A raw Booster, so its own class is rebuilt around it and
    # sigma comes back with it: for C1 and C2 sigma shaped the likelihood that
    # produced these trees, and a bundle that dropped it could not say what it
    # fitted. A bundle written before step 6 simply has none, and says so with
    # None rather than a stand-in.
    # SIGMA IS MANDATORY, and its absence RAISES rather than defaulting. Sigma
    # sets the width of the win curve, the width sets the chosen bid, and the bid
    # sets every economic number this head reports. A bundle that quietly rebuilt
    # its curve at the wrong width would produce plausible wrong economics and
    # nothing would say so -- which is precisely v1's AFT_SCALE defect, moved to
    # a new place.
    ph = None
    if mm.get("price"):
        if mm["price"].get("sigma") is None:
            raise BundleError(
                "the price head has no sigma in its manifest. Sigma sets the win "
                "curve width, which sets the chosen bid, which sets every "
                "economic number this head reports. Refusing to default.")
        from .train.price import PriceHead
        ph = PriceHead()
        ph.cols = list(m["contract"]["price_cols"])
        ph.sigma = float(mm["price"]["sigma"])
        ph.exact = bool(mm["price"]["exact_labels"])
        ph.n_train = int(mm["price"]["n"])
        ph.model = xgb.Booster()
        ph.model.load_model(str(root / mm["price"]["file"]))

    return {"tier1": t1, "tier2": t2, "price": ph,
            "encoders": _load_encoders(mm["encoders"], root / "encoders"),
            "t1_cols": t1.cols, "t2_cols": t2.cols,
            "price_cols": list(m["contract"].get("price_cols") or []),
            "manifest": m}
