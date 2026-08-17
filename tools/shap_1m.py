# -*- coding: utf-8 -*-
"""Stage 6 step 6: one SHAP run at 1M, all 4 views.

    python tools/shap_1m.py [--seed 20250] [--sample 50000]

Writes `docs/gates/shap_1m.md` and `docs/gates/shap_1m.json`.

WHY THIS RUN EXISTS. The contrast tables say how much each layer is worth. They
do not say WHICH columns carry it, and for the SSP result that is the whole
question: C3 is handed 2 extra columns and gains nothing, so the informative
thing is whether the model reaches for them and finds them empty, or never
reaches for them at all. A SHAP run answers that directly, per feature.

NO `shap` DEPENDENCY. XGBoost implements exact TreeSHAP itself, reached through
`Booster.predict(..., pred_contribs=True)`. That returns one contribution per
feature per row plus a bias term, and it is the same algorithm the `shap`
package calls for tree models. Adding a package to get a function the installed
one already exposes would be a dependency taken on for nothing.

WHAT IS REPORTED. Mean absolute contribution per feature, in log-odds for the
classifiers, averaged over a sample of TEST rows. Mean absolute rather than mean:
contributions are signed and a feature that pushes hard in both directions
averages to nothing while mattering a great deal. The sample is drawn from test
only, so no row it reports on was fitted.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t9v2 import censor as CEN  # noqa: E402
from t9v2.core.config import load  # noqa: E402
from t9v2.train import encoder as E, features as F  # noqa: E402
from t9v2.train.tier1 import Tier1  # noqa: E402
from t9v2.train.tier2 import Tier2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
VIEWS = ["C1", "C2", "C3", "C4"]
# What SSP adds, and it does NOT arrive as a raw column. `winning_price` and
# `bid_density` are consumed by the empirical-Bayes price encoders and reach the
# model as `_enc_ssp_price`, `_enc_ssp_lost_price` and `_enc_ssp_density`. Naming
# the raw columns here found nothing and reported "not present" for all 4 views,
# which would have read as the SSP features being absent when they were simply
# under their encoded names.
def is_ssp(feature):
    return feature.startswith("_enc_ssp")


def booster_of(model):
    """The XGBoost booster inside whatever wrapper it arrived in.

    Tier 1 hands back a `Head`, which holds its estimator on `.model` and may
    hold None when the head fell back to an intercept because the view had too
    few positive labels to fit. Tier 2 hands back the estimator directly.
    """
    m = getattr(model, "model", model)
    return m.get_booster() if hasattr(m, "get_booster") else None


def contribs(model, X):
    """Mean |TreeSHAP| per feature, exact, via XGBoost's own implementation."""
    import xgboost as xgb
    booster = booster_of(model)
    if booster is None:
        return None
    dm = xgb.DMatrix(X, enable_categorical=True)
    c = booster.predict(dm, pred_contribs=True)
    # last column is the bias, which is not a feature
    return np.abs(c[:, :-1]).mean(axis=0)


def run_view(master, view, s, seed, sample, rng):
    censored = CEN.censor(master, view, s)
    d = F.prepare(censored, s)
    del censored
    masks = F.split(d, s)
    enc = E.build(d[masks["train"]].copy(), view, s)
    enc_cols = E.apply(enc, d)
    t1_cols = F.tier1_features(d, "click", extra=enc_cols)
    t2_cols = F.tier2_features(d, extra=enc_cols)

    t1 = Tier1().fit(d, masks, t1_cols, s, seed)
    t2 = Tier2().fit(d, masks, t2_cols, s, seed)

    te = np.flatnonzero(np.asarray(masks["test"]))
    if len(te) > sample:
        te = rng.choice(te, sample, replace=False)
    d_te = d.iloc[te]

    out = {}
    for name, model, cols in [("click", t1.models.get("click"), t1_cols),
                              ("win", t2.model, t2_cols)]:
        if model is None:
            continue
        vals = contribs(model, F.matrix(d_te, cols))
        if vals is None:          # head fell back to an intercept, nothing to attribute
            continue
        tot = float(vals.sum()) or 1.0
        out[name] = sorted(
            [{"feature": c, "mean_abs_shap": float(v), "share": float(v) / tot}
             for c, v in zip(cols, vals)],
            key=lambda r: -r["mean_abs_shap"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20250)
    ap.add_argument("--sample", type=int, default=50000)
    a = ap.parse_args(argv)

    s = load("default")
    path = OUT / ("t9v2_1M_seed%d.parquet" % a.seed)
    if not path.exists():
        raise SystemExit("missing %s" % path)
    master = pd.read_parquet(path)
    rng = np.random.default_rng(0)

    res = {}
    for v in VIEWS:
        res[v] = run_view(master, v, s, 0, a.sample, rng)
        top = res[v]["win"][0]
        print("%s  win top feature %s (%.1f%% of total)"
              % (v, top["feature"], 100 * top["share"]), flush=True)

    md = ["# SHAP at 1M, all 4 views", "",
          "*Generated by `tools/shap_1m.py`. This file is written, never "
          "edited.*", "",
          "Seed %d, 1M rows, mean absolute TreeSHAP over %s sampled TEST rows. "
          "Exact TreeSHAP from XGBoost's own `pred_contribs`, not an "
          "approximation and not the `shap` package. Contributions are in "
          "log-odds. Mean ABSOLUTE, because a feature that pushes hard in both "
          "directions averages to nothing while mattering a great deal."
          % (a.seed, "{:,}".format(a.sample)), ""]

    for head in ("win", "click"):
        md += ["---", "", "## The %s head" % head, ""]
        for v in VIEWS:
            rows = res[v].get(head)
            if not rows:
                continue
            md += ["### %s" % v, "",
                   "| Rank | Feature | mean \\|SHAP\\| | share |",
                   "|---:|---|---:|---:|"]
            for i, r in enumerate(rows[:12], 1):
                md.append("| %d | `%s` | %.4f | %.1f%% |"
                          % (i, r["feature"], r["mean_abs_shap"],
                             100 * r["share"]))
            md.append("")

    # --- the question this run was commissioned to answer
    md += ["---", "", "## What the SSP columns are worth to the win head", "",
           "C3 and C4 see `winning_price` and `bid_density`, which reach the "
           "model as the `_enc_ssp_*` encoders; C1 and C2 have no such columns. "
           "Two outcomes would mean different things. If the encoders sit at the "
           "bottom of the table, the SSP columns are empty and the model ignored "
           "them. If the model reaches for them and the win AUC still does not "
           "move, they are not empty but REDUNDANT, carrying only what the view "
           "could already reconstruct from the columns it had.", "",
           "| View | SSP encoders | share of total \\|SHAP\\| | best rank |",
           "|---|---:|---:|---:|"]
    for v in VIEWS:
        rows = res[v]["win"]
        present = [(i, r) for i, r in enumerate(rows, 1) if is_ssp(r["feature"])]
        if not present:
            md.append("| %s | none | - | - |" % v)
            continue
        share = sum(r["share"] for _, r in present)
        md.append("| %s | %d | %.1f%% | %d of %d |"
                  % (v, len(present), 100 * share,
                     min(i for i, _ in present), len(rows)))
    md += ["", "The public price columns C1 already has for comparison:", "",
           "| View | `floor_price` + `bid_price` + `slot_format` |", "|---|---:|"]
    for v in VIEWS:
        base = sum(r["share"] for r in res[v]["win"]
                   if r["feature"] in ("floor_price", "bid_price", "slot_format"))
        md.append("| %s | %.1f%% |" % (v, 100 * base))

    d = ROOT / "docs" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    (d / "shap_1m.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (d / "shap_1m.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote docs/gates/shap_1m.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
