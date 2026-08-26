# -*- coding: utf-8 -*-
"""SHAP attribution over every fitted head, all 4 views.

    python tools/shap_1m.py [--seed 20250] [--scale 10M] [--sample 50000]
    python tools/shap_1m.py --master output/wiring_1K_seed20250.parquet \
                            --bundles output/bundles --tag 1K

Writes `docs/shap_attribution_<tag>_seed<seed>.{md,json}`.

WHY THIS RUN EXISTS. The contrast tables say how much each layer is worth. They
do not say WHICH columns carry it, and for the SSP result that is the whole
question: C3 is handed extra columns and gains nothing on the funnel, so the
informative thing is whether the model reaches for them and finds them empty, or
never reaches for them at all. A SHAP run answers that directly, per feature.

THE HEAD LIST WAS AN ACCIDENT OF CHRONOLOGY AND IS NOW DELIBERATE. This tool was
written 17 August at v2's stage 6, when Tier 1 was attributed on `click` alone
and the price head did not exist -- `train/price.py` was first committed on
23 August. Nothing decided that `install`, `payer`, `spend` and `price` should be
excluded; they were simply not there or not thought about. All six fitted heads
are attributed now.

WHICH HEAD ANSWERS WHICH QUESTION.
  click, install, payer, spend  receive every SSP encoder in C3 and C4 while
      their labels are identical in all four views, so their attribution is the
      FEATURE-channel evidence. This is the load-bearing set.
  price                         has 20 features identical in all four views, every
      SSP encoder barred from it by name, so its C1-vs-C3 difference is a pure
      LABEL-quality effect: the same feature list, attributed by a model that saw
      exact clearing prices rather than bounds.
  win                           is the feature channel again, on the head that is
      being demoted out of the dissertation body. Kept for the reflective essay.

NO REFITTING. The per-seed bundles under `output/runs/<scale>/seed<n>/bundle/`
hold the models of record, so this attributes the trees the reported numbers came
from rather than fresh ones trained beside them. Earlier this tool refit every
head, which cost about 35 minutes per seed at 10M and attributed a model nobody
had reported.

NO `shap` DEPENDENCY. XGBoost implements exact TreeSHAP itself, reached through
`Booster.predict(..., pred_contribs=True)`. That returns one contribution per
feature per row plus a bias term, and it is the same algorithm the `shap`
package calls for tree models. Adding a package to get a function the installed
one already exposes would be a dependency taken on for nothing.

WHAT IS REPORTED. Mean absolute contribution per feature, averaged over a sample
of TEST rows. Mean absolute rather than mean: contributions are signed and a
feature that pushes hard in both directions averages to nothing while mattering a
great deal. The sample is drawn from test only, so no row it reports on was
fitted.

UNITS DIFFER BY HEAD AND `share` IS THE COMPARABLE COLUMN. The four classifiers
contribute in log-odds, `spend` in log-dollars, `price` in the AFT model's own
log score. Raw `mean_abs_shap` is therefore not comparable ACROSS heads. `share`
is normalised within its own head and is.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t9v2 import bundle as BUN  # noqa: E402
from t9v2 import censor as CEN  # noqa: E402
from t9v2.core.config import load  # noqa: E402
from t9v2.train import encoder as E, features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
VIEWS = ["C1", "C2", "C3", "C4"]

# The order heads are reported in. Tier 1 first because it now carries the
# feature-channel claim, then the two Tier 2 heads.
HEADS = ["click", "install", "payer", "spend", "price", "win"]


# What SSP adds, and it does NOT arrive as a raw column. `winning_price` and
# `bid_density` are consumed by the empirical-Bayes price encoders and reach the
# model as `_enc_ssp_price`, `_enc_ssp_lost_price`, `_enc_ssp_density` and
# `_enc_ssp_minwin_price`. Naming the raw columns here found nothing and reported
# "not present" for all 4 views, which would have read as the SSP features being
# absent when they were simply under their encoded names.
def is_ssp(feature):
    return feature.startswith("_enc_ssp")


def booster_of(model):
    """The XGBoost booster inside whatever wrapper it arrived in.

    Tier 1 hands back a `Head`, which holds its estimator on `.model` and may
    hold None when the head fell back to an intercept because the view had too
    few positive labels to fit. Tier 2 hands back the estimator directly. The
    price head is a `PriceHead` holding a RAW Booster on `.model`, which has no
    `get_booster` and would have returned None here -- the head would have been
    silently skipped rather than attributed.
    """
    import xgboost as xgb
    m = getattr(model, "model", model)
    if isinstance(m, xgb.Booster):
        return m
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


def run_view(master, view, s, bundle_root, sample, rng):
    """Attribute every fitted head of one view, from its saved bundle."""
    b = BUN.load_bundle(bundle_root / view)

    censored = CEN.censor(master, view, s)
    d = F.prepare(censored, s)
    del censored
    masks = F.split(d, s)
    # THE SAVED ENCODERS, NOT REBUILT ONES, and with the same `train_mask` the
    # runner passed. The mask only changes fitted rows, which this run never
    # scores, but reproducing the call exactly means the frame handed to the
    # booster is the frame the booster was scored on.
    E.apply(b["encoders"], d, train_mask=masks["train"])

    te = np.flatnonzero(np.asarray(masks["test"]))
    if len(te) > sample:
        te = rng.choice(te, sample, replace=False)
    d_te = d.iloc[te]

    t1, t2, ph = b["tier1"], b["tier2"], b["price"]
    todo = [(h, t1.models.get(h), b["t1_cols"]) for h in
            ("click", "install", "payer", "spend")]
    todo.append(("price", ph, b["price_cols"]))
    todo.append(("win", t2.model, b["t2_cols"]))

    out = {}
    for name, model, cols in todo:
        if model is None:                 # price head absent from an old bundle
            continue
        vals = contribs(model, F.matrix(d_te, cols))
        if vals is None:                  # head fell back to an intercept
            continue
        tot = float(vals.sum()) or 1.0
        out[name] = sorted(
            [{"feature": c, "mean_abs_shap": float(v), "share": float(v) / tot}
             for c, v in zip(cols, vals)],
            key=lambda r: -r["mean_abs_shap"])
    return out


def ssp_rows(res, head):
    """Per-view SSP-encoder summary for one head, or None if it has no SSP columns."""
    rows, any_ssp = [], False
    for v in VIEWS:
        got = res[v].get(head)
        if not got:
            rows.append((v, None))
            continue
        present = [(i, r) for i, r in enumerate(got, 1) if is_ssp(r["feature"])]
        if present:
            any_ssp = True
        rows.append((v, (present, len(got))))
    return rows if any_ssp else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20250)
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--scale", default="10M", choices=["100K", "1M", "10M"])
    # Overrides, so a wiring test can point at the 1K master and the 1K smoke
    # bundle without either becoming a special case inside this file.
    ap.add_argument("--master", default=None, help="explicit master parquet")
    ap.add_argument("--bundles", default=None,
                    help="explicit bundle root holding C1..C4")
    ap.add_argument("--tag", default=None, help="output stem tag, defaults to scale")
    a = ap.parse_args(argv)

    s = load("default")
    path = (Path(a.master) if a.master
            else OUT / ("t9v2_%s_seed%d.parquet" % (a.scale, a.seed)))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit("missing master %s" % path)

    broot = (Path(a.bundles) if a.bundles
             else OUT / "runs" / a.scale / ("seed%d" % a.seed) / "bundle")
    if not broot.is_absolute():
        broot = ROOT / broot
    if not broot.exists():
        raise SystemExit("missing bundle root %s" % broot)

    tag = a.tag or a.scale
    print("master  %s" % path)
    print("bundles %s" % broot, flush=True)

    master = pd.read_parquet(path)
    rng = np.random.default_rng(0)

    res = {}
    for v in VIEWS:
        res[v] = run_view(master, v, s, broot, a.sample, rng)
        got = ", ".join("%s:%d" % (h, len(res[v][h]))
                        for h in HEADS if h in res[v])
        print("%s  attributed -> %s" % (v, got or "nothing"), flush=True)

    md = ["# SHAP attribution, %s, all 4 views" % tag, "",
          "*Generated by `tools/shap_1m.py`. This file is written, never "
          "edited.*", "",
          "Seed %d, master `%s`, mean absolute TreeSHAP over %s sampled TEST "
          "rows, taken from the FROZEN bundles rather than refitted. Exact "
          "TreeSHAP from XGBoost's own `pred_contribs`, not an approximation and "
          "not the `shap` package. Mean ABSOLUTE, because a feature that pushes "
          "hard in both directions averages to nothing while mattering a great "
          "deal."
          % (a.seed, path.name, "{:,}".format(a.sample)), "",
          "**Units differ by head.** The four classifiers contribute in log-odds, "
          "`spend` in log-dollars, `price` in the AFT model's own log score. "
          "`mean |SHAP|` is therefore NOT comparable across heads. `share` is "
          "normalised within its head and is.", ""]

    # ------------------------------------------------------------------
    # THE MARKDOWN ANSWERS QUESTIONS, THE JSON HOLDS THE DATA. An earlier
    # version printed a rank-12 table per head per view, which is 22 tables and
    # about 12,000 words of numbers with no claim attached to any of them. Every
    # figure is still in the `.json` beside this file, in full and unranked.
    # ------------------------------------------------------------------

    # --- 1. does the model reach for the SSP columns?
    md += ["---", "", "## 1. Do the SSP columns get used?", "",
           "C3 and C4 see `winning_price` and `bid_density`, which reach the "
           "model as the `_enc_ssp_*` encoders; C1 and C2 have no such columns. "
           "Two outcomes mean different things. If the encoders sit at the bottom "
           "of the table, the SSP columns are empty and the model ignored them. "
           "If the model reaches for them and the metric still does not move, "
           "they are not empty but REDUNDANT, carrying only what the view could "
           "already reconstruct from the columns it had.", "",
           "| Head | SSP share, C3 | best rank, C3 | SSP share, C4 | "
           "best rank, C4 |",
           "|---|---:|---:|---:|---:|"]
    for head in HEADS:
        tab = ssp_rows(res, head)
        if tab is None:
            continue
        got = dict(tab)

        def _share(v):
            g = got.get(v)
            return sum(r["share"] for _, r in g[0]) if g and g[0] else None

        def _rank(v):
            g = got.get(v)
            return min(i for i, _ in g[0]) if g and g[0] else None

        def _n(v):
            g = got.get(v)
            return g[1] if g else None

        s3, s4, r3, r4 = _share("C3"), _share("C4"), _rank("C3"), _rank("C4")
        md.append("| `%s` | %s | %s | %s | %s |" % (
            head,
            "-" if s3 is None else "%.1f%%" % (100 * s3),
            "-" if r3 is None else "%d of %d" % (r3, _n("C3")),
            "-" if s4 is None else "%.1f%%" % (100 * s4),
            "-" if r4 is None else "%d of %d" % (r4, _n("C4"))))
    md.append("")
    md += ["The `price` head is absent from this table BY CONSTRUCTION. Its 20 "
           "features are identical in all four views, every SSP encoder barred "
           "from it by name, which is what makes its contrast a pure "
           "label-quality effect. There are no SSP columns in it to attribute.",
           ""]

    # --- 2. the price head, censored labels against exact ones
    if res["C1"].get("price") and res["C3"].get("price"):
        a1 = {r["feature"]: r["share"] for r in res["C1"]["price"]}
        a3 = {r["feature"]: r["share"] for r in res["C3"]["price"]}
        moved = sorted(a1, key=lambda f: -abs(a3.get(f, 0.0) - a1[f]))
        md += ["---", "",
               "## 2. What changes in the price head when the labels go exact",
               "",
               "The same twenty features in every view, so nothing here is a "
               "feature effect. C1 and C2 see only a BOUND on the clearing price, "
               "won rows below their bid and lost rows above it. C3 and C4 see "
               "the price itself. Any difference below is the model reacting to "
               "label quality alone.", "",
               "| Feature | C1, censored | C3, exact | change |",
               "|---|---:|---:|---:|"]
        for f in moved[:8]:
            d1, d3 = 100 * a1[f], 100 * a3.get(f, 0.0)
            md.append("| `%s` | %.1f%% | %.1f%% | %+.1f pts |"
                      % (f, d1, d3, d3 - d1))
        md.append("")
        top1 = res["C1"]["price"][0]["feature"]
        top3 = res["C3"]["price"][0]["feature"]
        if top1 != top3:
            md += ["The head's leading feature changes, from `%s` under censored "
                   "labels to `%s` under exact ones." % (top1, top3), ""]

    # --- 3. one compact orientation table instead of 22 ranked ones
    md += ["---", "", "## 3. What each head leans on", "",
           "Top three features by share, per head per view. Full rankings for "
           "every feature are in the `.json` beside this file.", "",
           "| Head | View | 1st | 2nd | 3rd |", "|---|---|---|---|---|"]
    for head in HEADS:
        if not any(head in res[v] for v in VIEWS):
            continue
        for v in VIEWS:
            rows = res[v].get(head)
            if not rows:
                md.append("| `%s` | %s | *intercept, nothing to attribute* | | |"
                          % (head, v))
                continue
            cells = ["`%s` %.0f%%" % (r["feature"], 100 * r["share"])
                     for r in rows[:3]]
            while len(cells) < 3:
                cells.append("")
            md.append("| `%s` | %s | %s |" % (head, v, " | ".join(cells)))
    md.append("")

    # THE OUTPUT IS NAMED FOR ITS SCALE AND ITS SEED, AND NOT `shap_1m`.
    # `docs/gates/shap_1m.*` is v2's artifact, written 17 August against a model
    # whose C3 win head had 24 features; v2.2's has 25, the extra one being
    # `_enc_ssp_minwin_price`. The two must be distinguishable, and `shap_1M.json`
    # would not do it: Windows paths are case-insensitive, so it IS `shap_1m.json`
    # and a 1M re-run would silently overwrite the v2 record. The SEED is in the
    # stem because this run is three seeds, and a stem without one would have let
    # the second overwrite the first.
    #
    # NOT IN `docs/gates/`. A gate file exists only if its checks passed and is
    # written by a gate script. This is a report and asserts no verdict, so it
    # sits in `docs/` proper.
    d = ROOT / "docs"
    d.mkdir(parents=True, exist_ok=True)
    stem = "shap_attribution_%s_seed%d" % (tag, a.seed)
    (d / (stem + ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    (d / (stem + ".json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote docs/%s.md and .json" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
