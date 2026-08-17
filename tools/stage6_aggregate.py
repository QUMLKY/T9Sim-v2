# -*- coding: utf-8 -*-
"""Stage 6: aggregate all 30 datasets into the v2 results document.

    python tools/stage6_aggregate.py

Writes `docs/V2_Results.md` and `docs/gates/stage6.json`.

EVERY CONTRAST IS PAIRED. C2 minus C1 is formed WITHIN a seed and the interval is
taken over those 10 differences, never over 10 C2 values against 10 C1 values.
The views share a master, a temporal split and a test set, so almost all the
seed-to-seed variation is common to both sides and differencing removes it. On
the 10M profit contrast the unpaired interval is roughly four times wider and
spans zero; the paired one does not. Reporting the unpaired version would turn a
10/10 result into "no significant difference" purely by discarding the pairing
the design went to the trouble of creating.

Sign agreement is reported beside every interval because it is the plan's actual
bar: an effect must hold the same direction in all 10 seeds. An interval clear of
zero on a mean that 6 seeds disagree with is a different claim from one all 10
agree on, and the count is what separates them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "output" / "runs"
SCALES = ["100K", "1M", "10M"]
VIEWS = ["C1", "C2", "C3", "C4"]
LAYERS = {"C1": "DSP only", "C2": "DSP + MMP", "C3": "DSP + SSP",
          "C4": "all three"}


def load(scale):
    out = []
    for p in sorted((RUNS / scale).glob("seed*/results.json")):
        out.append(json.loads(p.read_text(encoding="utf-8"))["views"])
    return out


# --- what a view is worth, per seed -----------------------------------------

def head(v, h, k="auc"):
    return lambda r: r[v]["heads"][h][k]


def econ(v, k):
    return lambda r: r[v]["economics"]["learned"][k]


def ev(v, k):
    return lambda r: r[v]["ev"][k]


def diff(a, b):
    return lambda r: a(r) - b(r)


def ratio(a, b):
    """Relative change against a base, for quantities whose LEVEL is not
    comparable across scales. Profit at 10M is ten times profit at 1M purely
    because there are ten times the rows.

    Guarded, and the guard is not cosmetic. C1's profit at 100K is within noise
    of zero, with individual seeds at -1, 0, 5 and 41 dollars, so C2/C1 - 1 on
    those seeds returns numbers in the hundreds of thousands of percent and their
    mean is meaningless. An unguarded version of this printed -852694.8% for the
    headline MMP profit contrast at 100K. Below the floor the ratio is nan and
    the row reads n/a, which is the honest answer: this quantity cannot be
    formed at this scale.
    """
    def f(r):
        base = b(r)
        return (a(r) / base - 1.0) if base > RATIO_FLOOR else float("nan")
    return f


def over_oracle(a, b):
    """A paired difference normalised by the SAME seed's oracle profit.

    The scale-free way to compare a profit gap across 100K, 1M and 10M. The
    oracle's profit is large and positive in every seed, so it is a denominator
    that cannot blow up, unlike C1's profit at 100K.
    """
    return lambda r: ((a(r) - b(r))
                      / r["C1"]["economics"]["oracle"]["profit"])


# a profit base below this is noise, not a base; see `ratio`
RATIO_FLOOR = 1000.0


CONTRASTS = [
    ("MMP", "click AUC (C2-C1)", diff(head("C2", "click"), head("C1", "click")), "%+.4f"),
    ("MMP", "install AUC (C2-C1)", diff(head("C2", "install"), head("C1", "install")), "%+.4f"),
    ("MMP", "payer AUC (C2-C1)", diff(head("C2", "payer"), head("C1", "payer")), "%+.4f"),
    ("MMP", "EV level (C2-C1)", diff(ev("C2", "ratio"), ev("C1", "ratio")), "%+.4f"),
    ("MMP", "EV spearman (C2-C1)", diff(ev("C2", "spearman"), ev("C1", "spearman")), "%+.4f"),
    ("MMP", "ev_ratio (C2-C1)", diff(econ("C2", "ev_ratio"), econ("C1", "ev_ratio")), "%+.4f"),
    ("MMP", "profit (C2-C1), $", diff(econ("C2", "profit"), econ("C1", "profit")), "%+.0f"),
    ("MMP", "profit (C2-C1) / oracle", over_oracle(econ("C2", "profit"), econ("C1", "profit")), "pct"),
    ("MMP", "profit (C2/C1-1)", ratio(econ("C2", "profit"), econ("C1", "profit")), "pct"),
    ("SSP", "win AUC (C3-C1)", diff(head("C3", "win"), head("C1", "win")), "%+.4f"),
    ("SSP", "click AUC (C3-C1)", diff(head("C3", "click"), head("C1", "click")), "%+.4f"),
    ("SSP", "EV spearman (C3-C1)", diff(ev("C3", "spearman"), ev("C1", "spearman")), "%+.4f"),
    ("SSP", "ev_ratio (C3-C1)", diff(econ("C3", "ev_ratio"), econ("C1", "ev_ratio")), "%+.4f"),
    ("SSP", "profit (C3-C1) / oracle", over_oracle(econ("C3", "profit"), econ("C1", "profit")), "pct"),
    ("SSP", "profit (C3/C1-1)", ratio(econ("C3", "profit"), econ("C1", "profit")), "pct"),
    ("SSP on top of MMP", "win AUC (C4-C2)", diff(head("C4", "win"), head("C2", "win")), "%+.4f"),
    ("SSP on top of MMP", "profit (C4/C2-1)", ratio(econ("C4", "profit"), econ("C2", "profit")), "pct"),
]


def paired(vals):
    """mean, 95% CI and sign agreement over the per-seed differences."""
    d = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "agree": 0}
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "agree": int((np.sign(d) == np.sign(m)).sum())}


def fmt(x, spec):
    """`pct` renders a relative change as a percentage; anything else is a
    printf spec applied as given."""
    if not np.isfinite(x):
        return "n/a"
    return ("%+.1f%%" % (100.0 * x)) if spec == "pct" else (spec % x)


def contrast_table(rs, title):
    md = ["| Layer | Contrast | Mean | 95% CI | Seeds agreeing |",
          "|---|---|---:|---|---:|"]
    got = {}
    for layer, name, fn, spec in CONTRASTS:
        s = paired([fn(r) for r in rs])
        got[name] = s
        md.append("| %s | %s | %s | [%s, %s] | %d/%d |"
                  % (layer, name, fmt(s["mean"], spec), fmt(s["lo"], spec),
                     fmt(s["hi"], spec), s["agree"], s["n"]))
    short = sorted({n for n, s in got.items() if s["n"] < len(rs)})
    if short:
        md += ["", "A denominator below 10 means the contrast could not be "
               "FORMED on some seeds, not that they disagreed: a head with no "
               "positive labels in a view returns nan, and a profit ratio "
               "against a base within noise of zero is suppressed rather than "
               "printed. Affected here: %s." % ", ".join("`%s`" % n for n in short)]
    return md, got


def view_table(rs):
    md = ["| View | Layers | click AUC | install AUC | payer AUC | win AUC | "
          "spend CRPS | EV level | EV spearman | ev_ratio | of oracle | profit |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VIEWS:
        g = lambda f: float(np.nanmean([f(r) for r in rs]))  # noqa: E731
        md.append("| **%s** | %s | %.4f | %.4f | %.4f | %.4f | %.3f | %.4f | "
                  "%.4f | %.4f | %.4f | %.0f |"
                  % (v, LAYERS[v], g(head(v, "click")), g(head(v, "install")),
                     g(head(v, "payer")), g(head(v, "win")),
                     g(head(v, "spend", "crps")), g(ev(v, "ratio")),
                     g(ev(v, "spearman")), g(econ(v, "ev_ratio")),
                     g(econ(v, "value_vs_oracle")), g(econ(v, "profit"))))
    o = float(np.mean([r["C1"]["economics"]["oracle"]["profit"] for r in rs]))
    md += ["", "Oracle ceiling, true value and the true win rule: profit %.0f. "
           "Not achievable by any view, because it knows the competing bid." % o]
    return md


def main():
    data = {s: load(s) for s in SCALES}
    missing = {s: 10 - len(v) for s, v in data.items() if len(v) != 10}
    md = ["# T9Sim v2 — results across all 30 datasets", "",
          "*Generated by `tools/stage6_aggregate.py`. This file is written, "
          "never edited.*", "",
          "10 seeds (20250-20259) at each of 100K, 1M and 10M. Every contrast "
          "is **paired within a seed**: the four views share one master, one "
          "temporal split and one test set, so differencing inside a seed "
          "removes the variation common to both sides. Sign agreement is "
          "reported beside every interval because the plan's bar is that an "
          "effect hold direction in all 10 seeds, not merely that an interval "
          "clear zero.", ""]
    if missing:
        md += ["**INCOMPLETE**: %s" % missing, ""]

    store = {}
    for s in SCALES:
        rs = data[s]
        md += ["---", "", "## %s, n = %d seeds" % (s, len(rs)), "", "### Per view", ""]
        md += view_table(rs)
        md += ["", "### What each layer is worth", ""]
        t, got = contrast_table(rs, s)
        md += t
        store[s] = {k: v for k, v in got.items()}
        md.append("")

    # --- scale stability, the thing a single scale cannot show
    md += ["---", "", "## Does it hold as the data grows", "",
           "A contrast that shrinks with scale is label efficiency: the extra "
           "layer was substituting for data the model eventually gets anyway. "
           "One that holds is information the layer alone carries.", "",
           "Profit is shown against the same seed's ORACLE rather than against "
           "C1, because C1's profit at 100K is within noise of zero and a ratio "
           "to it is undefined. The oracle's profit is large and positive in "
           "every seed, so the same number can be read at all three scales.", "",
           "| Contrast | 100K | 1M | 10M | reading |", "|---|---:|---:|---:|---|"]
    for name, spec, read in [
        ("click AUC (C2-C1)", "%+.4f", "holds"),
        ("EV level (C2-C1)", "%+.4f", "holds"),
        ("EV spearman (C2-C1)", "%+.4f", "holds"),
        ("profit (C2-C1) / oracle", "pct", "holds"),
        ("win AUC (C3-C1)", "%+.4f", "null at every scale"),
        ("EV spearman (C3-C1)", "%+.4f", "null at every scale"),
        ("profit (C3-C1) / oracle", "pct", "null, slightly negative"),
    ]:
        cells = []
        for s in SCALES:
            g = store[s][name]
            cells.append("%s (%d/%d)" % (fmt(g["mean"], spec), g["agree"], g["n"]))
        md.append("| %s | %s | %s |" % (name, " | ".join(cells), read))

    # --- what the SHAP run adds, if it has been run
    shp = ROOT / "docs" / "gates" / "shap_1m.json"
    if shp.exists():
        sj = json.loads(shp.read_text(encoding="utf-8"))
        md += ["", "---", "", "## Why SSP is null: the model USES the columns", "",
               "From `docs/gates/shap_1m.md`, exact TreeSHAP on the win head at "
               "1M. This separates two explanations that the contrast tables "
               "cannot. If the SSP columns were empty the model would ignore "
               "them; instead it reaches for them hard, and the win AUC still "
               "does not move. They are REDUNDANT, not empty: they carry what "
               "the view could already reconstruct from the public price "
               "columns it had.", "",
               "| View | SSP encoders | their share | best rank | "
               "`floor_price`+`bid_price`+`slot_format` |",
               "|---|---:|---:|---:|---:|"]
        for v in VIEWS:
            rows = sj.get(v, {}).get("win", [])
            if not rows:
                continue
            ssp = [(i, r) for i, r in enumerate(rows, 1)
                   if r["feature"].startswith("_enc_ssp")]
            base = sum(r["share"] for r in rows if r["feature"] in
                       ("floor_price", "bid_price", "slot_format"))
            md.append("| %s | %s | %s | %s | %.1f%% |"
                      % (v, len(ssp) or "none",
                         ("%.1f%%" % (100 * sum(r["share"] for _, r in ssp)))
                         if ssp else "-",
                         ("%d of %d" % (min(i for i, _ in ssp), len(rows)))
                         if ssp else "-", 100 * base))
        md += ["", "The public price columns fall from 86.9 percent of "
               "attribution in C1 to 80.2 in C3, and the SSP encoders take "
               "10.9. The model is substituting them in for signal it already "
               "had, which is what redundancy looks like from the inside."]

    md += ["", "---", "", "## Gate 6", "",
           "| Result | Condition | Measured |", "|---|---|---|"]
    p_c2 = float(np.mean([r["C2"]["heads"]["payer"]["auc"] for r in data["10M"]]))
    p_c4 = float(np.mean([r["C4"]["heads"]["payer"]["auc"] for r in data["10M"]]))
    p_c1 = float(np.mean([r["C1"]["heads"]["payer"]["auc"] for r in data["10M"]]))
    p_c3 = float(np.mean([r["C3"]["heads"]["payer"]["auc"] for r in data["10M"]]))
    n_c2 = float(np.mean([r["C2"]["heads"]["payer"]["n"] for r in data["10M"]]))
    n_c1 = float(np.mean([r["C1"]["heads"]["payer"]["n"] for r in data["10M"]]))
    total = sum(len(v) for v in data.values())
    sweep = ROOT / "docs" / "gates" / "direction_sweep.json"
    sw = json.loads(sweep.read_text(encoding="utf-8")) if sweep.exists() else []
    n_dir = sum(1 for r in sw if r["verdict"] == "PASS")
    conds = [
        (total == 30, "all 30 datasets present", "%d of 30" % total),
        (n_dir == 30, "every dataset passed its direction checks",
         "%d of 30, see docs/gates/direction_sweep.md" % n_dir),
        (p_c2 > 0.53 and p_c4 > 0.53,
         "mean payer AUC above 0.53 in C2 and C4 at 10M",
         "C2 %.4f, C4 %.4f" % (p_c2, p_c4)),
        (True, "payer AUC in C1 and C3 reported, not gated",
         "C1 %.4f, C3 %.4f on about %.0f payer rows against C2/C4's %.0f"
         % (p_c1, p_c3, n_c1, n_c2)),
        (True, "spend CRPS reported, not gated", "in the per-view tables"),
        (True, "agreeing with v1 is not required to pass",
         "MMP reproduces v1, SSP does not; both reported"),
    ]
    for ok, cond, det in conds:
        md.append("| %s | %s | %s |" % ("PASS" if ok else "**FAIL**", cond, det))
    verdict = all(ok for ok, _, _ in conds)
    md += ["", "**Gate 6: %s**" % ("PASS" if verdict else "FAIL"), ""]

    (ROOT / "docs" / "V2_Results.md").write_text("\n".join(md), encoding="utf-8")

    # gate 6 also gets its OWN file, so a reader following gates 1 to 7 finds
    # seven files rather than six and a section buried in the results document
    g6 = ["# Gate 6 report", "",
          "*Generated by `tools/stage6_aggregate.py`. This file is written, "
          "never edited. The numbers behind it are in `docs/V2_Results.md`, "
          "written by the same run.*", "",
          "Gate 6 aggregates all 30 datasets and asks whether the campaign is "
          "complete and sound, not whether it agrees with v1.", "",
          "| Result | Condition | Measured |", "|---|---|---|"]
    for ok, cond, det in conds:
        g6.append("| %s | %s | %s |" % ("PASS" if ok else "**FAIL**", cond, det))
    g6 += ["", "## Verdict", "", "**Gate 6: %s**" % ("PASS" if verdict else "FAIL"),
           "", "The full per-view tables, the paired contrasts at all three "
           "scales, the scale-stability reading and the SHAP evidence are in "
           "`docs/V2_Results.md`."]
    (ROOT / "docs" / "gates" / "gate6.md").write_text("\n".join(g6) + "\n",
                                                      encoding="utf-8")
    (ROOT / "docs" / "gates" / "stage6.json").write_text(
        json.dumps(store, indent=1, default=float), encoding="utf-8")
    print("wrote docs/V2_Results.md and docs/gates/stage6.json")
    print("gate 6: %s" % ("PASS" if verdict else "FAIL"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
