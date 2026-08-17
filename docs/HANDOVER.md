# T9Sim v2 — handover

*Written at stage 7. What v2 is, how to run it, what it found, and what the next
person needs to know before touching it.*

---

## 1. Run it

```
cd t9v2
venv/Scripts/python.exe -m t9v2.campaign --status          # what exists
venv/Scripts/python.exe -m t9v2.campaign --scale 10M       # all 10 seeds
venv/Scripts/python.exe -m t9v2.campaign --scale 1M --seeds 20250
```

One seed is generated, censored into 4 views, trained, evaluated and written to
`output/runs/<scale>/seed<n>/results.json`. Each seed runs in a fresh subprocess,
because a 10M seed peaks near 5 GB and only a process exit reliably returns it.

| Tool | What it produces |
|---|---|
| `tools/gate_report.py` | `docs/gates/conformance.md`, `docs/provenance.md` |
| `tools/gate2_report.py` … `gate5_report.py` | the per-stage gate reports |
| `tools/direction_sweep.py` | the 5 direction checks on all 30 |
| `tools/stage6_aggregate.py` | `docs/V2_Results.md` |
| `tools/shap_1m.py` | `docs/gates/shap_1m.md` |
| `tools/stage7_release.py --reproduce` | `docs/gates/checksums.md` |
| `tools/make_datasheet.py` | `docs/DATASHEET.md` |

All of these are regenerable. Every file they write says so at the top and none
should be hand-edited: the next regeneration silently discards the edit.

---

## 2. What it found

**MMP is large, and holds at every scale.** Profit gap against the same seed's
oracle: **+8.7% at 100K, +8.1% at 1M, +6.9% at 10M**, 10 seeds out of 10 at each.
Click AUC +0.0615 at 10M, 10/10. The EV-level diagnostic runs C1 0.569 → C2 0.893
against **v1's 0.52 → 0.89**, which is an independent rebuild recovering the same
selection-bias correction.

**SSP is null at every scale**, and v1's headline does not reproduce. v1 published
a win-AUC gain of +0.0137, 10/10 at 10M. v2 gives **+0.0001, 5/10**.

**And the SHAP run says which kind of null it is.** The columns are not empty. C3's
three `_enc_ssp_*` encoders take **10.9 percent** of attribution on the win head
and rank **3rd of 24**, while the public price columns fall from **86.9 percent in
C1 to 80.2 in C3** — almost exactly the share the encoders take. So SSP is
**redundant, not empty**: the model substitutes it in for signal the view could
already reconstruct from `floor_price`, `bid_price` and `slot_format`.

That is consistent with v1's own code, which calls its classifier a "standing
shrink test — the AFT contrast is label-efficiency iff it vanishes here" and
records v8's SSP gain decaying +0.0044 → +0.0002 with scale. The reading is that
SSP's value in v1 was label efficiency, which disappears once there is enough data.

Full numbers: `docs/V2_Results.md`.

---

## 3. Four things that will bite you

**The views are masks, not copies.** C1-C4 share one master, one temporal split and
one test set. Every contrast is therefore PAIRED within a seed, and the aggregation
does this. Comparing 10 C2 values against 10 C1 values instead widens the 10M
profit interval about fourfold and makes it span zero, turning a 10/10 result into
"no significant difference" purely by discarding the pairing.

**`ev_ratio` means two different things in v1 and v2.** v1's is
`mean(ev_hat)/mean(ev_truth)`, an EV-BIAS diagnostic, labelled as such in its own
source, whose oracle is 1.0 by construction. v2's is a value-capture share whose
oracle is about 0.90, because its denominator includes rows no policy should buy.
v2 reports v1's quantity separately as `ev.ratio`. Do not compare them by name.

**Ratios against C1's profit explode at 100K.** C1's profit there is within noise of
zero, with individual seeds at −1, 0, 5 and 41 dollars. An unguarded C2/C1 − 1
printed −852,694.8% for the headline MMP contrast. The aggregation floors it and
reads n/a below the floor; for cross-scale reading, divide the paired difference by
the same seed's oracle profit instead, which is large and positive in every seed.

**Do not re-gate the click-AUC floor at 100K.** It is judged at 1M and reported at
100K, where C1 (0.5414) and C3 (0.5435) sit below 0.55. That gap is the censoring
working as designed: those two views see funnel labels on won rows only, so their
click head trains on about 22,000 selected rows, while C2 and C4 see every row and
reach 0.65 at the same scale. A floor only the uncensored views can meet
re-measures the censoring rather than the model, and the ablation already measures
the censoring on purpose. Install AUC has had this treatment from the start.

---

## 4. What was never built

Eight things are deferred by name and were verified absent at gate 7, not merely
recalled: the `benchmarks/` folder; deep models and serving infrastructure; the
all/won/lost-rows scorecard; diagrams generated from settings; the sequence dataset
builder, history models and schema mapping; bid shading, budget pacing, other
auction formats and hyperparameter search; method comparison; and the provenance
pass on the A3, A4 and A5 marginals.

`ev_spearman` was on that list and came off it on 16 August.

---

## 5. What must never be touched

v2 may disagree with v1 on anything it produces — its data, its results, its
economics, its headline conclusion. What it may never do is change something
already published or frozen.

| Artifact | Where |
|---|---|
| Zenodo deposit | reserved DOI 10.5281/zenodo.21533031 |
| Release repo tag | `v1.0.1` in github.com/QUMLKY/T9-simulator |
| v1 golden fingerprint | `0xdf0ac3e18624cf2b` |
| v1 result documents | `docs/v1/v10_Training_Results.md`, `docs/Method_Benchmark_10M_Results_13Jul2026.md`, the submitted paper |
| v1 environment and parquets | `t9_sim/`, seeds 90213-90222 |

The working repository is never published: its history contains personal files.

---

## 6. State at handover

| | |
|---|---|
| Datasets | 30 of 30, 19.8 GB, all hashed and reproducible |
| Gates | 1 to 7, all PASS |
| Tests | 68 passed |
| Settings hash | `49e9b32a49d40b8ac559f07c6e4ec934` |
| Design fingerprint | `caeb2b87716e00e5` |
| Direction checks | 30 of 30, 150 checks, no failures |
| Reproduction | one seed per scale regenerated, byte-identical at all three |

The seven stages are complete. What remains is not build work: the SSP divergence
from v1 is the most interesting thing the rebuild produced, and it needs writing up
as a finding with its mechanism, not filed as a gate failure.
