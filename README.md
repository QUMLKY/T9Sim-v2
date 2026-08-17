# T9Sim v2

A synthetic auction simulator for a controlled **ablation**: how much is each integrated AdTech
data layer actually worth to a bidder that has to predict conversions and win auctions?

One synthetic ground truth is generated, then censored four ways. The same models are trained on
each view. What differs between them is only what each is allowed to see.

| View | Layers | What it adds |
|---|---|---|
| **C1** | DSP only | the baseline — funnel outcomes on WON rows only |
| **C2** | DSP + MMP | **rows** — funnel outcomes on every row |
| **C3** | DSP + SSP | **columns** — `bid_density`, `winning_price` |
| **C4** | all three | both |

That asymmetry is the design: **MMP adds rows, SSP adds columns.** C1→C2 is a selection-bias
correction; C1→C3 is a price-visibility gain. The four views are *masks over one master*, sharing
rows, order and temporal split, which is what makes this an ablation rather than four experiments.

---

## What it found

**MMP is large, and holds at every scale.** Profit against the same seed's oracle:

| | 100K | 1M | 10M |
|---|---:|---:|---:|
| MMP (C2−C1) | +8.7% | +8.1% | +6.9% |
| SSP (C3−C1) | −0.6% | −0.5% | −0.2% |

All MMP rows agree in 10 seeds out of 10. Click AUC +0.0615 at 10M, also 10/10.

**SSP is null at every scale** — and the SHAP run says which *kind* of null. The SSP columns are
not empty: the model reaches for them hard, taking 10.9% of attribution on the win head at rank 3
of 24. Meanwhile the public price columns fall from 86.9% of attribution in C1 to 80.2% in C3 —
almost exactly the share the SSP encoders take. They are **redundant, not empty**. The model
substitutes them in for signal the view could already reconstruct from `floor_price`, `bid_price`
and `slot_format`.

**Relationship to v1.** v2 shares no code with v1 and was built without sight of v1's results, so
agreement is a reproduction rather than a fit. MMP reproduces: the EV-level diagnostic runs
C1 0.569 → C2 0.893 against v1's 0.52 → 0.89. **SSP does not.** v1 published a win-AUC gain of
+0.0137 (10/10 at 10M); v2 gives +0.0001 (5/10). That divergence is the most interesting thing the
rebuild produced, and it is consistent with v1's own code, which calls its classifier a "standing
shrink test — the AFT contrast is label-efficiency iff it vanishes here."

Full numbers: [`docs/V2_Results.md`](docs/V2_Results.md).

---

## Run it

```bash
pip install -e . -c constraints.txt

python -m t9v2.campaign --status                    # what exists
python -m t9v2.campaign --scale 1M --seeds 20250    # one seed end to end
python -m t9v2.campaign --scale 10M                 # all 10 seeds, ~3 hours
```

One seed is generated, censored into four views, trained, evaluated, and written to
`output/runs/<scale>/seed<n>/results.json`. Each seed runs in a fresh subprocess, because a 10M
seed peaks near 5 GB and only a process exit reliably returns it.

**Every report in this repository is generated**, never hand-written. Each says so at its top, and
hand-editing one means losing the edit at the next regeneration.

| Tool | Produces |
|---|---|
| `tools/gate_report.py` | `docs/gates/gate1.md`, `docs/provenance.md`, `docs/SPEC_GAPS.md` |
| `tools/gate2_report.py` … `gate5_report.py` | the per-stage gate reports |
| `tools/direction_sweep.py` | the 5 direction checks across all 30 datasets |
| `tools/stage6_aggregate.py` | `docs/V2_Results.md`, `docs/gates/gate6.md` |
| `tools/shap_1m.py` | `docs/gates/shap_1m.md` |
| `tools/stage7_release.py --reproduce` | `docs/gates/checksums.md` |
| `tools/make_datasheet.py` | `docs/DATASHEET.md` |

---

## The data

30 datasets: 10 seeds (20250–20259) at each of 100K, 1M and 10M impressions. **18.5 GB**, so they
are not in this repository. Every one carries a sha256 in
[`docs/gates/checksums.md`](docs/gates/checksums.md), and every one regenerates from this source.

Reproducibility is **tested, not asserted**: one seed at each scale is regenerated into a temporary
path and compared byte for byte against the shipped file. All three match. See
[`docs/gates/gate7.md`](docs/gates/gate7.md).

| | |
|---|---|
| settings hash | `49e9b32a49d40b8ac559f07c6e4ec934` |
| design fingerprint | `caeb2b87716e00e5` |
| RNG scheme | `t9v2-rng-1` |

Two hashes, because each misses what the other catches: a settings hash alone would not notice a
changed law, and a code hash alone would not notice a changed constant. The settings hash
deliberately excludes prose, so improving a comment does not force a re-baseline.

**One caveat on cross-machine reproduction.** The random draws are portable — streams are keyed by
blake2b, not by a global seed. The *file bytes* additionally depend on the writer, so
`constraints.txt` pins every library including `pyarrow`. Reproduce inside those pins and the
comparison is byte-for-byte; outside them, expect identical data content in a file that may not
hash the same.

---

## Read it in this order

| Document | What it is |
|---|---|
| [`docs/V2_Results.md`](docs/V2_Results.md) | the results — per-view tables and paired contrasts at all three scales |
| [`docs/DATASHEET.md`](docs/DATASHEET.md) | what the data is, what it must **not** be used for, and all 108 inferred values |
| [`docs/HANDOVER.md`](docs/HANDOVER.md) | how to run it, what it found, four things that will bite you |
| [`docs/provenance.md`](docs/provenance.md) | every setting with a route and a source |
| [`docs/gates/`](docs/gates/) | gates 1–7, the direction sweep, SHAP, checksums, the τ and ρ sweeps |
| [`design/`](design/) | the specification, plan, symbol plan and node register this was built **from** |

---

## Three things a reviewer should know up front

**Every contrast is paired within a seed.** The four views share one master, one split and one test
set, so differencing inside a seed removes the variation common to both sides. Comparing 10 C2
values against 10 C1 values instead widens the 10M profit interval about fourfold and makes it span
zero — turning a 10/10 result into "no significant difference" purely by discarding the pairing.

**`ev_ratio` means different things in v1 and v2.** v1's is `mean(ev_hat)/mean(ev_truth)`, an
EV-*bias* diagnostic whose oracle is 1.0 by construction. v2's is a value-capture share whose oracle
is about 0.90, because its denominator includes rows no policy should buy. v2 reports v1's quantity
separately as `ev.ratio`. Do not compare them by name.

**Gate 4's click-AUC floor is judged at 1M, and reported at 100K.** At 100K, C1 reads 0.5414 and C3
0.5435 against a 0.55 floor while C2 and C4 pass — but that is the censoring, not the model. C1 and
C3 see funnel labels on won rows only, so their click head trains on ~22,000 selected rows carrying
a few hundred clicks. A floor only the uncensored views can meet at a given scale re-measures the
censoring, which the ablation already measures on purpose. At 1M all four clear it with no change to
the model (C1 0.5706, C3 0.5731). Install AUC has the same treatment for the same reason. The 100K
numbers are still reported in [`docs/gates/gate4.md`](docs/gates/gate4.md).

---

## Status

Seven stages complete. Gates 1 to 7 all PASS. 77 tests green.
30 of 30 datasets generated, trained, direction-checked and hashed.

## Licence

Code MIT ([`LICENSE`](LICENSE)). Data CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).

## Citing

See [`CITATION.cff`](CITATION.cff).
