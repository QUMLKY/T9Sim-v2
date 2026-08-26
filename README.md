# T9Sim v2.2

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

Ten seeds at 10 million rows each. Every contrast is paired within a seed. The bidder is held to a
ROAS target of 3, which is what a buyer working to roughly a 33 per cent margin requires.

| | profit against C1 | interval | seeds agreeing |
|---|---:|---|---:|
| **MMP** (C2−C1) | **+221.1%** | [+155.4%, +286.8%] | 10/10 |
| **SSP** (C3−C1) | **+15.4%** | [+13.0%, +17.8%] | 10/10 |
| **SSP on top of MMP** (C4−C2) | **+17.3%** | [+15.3%, +19.3%] | 10/10 |
| **all three** (C4−C1) | **+274.8%** | [+201.7%, +347.9%] | 10/10 |

**MMP is decisive, and it works by correcting a valuation.** C1 believes the inventory is worth
about half what it is, an EV level of 0.532 against the truth. C2 believes 0.897. The gain deepens
down the funnel, because each stage depends on the one before it clearing, so the survivors grow
scarcer and harder to identify at every step. Click AUC gains 0.0220, install 0.0377 and payer
0.0948, every seed agreeing.

**SSP is worth less, and it works through price rather than value.** It leaves all 4 funnel heads
untouched, which is what it should do, since supply-side data carries prices and not user outcomes.
It moves the price head instead. `rmse_log` falls from 1.315 to 0.913 and `bias_log` from +0.455 to
−0.002, both on all 10 seeds. C1 over-predicts the clearing price, so it bids too much on rows it
cannot win and too little on the ones it can. C3 corrects both.

**The two layers compound and neither gets in the other's way.** MMP alone multiplies profit by
3.211 and SSP alone by 1.154. If the two were independent their product would be 3.678, and the
measured figure is 3.748, a ratio of 1.017 [0.998, 1.035]. The interval touches 1, so the small
excess is not established. The absence of interference is.

---

### Two bidders, and why the design is the contribution

Tier 2 runs a win classifier beside an interval-censored AFT price head, each constrained so it can
move through 1 channel only.

- the **classifier**'s label `won` is visible in all 4 views, so its C3−C1 contrast moves only
  through **features**
- the **price head**'s features are identical in all 4 views, every SSP encoder barred from it by
  name, so its contrast moves only through **labels**

A disagreement between them locates a mechanism rather than cancelling a claim. The AFT price head
is the reported bidder. Both are built and both are kept, `--bidders both|aft`.

**This supersedes v2, and 2 of v2's claims are dead.** v2 ran 1 head, found nothing on the SSP
layer, and reported it as a flat null. v2.2 finds that null is real on the feature channel and
large on the label channel, so it was a channel result reported as a flat one. v2 also held that
five sixths of the gap to the oracle is pricing. That was a property of the classifier's win curve
rather than of the study, and under the price head's curve the same split reverses.

**A v2 number must never be placed beside a v2.2 number.** The floor fix, the units fix, the wider
bid ladder, the added column and the second head each move every figure, so the 2 together compare
2 simulators rather than 2 data layers.

Full numbers: [`docs/V2.2_Results_AFT.md`](docs/V2.2_Results_AFT.md), and the analysis beside it.


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

**To rebuild the results documents, run one command rather than the pieces:**

```
venv/Scripts/python.exe tools/make_report.py            # everything
venv/Scripts/python.exe tools/make_report.py --check    # say what would run
venv/Scripts/python.exe tools/make_report.py --no-docx  # markdown only
```

The chain has an order that matters — `results_report.py` reads what
`tier1_allrows.py` writes, and refuses to run without it — and a flag that is off
by default and changes the finished document. `make_report.py` holds both. Run
the steps by hand only after reading why it exists.

| Tool | Produces |
|---|---|
| **`tools/make_report.py`** | **all three results documents. This is the entry point** |
| &nbsp;&nbsp;↳ `tools/tier1_allrows.py` | `docs/gates/tier1_allrows.{json,md}` |
| &nbsp;&nbsp;↳ `tools/results_report.py` | `docs/V2.2_Results.md`, `docs/gates/agreement_*.md`, `gate6.md`, `stage6.json` |
| &nbsp;&nbsp;↳ `tools/results_v1_layout.py` | `Training_Results v2.2 + CI.docx` |
| &nbsp;&nbsp;↳ `../tools/safe_docx_export.py` | `Training_Results v2.2.docx`, `V2.2 Analysis.docx` |
| `tools/gate_report.py` | `docs/gates/gate1.md`, `docs/provenance.md`, `docs/SPEC_GAPS.md` |
| `tools/gate2_report.py` … `gate5_report.py` | the per-stage gate reports |
| `tools/direction_sweep.py` | the 5 direction checks across all 30 datasets |
| `tools/shap_1m.py --scale 10M` | `docs/gates/shap_attribution_<scale>.md` |
| `tools/stage7_release.py --reproduce` | `docs/gates/checksums.md` |
| `tools/make_datasheet.py` | `docs/DATASHEET.md` |

The Word documents need two packages the analysis does not: `pip install -e ".[docx]"`.
Without them the markdown still generates and the docx steps are skipped.
`pypandoc_binary` bundles pandoc, so nothing outside the venv is required.

---

## The data

30 datasets: 10 seeds (20250–20259) at each of 100K, 1M and 10M impressions. **19.8 GB**, so they
are not in this repository. Every one carries a sha256 in
[`docs/gates/checksums.md`](docs/gates/checksums.md), and every one regenerates from this source.

Reproducibility is **tested, not asserted**: one seed at each scale is regenerated into a temporary
path and compared byte for byte against the shipped file. All three match. See
[`docs/gates/gate7.md`](docs/gates/gate7.md).

| | |
|---|---|
| design fingerprint | `1828f8a2db558222` |
| settings hash, as the runs were generated | `49e9b32a49d40b8ac559f07c6e4ec934` |
| settings hash, current | `aafcc28319b73158561f5ce9444122664da78d7e6054bd6776cfa11aa1e9f0b7` |
| RNG scheme | `t9v2-rng-1` |

Two hashes, because each misses what the other catches. A settings hash alone would not notice a
changed law, and a code hash alone would not notice a changed constant. The settings hash
deliberately excludes prose, so improving a comment does not force a re-baseline.

**The 2 settings hashes differ, and that is the distinction working rather than a fault.** The ROAS
target moved from 1.0 to 3.0 on 25 August 2026, after the 30 datasets were generated. `training.yaml`
is inside the settings hash and deliberately outside the design fingerprint, because it changes how
a dataset is SCORED and never what it contains. So the settings hash moved, the design fingerprint
did not, and every dataset on disk is still the one this source generates. The fingerprint above was
recomputed from the current settings on 26 August 2026 and matches all 30 manifests.

**One caveat on cross-machine reproduction.** The random draws are portable — streams are keyed by
blake2b, not by a global seed. The *file bytes* additionally depend on the writer, so
`constraints.txt` pins every library including `pyarrow`. Reproduce inside those pins and the
comparison is byte-for-byte; outside them, expect identical data content in a file that may not
hash the same.

---

## Read it in this order

| Document | What it is |
|---|---|
| [`docs/V2.2_Results_AFT.md`](docs/V2.2_Results_AFT.md) | **the results** — per-view tables at 10M, at a ROAS target of 3 |
| [`docs/V2.2_Analysis_AFT_V2.md`](docs/V2.2_Analysis_AFT_V2.md) | what the 2 layers are worth, with intervals and agreement counts |
| [`docs/ROAS_Change_Background.md`](docs/ROAS_Change_Background.md) | why the target is 3, and why the runs on disk still say 1 |
| [`docs/roas_sweep_10M.md`](docs/roas_sweep_10M.md) | the same contrasts at 4 targets, which is what licenses the re-scoring |
| [`docs/DATASHEET.md`](docs/DATASHEET.md) | what the data is, what it must **not** be used for, and all 108 inferred values |
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

30 of 30 datasets generated, trained, direction-checked and hashed, with 0 failures. 218 tests green
and 12 skipped.

Gates 1 to 6 are v2.2's and PASS. **Gate 7 is v2's and has not been re-run**, so its settings hash
and checksums describe the v2 build rather than this one. It is kept because its reproduction
evidence still stands, and it is named here rather than quietly carried.

**The 30 `results.json` hold economics at a ROAS target of 1.0.** The reported documents re-score
their bidder rows to 3.0 at read time from `docs/roas_sweep_10M.json`, whose target-1.0 column
reproduces each run's recorded economics exactly, wins and profit, in all 4 views on all 10 seeds.
Re-scoring the runs in place needs about 8 GB a seed at 10M, so the frozen runs are left exactly as
generated and the re-scoring is auditable rather than silent. `--roas 3` on `results_report.py` and
`results_v1_layout.py` does the overlay, and `make_report.py` passes it automatically.

## Licence

Code MIT ([`LICENSE`](LICENSE)). Data CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).

## Citing

See [`CITATION.cff`](CITATION.cff).
