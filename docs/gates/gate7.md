# Gate 7 report — the release gate

*Hand-assembled from generated evidence, and every row below names the artifact it
is read from. Unlike gates 1 to 6 this one is not itself generated, because one of
its conditions is a judgement about what was deferred, and no settings file holds
that.*

## What stage 7 built

| Piece | Where |
|---|---|
| the checksum manifest over the 30 datasets | `tools/stage7_release.py` → `docs/gates/checksums.md` |
| golden fingerprints, settings and design | same, and `generate.fingerprint` |
| the reproduction proof | `tools/stage7_release.py --reproduce` |
| the datasheet | `tools/make_datasheet.py` → `docs/DATASHEET.md` |
| the handover | `docs/HANDOVER.md` |

## The two fingerprints, and why there are two

| | |
|---|---|
| settings hash | `49e9b32a49d40b8ac559f07c6e4ec934` |
| design fingerprint | `caeb2b87716e00e5` |
| RNG scheme | `t9v2-rng-1` |

The settings hash covers the configuration that shapes the data. The design
fingerprint additionally folds in the bytes of the 7 modules that generate it. Two
hashes rather than one because each misses what the other catches: a settings hash
alone would not notice a changed law, and a code hash alone would not notice a
changed constant. The design fingerprint is the one written into every parquet's
manifest, and `campaign.run_one` refuses to reuse a parquet whose fingerprint does
not match the current design.

**The settings hash excludes prose, and it was changed to (17 Aug 2026).** It first
read `423de1651e374e7e8be9a2318dd62edd`. Editing one explanatory `note` in
`validation.yaml` moved it to `7334911afa1eafb5` — a documentation change that
altered no byte of any dataset. A fingerprint that has to be re-baselined whenever
a comment improves teaches the reader to re-baseline it, which is the exact habit
it exists to prevent. `settings_hash` now strips `note`, `doc`, `comment` and
`description` before hashing, and the value above is that stable hash. `route` and
`source` are deliberately KEPT: a provenance change is a substantive claim about
the data even though it shapes none of it.

Verified both ways rather than asserted: rewriting a note leaves the hash at
`49e9b32a49d40b8a`, and changing a single float moves it to `588b0b1104154aad`.

## Pass conditions

| Result | Condition | Evidence |
|---|---|---|
| PASS | tests green | 68 passed, 39.75s, `pytest -q` |
| PASS | re-running stages 2 and 3 reproduces their fingerprints | one seed regenerated at each scale into a temp path: 100K, 1M and 10M all **byte-identical** by sha256 to the shipped file. Stage 3 reproduces with it, its censored column sets and visible-cell counts recorded per view in `checksums.json` |
| PASS | the deferred list matches what Ken was told | checked against the plan's section 6 list, below |
| PASS | the checksum manifest holds one entry per dataset, and re-hashing reproduces it | 30 of 30 entries, 19.8 GB, and the reproduction step re-hashed and matched rather than trusting the recorded value |

## The reproduction, in detail

This is the only condition that tests determinism rather than asserting it.

| Scale | Seed | Result | Seconds |
|---|---:|---|---:|
| 100K | 20250 | MATCH | 3 |
| 1M | 20250 | MATCH | 19 |
| 10M | 20250 | MATCH | 191 |

A match here means the generator, the named-stream RNG scheme and the settings
together reproduce the shipped bytes exactly, months after the fact and in a
different process. It is what makes the manifest worth having: a list of hashes
whose files could not be regenerated would record what happened, not what can be
checked.

## The deferred list, verified rather than recalled

The plan defers 8 things by name. Each was checked to be genuinely absent, not
quietly half-built, because a deferred item that turns out to exist in part is
worse than one that does not exist at all.

| Deferred | Verified |
|---|---|
| the `benchmarks/` folder | absent |
| deep models and serving infrastructure | no `serving/` module, no deep-model code |
| the all-rows / won-rows / lost-rows scorecard | 0 references |
| diagrams generated from the settings file | no `diagrams/` module in v2 |
| the sequence dataset builder, history models, schema mapping | no `sequence/` module |
| bid shading, budget pacing, other auction formats, hyperparameter search | 0 references to second-price or to hyperparameter search. The `shade` and `pacing` hits are RIVAL-side generator features (the rival pool's LR3 pacing and the rival bid formula's shade constant), which are in scope; our own bidder does neither |
| method comparison | 0 references; v2 builds the XGBoost stack only |
| the provenance pass on the A3, A4 and A5 marginals | still outstanding, and named in the datasheet as inferred |

One item moved OFF the deferred list during stage 5, and it is recorded so the
list stays honest: `ev_spearman` was deferred at stage 5 and added on Ken's
instruction on 16 August. The plan's "good to have, deferred" section was rewritten
to "added at stage 5" rather than left standing.

## One number in the plan is stale

The plan's datasheet section says the provenance sweep leaves "47 of the 83
settings leaves" on the `inferred` route. The register now reads **108 of 156**.
The settings grew as stages 2 to 4 added laws, and the sentence was written before
they did. The datasheet is generated from the live settings, so it carries the
correct 108; this note exists so the discrepancy is not read as the datasheet
being wrong.

## Verdict

**Gate 7: PASS**

30 datasets, reproducible, hashed, documented, with the deferred list verified and
the inferred values enumerated.
