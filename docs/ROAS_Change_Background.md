# The ROAS target changes from 1.0 to 3.0 — background and what still needs doing

*Written 24 August 2026 on branch `t9-v2.2-roas3`, for a session that will carry the change
through the code and the remaining documents. Nothing in this file is a proposal. The decision is
Ken's and it is made.*

## The decision, and why it is a correction rather than a choice

`roas_target` has been 1.0 since the build. The gate places a row when `ev / b >= target` at the
rung the argmax already chose.

**At 1.0 the gate never fires.** Measured across ten seeds at 10M, the learned bidder places on
**99.99 per cent** of test rows in C1 and **100.00 per cent** in C3. Every reported economic figure
in v2.2 therefore describes a bidder that bids on every auction it sees.

No advertiser behaves that way. A buyer with no return requirement is not a buyer any DSP would
run, so the ROAS 1.0 results describe a policy that does not exist in the market. A threefold return makes the gate
bind, and it is sourceable rather than picked: the standard rule is break-even ROAS = 1 divided by
the profit margin, so 3:1 is the requirement of a buyer working to roughly a 33 per cent margin.
Note also that the realised return runs well above the target, about 5.4 to 5.6 times across the
views, because the gate tests expected value at the chosen bid while a win also needs the bid to
clear. **The change fixes a defect in
the default; it is not a search for a better number.**

Ken's ruling, 24 August: *"ROAS = 1 resulted the bidder bidding on ALL auctions, which is
absolutely not how advertisers behave. So change ROAS is fixing a bug."*

## What the change does and does not touch

**It cannot touch any prediction metric.** The gate is applied after training and after the argmax,
so click, install and payer AUC, spend CRPS, EV level, `rmse_log`, `bias_log` and win AUC are all
identical at every target. Only the bidder block moves.

**One exception to be aware of:** `ece_win` is scored on placed rows only, deliberately, because a
declined row carries a probability the bidder never bet on. It therefore does move with the target.
It has already been removed from the reported tables for an unrelated reason, that it is the one
Tier 2 metric the sigma placeholder can move.

**It does not invalidate any dataset.** `training.yaml` is excluded from the design fingerprint by
design, `generate.py:53-56`, because it changes how a dataset is scored and never what it contains.
No regeneration is needed and the thirty parquets stay valid.

## What the change is worth

Profit contrasts, ten seeds at 10M, paired within seed.

| Contrast | T = 1.0 | T = 1.5 | T = 2.0 | T = 3.0 |
|---|---:|---:|---:|---:|
| C2 − C1, MMP | +63.9% 10/10 | +71.0% 10/10 | +92.0% 10/10 | **+221.1%** 10/10 |
| C3 − C1, SSP alone | +1.5% **8/10** | +8.0% 10/10 | +13.5% 10/10 | **+15.4%** 10/10 |
| C4 − C2, SSP on top of MMP | +5.8% 9/10 | +10.3% 10/10 | +17.4% 10/10 | **+17.3%** 10/10 |

**The SSP null exists only at 1.0.** It is the single unsupported cell in the table, and it sits at
the one target where the gate is inert. At every binding target the contrast is unanimous.

**The mechanism reverses, and this is the sentence to keep.** A better price model bids lower.
At 1.0 that only wins fewer auctions, so SSP looks worthless: C3 wins **29,763 fewer** than C1,
all ten seeds. Under a binding gate the same lower bid raises `ev / b` and gets more rows past the
gate, so C3 wins **23,848 more**, also all ten seeds. Every sign in the bidder block flips.

## How the numbers were produced, and the check that licenses them

`tools/roas_sweep.py`, new on this branch. It re-scores from files and refits nothing.

Two things it had to get right:

1. **The per-row eval file describes the CLASSIFIER's bidder.** `evalfile.py:65` builds its curve
   with `t2.win_curve`. Its `bid_recommended` is the classifier's argmax, and its
   `profit_at_recommended` is EXPECTED profit, not realised. The price head's curve is therefore
   rebuilt analytically, which is exact rather than approximate because that curve is the win rule
   written down: `P(win | b) = Phi((log b − log m_hat) / sigma) . 1[b >= floor]`.
2. **`floor_price` and `ev_truth` are not in the eval file** and are joined from the master by row
   index.

The classifier is not scored at all. It is retired from the reporting, so the sweep covers the
price head and the oracle only.

**The check.** At target 1.0 the tool must reproduce the run's recorded economics exactly. It does,
wins and profit, in all four views on all ten seeds — forty checks. The tool
refuses to write results if that fails. Nothing at 1.5, 2.0 or 3.0 is trustworthy without it.

## What is done on this branch

| File | State |
|---|---|
| `tools/roas_sweep.py` | new, the price head and the oracle, four targets |
| `docs/roas_sweep_10M.{md,json}` | new, all four targets, all four views, n = 10 |
| `docs/V2.2_Results_AFT.md` + docx | re-scored to 3.0, script-written, complete |
| `docs/V2.2_Analysis_AFT_V2.md` | renamed from `V2.2_Results_AFT_V2.md`, docx now `Analysis V2.2 AFT V2.docx` |
| `docs/Archive/` | `V2.2_Analysis.md`, `V2.2_Analysis_Brief.md`, `V2.2_Analysis_Brief_AFT.md` and their docx, superseded |

## What was done on 25 August, and what was not

**The reported documents are at 3.0 and consistent with each other.** `V2.2_Results_AFT.md`,
`Training_Results v2.2 AFT.docx` and `Training_Results v2.2 AFT + CI.docx` now agree on every bidder
figure. All thirty published numbers in the results table were checked against the overlay and reproduce
exactly. `config/training.yaml` reads 3.0 and its note carries the argument above.

**The runs were NOT re-scored, and this is the decision of record.** `tools/rescore.py` re-scores in
place and reproduces a recorded run exactly at 100K in ten seconds. At 10M it holds about eight gigabytes
for one seed and had not finished one seed in twenty minutes on a sixteen gigabyte laptop, which is what
ended the earlier attempt. Ten seeds would have been three hours or more on the day before submission, and
it would not have changed a single reported number.

**So the reporting scripts re-score at read time instead.** `results_report.py --roas 3` and
`results_v1_layout.py --roas 3` overlay the bidder rows from `docs/roas_sweep_10M.json` and leave every
other row coming from the frozen runs. `make_report.py` passes the flag automatically, reading the target
from the settings so it cannot drift. Without that patch a plain `make_report` run rebuilt both AFT
documents at target 1.0 on top of the 3.0 ones, with nothing to notice.

**What is still at 1.0, on purpose.** The thirty `results.json`, the eval parquets and the gate files
`agreement_{100K,1M,10M}.md`, `gate4.md` and `gate5.md`. They are the record of what was generated, they
are labelled with their own target, and a reader who wants the reported economics has three documents that
carry them. `roas_sweep_10M.md` keeps all four targets, which is the point of it.

**What was not attempted.** `Dissertation_S7_Results_1250w.md` is untouched. It is not merely at the old
target, it is a v2-era chapter. It reports the SSP contrast as negative, carries the dead "five sixths of
the gap is pricing" claim and answers RQ3, which no longer exists. Fixing it is a rewrite against the
current results, not a target change, and it was left rather than half-corrected.

## What must not be touched

`V1_vs_V2_Comparison.md`, `V1_vs_V2_Improvements.md`, `T9Sim_v2_Stage1to5_Record.md`,
`PROJECT_LOG.md` and `Dissertation_Outline.md` record what v2 and v2.2 reported at target 1.0.
Editing them to 3.0 falsifies the trail. The ROAS 1.0 numbers remain correct statements about an
unconstrained bidder and should stay where they are, labelled as such.

## One thing to decide, not assume

The oracle at 3.0 bids on **7.8 per cent** of auctions and earns **149.42** per thousand wins,
against C4's 30.5 per cent and 40.02. The perfect bidder does not run a better version of the same
policy; it runs a quarter of the volume at nearly four times the margin. Whether the ROAS-3
document reports the oracle as a ceiling, given it is gated by the same target, is a reporting
decision nobody has made yet.
