# -*- coding: utf-8 -*-
"""The archetype tilt sensitivity sweep: 4 tau arms x 3 seeds at 100K.

    python tools/tilt_sweep.py

Open item O3 3b invented the 3 archetype tilt strengths and declared them,
because no outside dataset can supply them: archetype is T9's own latent, so
nothing external measures how strongly it should tilt os, device or day of week.
Every table is raked, so its marginal survives whatever tau is, which means NO
GATE IN THE BUILD can tell a right tau from a wrong one. These are the only
invented numbers in the design with no test attached.

This sweep is the mitigation, and its limit is stated rather than glossed: it
establishes whether the headline contrasts DEPEND on tau. It cannot establish
that 0.13 is right, and nothing can.

Run BEFORE the campaign, so it can still change the decision. Reported whatever
it shows: if a headline contrast moves outside its interval across the arms, that
is a finding, not a failure to fix quietly.
"""
import io
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np                                               # noqa: E402

from t9v2 import generate as G                                   # noqa: E402
from t9v2.core.config import load                                # noqa: E402
from t9v2.train.runner import run_seed                           # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = HERE.parents[1]
OUT = HERE.parent / "docs" / "gates" / "tilt_sweep.json"

SEEDS = [20250, 20251, 20252]
SCALE = "100K"


def contrasts(r):
    """The headline numbers the sweep is asking about."""
    h = lambda v, k, m="auc": r[v]["heads"][k][m]
    e = lambda v, k: r[v]["economics"]["learned"][k]
    return {
        "click_auc_C1": h("C1", "click"), "click_auc_C2": h("C2", "click"),
        "win_auc_C1": h("C1", "win"), "win_auc_C3": h("C3", "win"),
        "mmp_click_auc": h("C2", "click") - h("C1", "click"),
        "ssp_win_auc": h("C3", "win") - h("C1", "win"),
        "mmp_profit": e("C2", "profit") - e("C1", "profit"),
        "ssp_profit": e("C3", "profit") - e("C1", "profit"),
        "mmp_ev_ratio": e("C2", "ev_ratio") - e("C1", "ev_ratio"),
        "ssp_ev_ratio": e("C3", "ev_ratio") - e("C1", "ev_ratio"),
    }


def main():
    s = load("default")
    arms = s.raw["meta"]["tilt_sensitivity_sweep"]
    tau0 = s.raw["tables"]["A3_os_given_archetype"]["tilt"]["tau"]["value"]
    rows, t0 = [], time.time()

    print("tilt sweep: %d arms x %d seeds at %s, base tau = %g"
          % (len(arms), len(SEEDS), SCALE, tau0))
    for arm in arms:
        for seed in SEEDS:
            m = G.frame(SCALE, seed=seed, tau_scale=float(arm))
            r = run_seed(m, s, seed=0, quiet=True)
            c = contrasts(r)
            c.update(tau_scale=float(arm), tau=float(arm) * tau0, seed=seed)
            rows.append(c)
            print("  tau x%-4g seed %d   click C1 %.4f  MMP click %+.4f  "
                  "SSP win %+.5f  MMP profit %+.0f"
                  % (arm, seed, c["click_auc_C1"], c["mmp_click_auc"],
                     c["ssp_win_auc"], c["mmp_profit"]))

    # the null arm has to reproduce a generator with the tilt machinery gone
    a = G.frame(SCALE, seed=SEEDS[0], n_rows=20000, tau_scale=0.0)
    b = G.frame(SCALE, seed=SEEDS[0], n_rows=20000, untilted=True)
    null_ok = bool(a.equals(b))

    keys = [k for k in rows[0] if k not in ("tau_scale", "tau", "seed")]
    summary = {}
    for arm in arms:
        sub = [r for r in rows if r["tau_scale"] == float(arm)]
        summary[str(arm)] = {k: {"mean": float(np.mean([r[k] for r in sub])),
                                 "sd": float(np.std([r[k] for r in sub], ddof=1))}
                             for k in keys}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"scale": SCALE, "seeds": SEEDS, "base_tau": tau0, "arms": arms,
         "tau0_reproduces_untilted_bit_for_bit": null_ok,
         "rows": rows, "summary": summary}, indent=1, default=float), encoding="utf-8")

    print("\n%-22s %s" % ("contrast", "  ".join("tau x%-6g" % a for a in arms)))
    for k in ("mmp_click_auc", "ssp_win_auc", "mmp_ev_ratio", "ssp_ev_ratio"):
        print("%-22s %s" % (k, "  ".join("%+9.5f" % summary[str(a)][k]["mean"]
                                         for a in arms)))
    print("\ntau = 0 reproduces the untilted generator bit for bit: %s" % null_ok)
    print("wrote %s  (%.0fs)" % (OUT.relative_to(ROOT), time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
