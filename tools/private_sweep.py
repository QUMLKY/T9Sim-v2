# -*- coding: utf-8 -*-
"""Sweep the private-rival amplitude: one multiplier on beta_R and the w_k ranges.

    python tools/private_sweep.py

WHY. The competing bid's variance decomposes 82.7 percent to `base_e`, the shared
PUBLIC price core, and only 15.1 percent to all six rival latents combined, with
`slot_format`'s eCPM target alone accounting for 47 percent of the total. So the
quantity the SSP layer exists to reveal is a small share of the thing it reveals,
which caps the C3-minus-C1 contrast before any model is fitted. This sweep raises
the private amplitude and asks whether the contrast follows.

x1 reproduces the current build exactly, so it is a true null arm.

TWO THINGS THAT MUST HOLD or the arms are not comparable, both learned by getting
them wrong on 16 August 2026:

  k_global is RE-SOLVED per arm. Widening the rival bid distribution raises
  E[max], so our win rate falls below its 0.30 target. Without the re-solve the
  arms differ in win rate as well as in private structure and nothing can be
  attributed.

  The re-solve reads the SAME calibration layer the evaluation will,
  `calibrated=True`. The first run of this sweep did not: it solved against the
  unsolved starting constants, where `lognormal_mu` is 1.7918 rather than the
  solved 0.146, which inflates e_ltv and our bid, so k_global solved to 0.233
  instead of about 1.18. The evaluation then used the solved constants with that
  k_global and ran at a 2 percent win rate. Every number it produced was measured
  in a different auction from the one under study.
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

from t9v2 import autocal as AC                                   # noqa: E402
from t9v2 import generate as G                                   # noqa: E402
from t9v2.core import rng as R                                   # noqa: E402
from t9v2.core.config import load                                # noqa: E402
from t9v2.gen import auction as A                                # noqa: E402
from t9v2.gen import rival_market as M                           # noqa: E402
from t9v2.train.runner import run_seed                           # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = HERE.parent / "docs" / "gates" / "private_sweep.json"

ARMS = [1.0, 2.0, 3.0]
SEEDS = [20250, 20251, 20252]
BASE_BETA, BASE_G, BASE_NG = 0.5, [0.35, 0.75], [0.00, 0.20]


def ov(gain):
    """One dial on both, so the arm stays describable in a sentence."""
    return {"rival_pool.beta_R.value": BASE_BETA * gain,
            "rival_pool.value_loading_gaming.value": [x * gain for x in BASE_G],
            "rival_pool.value_loading_nongaming.value": [x * gain for x in BASE_NG]}


def decompose(gain, kg, seed, n=200_000):
    """Share of var(log lu7) owed to the rival latents, and the value correlation.

    Computed from the bid law before any auction settles, so it does not depend
    on k_global and is the one part of the first (broken) run that stood up.
    """
    s = G.apply_overrides(load("default"),
                          {**ov(gain), "auction.k_global.value": kg})
    run = G.Run(s, seed, "1M")
    st = lambda nm, sub: R.stream(run.seed, nm, block=0, sub=sub)   # noqa: E731
    d = run._join_context_truth(n, 0)
    z = A.value_score(d["ev"], run.z_mu, run.z_sigma)
    base = M.price_core(
        M.draw_pay_shape(st("market", "pay_shape"), n, run.pay_shape), d["fmt"], s)
    part = M.participate(st("market", "participate_k"), run.rivals, d["exch"], d["day"])
    bids = M.rival_bids(st("market", "b_k"), run.rivals, base, z,
                        run.users["os"][d["u"]], run.users["device_type"][d["u"]],
                        d["day"], run.rho, s)
    lb, lbase = np.log(M.competing_bid(bids, part)), np.log(base)
    return float(np.var(lb - lbase) / np.var(lb)), float(np.corrcoef(lb, z)[0, 1])


def main():
    s0 = load("default")
    out, t0 = [], time.time()
    print("private-rival amplitude sweep: x%s, %d seeds at 100K"
          % (", x".join("%g" % a for a in ARMS), len(SEEDS)))
    target, _ = AC.target_of(s0, "auction_win_rate")

    for gain in ARMS:
        def meas(x, g=gain):
            return G.measure("1M", 20250, 200_000, calibrated=True,
                             overrides={**ov(g), "auction.k_global.value": float(x)}
                             )["auction_win_rate"]
        kg, got, iters = AC.bisect(meas, 0.05, 40.0, target, +1)
        share, corr = decompose(gain, kg, SEEDS[0])
        print("\n  gain x%g   k_global %.4f -> win rate %.4f (%d iters)   "
              "latent share %.1f%%   corr(log lu7, z) %.3f"
              % (gain, kg, got, iters, 100 * share, corr))

        o = {**ov(gain), "auction.k_global.value": float(kg)}
        s_arm = G.apply_overrides(load("default"), o)
        for seed in SEEDS:
            m = G.frame("100K", seed=seed, overrides=o)
            r = run_seed(m, s_arm, seed=0, quiet=True)
            w = lambda v: r[v]["heads"]["win"]["auc"]                # noqa: E731
            ev = lambda v: r[v]["economics"]["learned"]["ev_ratio"]  # noqa: E731
            rec = {"gain": gain, "seed": seed, "k_global": float(kg),
                   "latent_share": share, "corr_z": corr,
                   "win_C1": w("C1"), "ssp_win": w("C3") - w("C1"),
                   "ssp_ev": ev("C3") - ev("C1"), "mmp_ev": ev("C2") - ev("C1"),
                   "win_rate": float(m["won"].mean())}
            out.append(rec)
            print("     seed %d   winAUC C1 %.4f   SSP %+.5f   SSP ev %+.5f   "
                  "win rate %.4f" % (seed, rec["win_C1"], rec["ssp_win"],
                                     rec["ssp_ev"], rec["win_rate"]))

    print("\n%-6s %9s %8s %12s %10s %11s %10s"
          % ("gain", "latent%", "corr z", "SSP winAUC", "sd", "C1 winAUC", "win rate"))
    for gain in ARMS:
        sub = [r for r in out if r["gain"] == gain]
        v = [r["ssp_win"] for r in sub]
        print("x%-5g %8.1f%% %8.3f %+12.5f %10.5f %11.4f %10.4f"
              % (gain, 100 * sub[0]["latent_share"], sub[0]["corr_z"],
                 np.mean(v), np.std(v, ddof=1),
                 np.mean([r["win_C1"] for r in sub]),
                 np.mean([r["win_rate"] for r in sub])))
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("\nv1 v10 reported SSP win AUC +0.0136.  wrote %s  (%.0fs)"
          % (OUT.name, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
