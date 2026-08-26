# -*- coding: utf-8 -*-
"""The generation driver: pools once, then auction rows in blocks.

    python -m t9v2.generate --scale 100K --seed 20250

Two passes, and the order between them is the design rather than an optimisation.

  1. The four pools are built and FROZEN. Nothing after this changes them.
  2. A warm-up sample fixes z_mu and z_sigma, the two constants that standardise
     the value score the market reads. They are frozen BEFORE the rows the market
     prices, so the market never reads a statistic of the rows it is pricing.
  3. Rows are drawn in blocks. Each block draws from its own substream of each
     named stream, so blocks are independent and a run can resume.

The block loop holds one block in memory at a time. At 10M rows and 55 columns a
whole-dataset frame would not fit in this machine's 16 GB, and the plan says
stream or batch everything for that reason.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import pools as P
from .core import config, rng as R
from .gen import auction as A
from .gen import funnel as F
from .gen import rival_market as M
from .network import laws as L

OUTPUT = Path(__file__).resolve().parents[2] / "output"
WARMUP_ROWS = 50_000


def fingerprint(s):
    """A hash of everything that decides what a generated row looks like.

    A parquet is reused by FILENAME, and a filename records the scale and the seed
    and nothing else. Change a law or a constant, re-run the campaign, and every
    seed whose file already exists is silently trained on data from the old
    design. Nothing would fail; the numbers would just be wrong, and they would
    look exactly like right ones.

    So the manifest carries this and the reuse path compares it. Two parts,
    because generation is decided by two things:

      SETTINGS  the merged generator settings. training.yaml and validation.yaml
                are excluded deliberately: they change how a dataset is SCORED,
                never what it contains, so including them would force pointless
                regeneration every time a metric changed.

      CODE      the source of the modules that draw the rows. A law edited in
                Python moves the data without touching a single setting, and a
                settings-only fingerprint would miss it entirely.
    """
    import hashlib
    import json as _json

    keys = ["nodes", "column_order", "tables", "meta", "auction", "entity_latents",
            "entity_latents_extra", "rival_pool", "ecpm_targets_usd", "advertiser_scale",
            "app_categories", "ad_genre_mix", "archetype_shares", "archetype_propensity",
            "archetype_ltv_multiplier", "archetype_interest", "app_audience", "device",
            "ltv", "funnel", "funnel_relevance_weights", "funnel_delays", "context",
            "time", "win_rate", "dependency_strengths", "rho", "pool_sizes", "scales",
            "block_rows"]
    blob = _json.dumps({k: s.raw.get(k) for k in keys}, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode("utf-8"))

    here = Path(__file__).resolve().parent
    for rel in ["pools.py", "generate.py", "core/rng.py", "network/laws.py",
                "gen/auction.py", "gen/funnel.py", "gen/rival_market.py"]:
        p = here / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def parquet_is_current(path, s):
    """Is the parquet beside this manifest the one the CURRENT design produces?

    Missing manifest counts as stale: a file with no provenance cannot be shown
    to be current, and the campaign's whole claim is that its 30 datasets came
    from one design.
    """
    man = Path(path).with_suffix(".manifest.json")
    if not Path(path).exists() or not man.exists():
        return False, "no manifest"
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "manifest does not parse"
    have, want = m.get("fingerprint"), fingerprint(s)
    if have is None:
        return False, "manifest predates fingerprinting"
    if have != want:
        return False, "fingerprint %s, current design is %s" % (have, want)
    return True, "current"


def apply_overrides(s, overrides):
    """Set dotted settings paths in place, for the solver only.

    The solver has to re-generate at a trial value of a constant, and the only
    honest way to do that is to change the setting and rebuild, rather than to
    patch the number somewhere downstream and hope the two paths agree. Returns
    the settings so a caller can chain; mutates, so callers pass a fresh load.
    """
    for path, value in (overrides or {}).items():
        node, parts = s.raw, path.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value
    return s


class Run:
    """Everything frozen before the first row: pools, tables, constants."""

    def __init__(self, s, seed, scale, k_global=None, tau_scale=1.0, untilted=False):
        self.s, self.seed, self.scale = s, seed, scale
        self.n_rows = s.raw["scales"][scale]
        sizes = s.raw["pool_sizes"][scale]
        self.n_days = s.raw["time"]["window_days"]["value"]
        # tau_scale and untilted are the stage-4 sensitivity sweep's 2 handles.
        # They touch only the 3 tilted tables; every other law is untouched.
        self.tau_scale, self.untilted = tau_scale, untilted
        self.tables = P.build_tables(s, tau_scale=tau_scale, untilted=untilted)
        self.users = P.build_users(s, seed, sizes["n_users"], self.tables)
        self.apps = P.build_apps(s, seed, sizes["n_apps"])
        self.camps = P.build_campaigns(s, seed, sizes["n_advertisers"], sizes["n_campaigns"])
        self.n_exch = len(s.raw["context"]["ad_exchanges"])
        self.rivals = P.build_rivals(s, seed, self.n_days, self.n_exch)
        self.k_global = k_global

        # tables that depend on the pools, built once
        self.pair_table = A.build_pair_table(s, self.users, self.apps)
        self.vbin, self.vscore = A.user_value_bins(self.users)
        self.cidx_table = A.build_cidx_table(s, self.camps, self.vbin, self.vscore)
        self.groups = [np.flatnonzero(self.users["lu1_archetype"] == k)
                       for k in range(len(P.ARCHETYPES))]
        self.floor_shape, self.pay_shape = M.price_shapes()
        self.rho = s.raw["rho"]["value"]

        self.z_mu, self.z_sigma = self._warm_up()

    def _warm_up(self):
        """Freeze the value-score constants on a separate sample and stream.

        A separate stream, so the warm-up does not consume variates the real rows
        would have drawn, and a separate sample, so nothing in the dataset is
        standardised by a statistic computed from itself.
        """
        n = min(WARMUP_ROWS, self.n_rows)
        ev = self._truth_only(n, block=-1)
        pos = ev > 0
        if not pos.any():
            raise RuntimeError("the warm-up sample produced no row with positive "
                               "expected value, so z cannot be standardised")
        lg = np.log(ev[pos])
        return float(lg.mean()), float(lg.std() or 1.0)

    # -- the row pipeline ---------------------------------------------------

    def _join_context_truth(self, n, block):
        """Steps 1 to 3. Returns the dict of everything the market and the funnel
        need, plus the columns those steps emit."""
        s, tb = self.s, self.tables
        st = lambda name, sub: R.stream(self.seed, name, block=block, sub=sub)
        n_apps = len(self.apps["app_id"])

        # --- 1 join
        pair = A.draw_pair(st("join", "pair_idx"), n, self.pair_table)
        app_i = A.decode_app(pair, n_apps)
        arch_cell = A.decode_archetype(pair, n_apps)
        u = A.draw_user(st("join", "u_rows"), arch_cell, self.groups)
        vbin = self.vbin[u]
        c = A.draw_campaign(st("join", "c_idx"), vbin, self.cidx_table)

        U, AP, CM = self.users, self.apps, self.camps
        arch = U["lu1_archetype"][u]
        cat = AP["app_category"][app_i]
        genre = CM["ad_genre"][c]
        owner = CM["owner"][c]
        tier = CM["adv_tier"][owner]

        # --- 2 context
        exch = A.draw_exchange(st("context", "B3"), n, s)
        fmt = A.draw_format(st("context", "B6"), n, s)
        size = A.draw_size(st("context", "_size"), fmt, s)
        week = A.draw_week(st("context", "D4"), n, s)
        hour = A.draw_hour(st("context", "D2"), arch, tb)
        dow = A.draw_dow(st("context", "D3"), arch, tb)
        ts = A.make_timestamp(st("context", "D1"), week, dow, hour, s)
        day = np.clip(A.day_index(week, dow), 0, self.n_days - 1)

        # --- 3 truth
        r = A.genre_relevance(U["lu6"][u], genre)
        m = A.stage_multipliers(r, s)
        v_slot = A.slot_value(fmt, size, s)
        ease = A.install_ease(cat, s)
        mu_cat = A.category_location(cat, s)
        plat = A.platform_multiplier(U["os"][u], s)
        t_pay = A.payer_timing(hour, dow, tb)

        p_c = A.true_p_click(U["lu2_click_prop"][u], v_slot, m,
                             AP["la1_app_quality"][app_i], CM["lc1_creative_appeal"][c], s)
        p_i = A.true_p_install(U["lu3_install_prop"][u], ease, m,
                               AP["la1_app_quality"][app_i], s)
        p_p = A.true_p_payer(U["lu4_payer_prob"][u], m, t_pay, s)
        e_l = A.expected_ltv(mu_cat, U["lu5_ltv_mult"][u], CM["lc2_game_quality"][c], plat, s)
        ev = A.expected_value(p_c, p_i, p_p, e_l)

        return dict(u=u, app_i=app_i, c=c, arch=arch, cat=cat, genre=genre,
                    tier=tier, exch=exch, fmt=fmt, size=size, hour=hour, dow=dow,
                    ts=ts, day=day, v_slot=v_slot, ease=ease, mu_cat=mu_cat,
                    plat=plat, p_c=p_c, p_i=p_i, p_p=p_p, e_l=e_l, ev=ev)

    def _truth_only(self, n, block):
        return self._join_context_truth(n, block)["ev"]

    def block(self, n, block):
        """One block of rows, as a dict of the 55 columns."""
        s = self.s
        st = lambda name, sub: R.stream(self.seed, name, block=block, sub=sub)
        d = self._join_context_truth(n, block)
        U, AP, CM = self.users, self.apps, self.camps
        u, app_i, c = d["u"], d["app_i"], d["c"]

        z = A.value_score(d["ev"], self.z_mu, self.z_sigma)

        # --- 4 to 9 the market
        fsh = M.draw_floor_shape(st("market", "floor_shape"), n, self.floor_shape)
        floor = M.floor_price(fsh, d["fmt"], s)
        psh = M.draw_pay_shape(st("market", "pay_shape"), n, self.pay_shape)
        base_e = M.price_core(psh, d["fmt"], s)
        part = M.participate(st("market", "participate_k"), self.rivals, d["exch"], d["day"])
        bids = M.rival_bids(st("market", "b_k"), self.rivals, base_e, z,
                            U["os"][u], U["device_type"][u], d["day"], self.rho, s)
        lu7 = M.competing_bid(bids, part)
        dens = M.bid_density(part)
        eltv = M.observable_ltv(d["cat"], s)
        bid = M.our_bid(st("market", "H2"), d["tier"], d["v_slot"], d["ease"],
                        eltv, s, self.k_global)
        won = M.won(bid, lu7, floor)
        sold = M.sold_but_lost(won, lu7, floor)
        wp = M.winning_price(won, sold, bid, lu7)
        # H5, and it draws nothing: a pure function of two columns already here,
        # so the RNG stream is identical with and without it
        mwp = M.min_winning_price(lu7, floor)

        # --- 10 the funnel, on EVERY row, won or lost
        clk = F.click(st("funnel", "E1"), d["p_c"])
        clk_ts = F.click_timestamp(st("funnel", "E2"), clk, d["ts"], s)
        ins = F.install(st("funnel", "F1"), clk, d["p_i"])
        ins_ts = F.install_timestamp(st("funnel", "F2"), ins, clk_ts, s)
        pay = F.is_payer(st("funnel", "G1"), ins, d["p_p"])
        ltv = F.ltv_value(st("funnel", "G2"), pay, d["mu_cat"],
                          U["lu5_ltv_mult"][u], CM["lc2_game_quality"][c], d["plat"], s)

        lu6 = U["lu6"][u]
        la2 = AP["la2"][app_i]
        return {
            "user_id": U["user_id"][u],
            "region": U["region"][u], "city": U["city"][u],
            "os": np.array(P.OS, dtype=object)[U["os"][u]],
            "os_version": U["os_version"][u],
            "device_type": np.array(P.DEVICE, dtype=object)[U["device_type"][u]],
            "lu1_archetype": np.array(P.ARCHETYPES, dtype=object)[d["arch"]],
            "lu2_click_prop": U["lu2_click_prop"][u],
            "lu3_install_prop": U["lu3_install_prop"][u],
            "lu4_payer_prob": U["lu4_payer_prob"][u],
            "lu5_ltv_mult": U["lu5_ltv_mult"][u],
            "lu6_casual": lu6[:, 0], "lu6_strategy": lu6[:, 1],
            "lu6_rpg": lu6[:, 2], "lu6_hypercasual": lu6[:, 3],
            "app_id": AP["app_id"][app_i],
            "app_category": np.array(P.CATEGORIES, dtype=object)[d["cat"]],
            "la1_app_quality": AP["la1_app_quality"][app_i],
            "la2_whale": la2[:, 0], "la2_engaged_spender": la2[:, 1],
            "la2_casual": la2[:, 2], "la2_time_filler": la2[:, 3],
            "la2_inactive": la2[:, 4],
            "campaign_id": CM["campaign_id"][c],
            "advertiser_id": CM["advertiser_id"][CM["owner"][c]],
            "advertiser_scale": np.array(P.TIERS, dtype=object)[d["tier"]],
            "ad_genre": np.array(P.GENRES, dtype=object)[d["genre"]],
            "lc1_creative_appeal": CM["lc1_creative_appeal"][c],
            "lc2_game_quality": CM["lc2_game_quality"][c],
            "ad_exchange": np.array(list(s.raw["context"]["ad_exchanges"]),
                                    dtype=object)[d["exch"]],
            "slot_format": d["fmt"],
            "slot_width": A.slot_width(d["size"]), "slot_height": A.slot_height(d["size"]),
            "timestamp": d["ts"], "hour_of_day": d["hour"], "day_of_week": d["dow"],
            "p_click": d["p_c"], "p_install": d["p_i"], "p_payer": d["p_p"],
            "e_ltv": d["e_l"], "ev_truth": d["ev"],
            "floor_price": floor, "lu7_competing_bid": lu7, "bid_price": bid,
            "won": won, "winning_price": wp, "min_winning_price": mwp,
            "click": clk, "click_timestamp": clk_ts,
            "install": ins, "install_timestamp": ins_ts,
            "is_payer": pay, "ltv_value": ltv,
            "ltv_7d": F.ltv_7d(ltv, s), "ltv_30d": F.ltv_30d(ltv, s),
            "bid_density": dens,
        }


def frame(scale="100K", seed=20250, n_rows=None, overrides=None, profile="default",
          tau_scale=1.0, untilted=False):
    """Generate in memory and return a DataFrame. The solver's measuring stick.

    Same code path as `generate`, deliberately: a solver that measured a
    different pipeline from the one that writes the data would calibrate the
    wrong generator.
    """
    import pandas as pd
    s = apply_overrides(config.load(profile), overrides)
    L.v3_laws_exist(s)
    run = Run(s, seed, scale, tau_scale=tau_scale, untilted=untilted)
    total = n_rows or run.n_rows
    order = s.raw["column_order"]
    out = []
    for start, stop, i in R.block_bounds(total, s.raw["block_rows"]):
        cols = run.block(stop - start, i)
        out.append(pd.DataFrame({c: cols[c] for c in order}))
    return pd.concat(out, ignore_index=True)


def measure(scale="100K", seed=20250, n_rows=None, overrides=None, profile="default",
            calibrated=False):
    """Accumulate the calibration levels block by block, holding no frame.

    Written 16 August 2026 because the solver could not see far enough. The two
    LTV constants are measured on PAYERS, about 4 rows in 10,000, so a 1M-row
    sample holds roughly 350 payers and its top 5 percent is 18 people. Solving a
    whale-concentration target against 18 people is solving against the sample,
    not against the design, and the answer moved from 0.60 to 0.68 between 1M and
    2M on the same seed.

    Reading 10M rows the obvious way would need several GB of frame. This keeps
    per-block counters and the payer spends only, which at 10M is about 4,000
    numbers, so the solver can measure at a scale where the estimator is steady.
    """
    s = apply_overrides(config.load(profile, calibrated=calibrated), overrides)
    L.v3_laws_exist(s)
    run = Run(s, seed, scale)
    total = n_rows or run.n_rows

    rows = clicks = installs = payers = wins = 0
    spend = []
    for start, stop, i in R.block_bounds(total, s.raw["block_rows"]):
        c = run.block(stop - start, i)
        rows += stop - start
        clicks += int(c["click"].sum())
        installs += int(c["install"].sum())
        payers += int(c["is_payer"].sum())
        wins += int(c["won"].sum())
        spend.append(c["ltv_value"][c["is_payer"] == 1])
    sp = np.concatenate(spend) if spend else np.array([])

    if sp.size:
        cut = np.quantile(sp, 0.95)
        whale = float(sp[sp >= cut].sum() / sp.sum())
        median = float(np.median(sp))
    else:
        whale = median = float("nan")
    return {
        "population_ctr": clicks / rows,
        "click_to_install": installs / clicks if clicks else float("nan"),
        "install_to_payer": payers / installs if installs else float("nan"),
        "whale_concentration": whale,
        "median_payer_spend_usd": median,
        "auction_win_rate": wins / rows,
        "_payers": payers,
    }


def generate(scale="100K", seed=20250, profile="default", out=None, k_global=None,
             n_rows=None, quiet=False):
    """Build the dataset and write one parquet. Returns (path, manifest)."""
    import pandas as pd

    s = config.load(profile)
    L.v3_laws_exist(s)                    # V3, the check stage 1 could not run
    run = Run(s, seed, scale, k_global=k_global)
    total = n_rows or run.n_rows
    order = s.raw["column_order"]
    block_rows = s.raw["block_rows"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = Path(out) if out else OUTPUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))

    # Written to disk BLOCK BY BLOCK. The obvious version accumulates the frames
    # and concatenates at the end, which needs the whole dataset in memory twice
    # over: about 2.8 GB at 10M rows, on a machine with 16 GB that also has to
    # hold the pools. The campaign runs ten 10M seeds, so the obvious version
    # would not have survived stage 5 even if it survived one run here.
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0, writer, rows, wins = time.time(), None, 0, 0
    try:
        for start, stop, i in R.block_bounds(total, block_rows):
            cols = run.block(stop - start, i)
            missing = [c for c in order if c not in cols]
            if missing:
                raise RuntimeError("block %d is missing %d of the frozen 55 columns: %s"
                                   % (i, len(missing), ", ".join(missing)))
            block = pa.Table.from_pandas(pd.DataFrame({c: cols[c] for c in order}),
                                         preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, block.schema, compression="snappy")
            writer.write_table(block)
            rows += stop - start
            wins += int(cols["won"].sum())
            if not quiet and (i % 20 == 0 or stop == total):
                print("  %8d / %d rows  %5.1fs" % (stop, total, time.time() - t0))
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "scale": scale, "seed": seed, "rows": int(rows), "columns": len(order),
        "fingerprint": fingerprint(s),
        "profile": profile, "rng_scheme": R.SCHEME,
        "rho": run.rho, "z_mu": run.z_mu, "z_sigma": run.z_sigma,
        "k_global": run.k_global if run.k_global is not None
                    else s.raw["auction"]["k_global"]["value"],
        "pool_sizes": s.raw["pool_sizes"][scale],
        "win_rate": wins / rows,
        "bytes": path.stat().st_size,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return path, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="100K")
    ap.add_argument("--seed", type=int, default=20250)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--rows", type=int, default=None, help="override for a smoke test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    path, man = generate(a.scale, a.seed, a.profile, a.out, n_rows=a.rows)
    print("wrote %s" % path)
    for k in ("rows", "columns", "win_rate", "z_mu", "z_sigma", "seconds"):
        print("  %-10s %s" % (k, man[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
