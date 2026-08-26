# -*- coding: utf-8 -*-
"""The 2 historical-price encoders, and the C3-minus-C1 mechanism.

This is where the SSP layer actually enters the model, and it is worth being
precise about why, because it is the whole result.

Both encoders answer the same question: what does this KIND of slot clear at?
The cell is (app_id, slot_format, ad_exchange, daypart). They differ only in
which rows they are allowed to average, and that difference is a selection bias:

  DSP encoder   the mean of OUR OWN BID on rows WE WON. Available in all four
                views, because every DSP knows what it bid and whether it won.
                It is biased, and biased in a specific direction: we win the
                auctions the market valued least, so a cell's won-row mean sits
                below what that cell really clears at. That is the biased view.

  SSP encoder   the mean of the WINNING PRICE over every row that cleared, won
                and lost-but-sold alike. Available only in C3 and C4. Nothing is
                selected on, so the estimate is unbiased.

The difference between them is not more data. It is the removal of a bias that no
amount of DSP data can remove, which is why C1 cannot reach C3 by seeing more
rows and why the contrast survives at scale.

Both are shrunk toward a backoff parent by empirical Bayes with ONE shared
constant, open item O13:

    EB(cell) = (sum + k * parent) / (count + k)

The parent drops app_id, so a thin app falls back to its (format, exchange,
daypart) neighbourhood rather than to a global mean. k is shared because the
shrinkage already adapts to sample size through `count`: the SSP encoder sees
more rows per cell and is therefore already shrunk less, with no second dial
needed. A second dial would sit exactly where the C3-minus-C1 result lives.

FITTED ON TRAIN ONLY, then applied to valid and test. An encoder fitted on the
rows it scores is a lookup table of the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_FALLBACK = ["slot_format", "ad_exchange", "_daypart"]      # the parent, no app_id


def mask_of(kind, df):
    """Which rows an encoder of this kind is allowed to average.

    Named rather than passed as a closure so an encoder can say what it did and
    reconstruct it later, which is what leave-one-out needs and what a bundle can
    carry. One definition, used by `build` at fit time and by `transform_train`
    at scoring time, so the two cannot disagree about which rows are in the fold.
    """
    if kind == "won":
        return df["won"].to_numpy() == 1
    if kind == "cleared":
        return df["winning_price"].notna().to_numpy()
    if kind == "lost_sold":
        return df["winning_price"].notna().to_numpy() & (df["won"].to_numpy() != 1)
    if kind == "all":
        return np.ones(len(df), dtype=bool)
    raise ValueError("unknown encoder mask %r" % kind)


class PriceEncoder:
    """One empirical-Bayes cell mean, with a backoff parent and a global root."""

    def __init__(self, keys, shrink, name, value_col=None, mask_kind=None):
        self.keys = list(keys)
        self.shrink = float(shrink)
        self.name = name
        self.cell = None
        self.parent = None
        self.root = np.nan
        self.extra = {}
        # WHAT THIS ENCODER AVERAGES AND WHICH ROWS IT MAY AVERAGE, as plain
        # strings. `transform_train` has to reconstruct a row's own contribution
        # to its own cell, which means knowing both; strings rather than a stored
        # closure so a bundle can carry them through parquet and JSON.
        self.value_col = value_col
        self.mask_kind = mask_kind
        # the raw sufficient statistics. An EB value cannot be un-averaged, so
        # leave-one-out needs the sum and the count it was formed from.
        self.cell_sum = None
        self.cell_count = None
        self.par_sum = None
        self.par_count = None

    def fit(self, df, value, weight_mask=None, extra_values=None):
        """`value` is the quantity averaged; `weight_mask` selects the rows allowed.

        The mask is the encoder's whole identity: the DSP encoder passes won rows,
        the SSP encoder passes every row that cleared.
        """
        d = df if weight_mask is None else df.loc[weight_mask]
        v = np.asarray(value if weight_mask is None else value[weight_mask], dtype=float)
        ok = np.isfinite(v)
        d, v = d.loc[ok], v[ok]
        if len(d) == 0:
            raise RuntimeError("encoder %r got no usable rows to fit on" % self.name)

        self.root = float(v.mean())
        tmp = d[self.keys].copy()
        tmp["_v"] = v
        g_cell = tmp.groupby(self.keys, observed=True)["_v"].agg(["sum", "count"])
        g_par = tmp.groupby(KEY_FALLBACK, observed=True)["_v"].agg(["sum", "count"])
        self.cell_sum, self.cell_count = g_cell["sum"], g_cell["count"]
        self.par_sum, self.par_count = g_par["sum"], g_par["count"]

        self.parent = ((g_par["sum"] + self.shrink * self.root)
                       / (g_par["count"] + self.shrink)).rename("eb")
        par_of_cell = self.parent.reindex(
            pd.MultiIndex.from_arrays(
                [g_cell.index.get_level_values(k) for k in KEY_FALLBACK])).to_numpy()
        par_of_cell = np.where(np.isfinite(par_of_cell), par_of_cell, self.root)
        self.cell = pd.Series(
            (g_cell["sum"].to_numpy() + self.shrink * par_of_cell)
            / (g_cell["count"].to_numpy() + self.shrink), index=g_cell.index, name="eb")

        for nm, vals in (extra_values or {}).items():
            e = np.asarray(vals if weight_mask is None else vals[weight_mask], float)[ok]
            t2 = d[self.keys].copy()
            t2["_v"] = e
            gc = t2.groupby(self.keys, observed=True)["_v"].agg(["sum", "count"])
            root = float(e.mean())
            self.extra[nm] = (pd.Series(
                (gc["sum"].to_numpy() + self.shrink * root) / (gc["count"].to_numpy() + self.shrink),
                index=gc.index), root)
        return self

    def transform(self, df):
        """The cell estimate per row, falling back parent then root."""
        idx = pd.MultiIndex.from_frame(df[self.keys].astype(object))
        out = self.cell.reindex(idx).to_numpy(dtype=float)
        pidx = pd.MultiIndex.from_frame(df[KEY_FALLBACK].astype(object))
        par = self.parent.reindex(pidx).to_numpy(dtype=float)
        out = np.where(np.isfinite(out), out, par)
        return np.where(np.isfinite(out), out, self.root)

    def transform_train(self, df):
        """The cell estimate for a row that is INSIDE the fit, with itself removed.

        THE FAILURE THIS EXISTS TO STOP IS SILENT, and it is the highest-risk
        item in v2.2. `build()` fits on the training split and `apply()` then runs
        over the whole frame, so on a training row the row's own value sits inside
        its own cell mean. For an ordinary feature that is a mild optimism. For
        `min_winning_price` it is not: `won = 1[bid_price >= min_winning_price]`
        is an identity, so any leakage of the row's own value is leakage of the
        label itself.

        HOW MUCH LEAKS WITHOUT THIS. Shrinkage holds it down but does not remove
        it. A row's own weight in its cell is `1 / (count + k)`, and the median
        cell holds 4 training rows at 100K and 9 at 1M against k = 20:

            cell count      1      4      9     20     55
            own weight   4.8%   4.2%   3.4%   2.5%   1.3%

        Four percent of an ordinary feature is nothing. Four percent of a column
        that equals the label would manufacture the SSP win-AUC lift v2's whole
        rebuild established was an artifact, and would reverse RQ2's answer. And
        nothing would crash.

        THE ARITHMETIC. EB is `(sum + k . parent) / (count + k)`, so removing row
        i means subtracting its value from the sum and one from the count, in the
        cell AND in the parent the cell shrinks toward:

            parent_i = (par_sum - v_i + k . root) / (par_count - 1 + k)
            cell_i   = (sum - v_i + k . parent_i) / (count - 1 + k)

        The root is left alone. It is a mean over every fitted row — tens of
        thousands — so one row moves it by ~1/n, which is below the precision of
        anything downstream, and subtracting it there would cost a second pass
        for no measurable change.

        A cell of one collapses to `k . parent_i / k`, which is `parent_i`
        exactly: with itself removed a singleton cell has nothing left but its
        backoff, which is the right answer rather than an edge case.

        APPLIED TO ALL FIVE EMITTED COLUMNS, not only the new one. The others are
        not identities, but they are still the row's own history leaking into its
        own features, and a correction applied selectively is one somebody has to
        remember to extend.

        Rows the fit never saw pass through `transform` untouched: a row the DSP
        encoder skipped because we lost it contributed nothing, so there is
        nothing to subtract.
        """
        if self.value_col is None or self.cell_sum is None:
            raise RuntimeError(
                "encoder %r cannot do leave-one-out: it does not know what it "
                "averaged. Refusing rather than silently returning the "
                "contaminated in-fold value." % self.name)

        base = self.transform(df)
        v = np.asarray(df[self.value_col].to_numpy(), dtype=float)
        inside = mask_of(self.mask_kind, df) & np.isfinite(v)
        if not inside.any():
            return base

        idx = pd.MultiIndex.from_frame(df[self.keys].astype(object))
        pidx = pd.MultiIndex.from_frame(df[KEY_FALLBACK].astype(object))
        c_s = self.cell_sum.reindex(idx).to_numpy(dtype=float)
        c_n = self.cell_count.reindex(idx).to_numpy(dtype=float)
        p_s = self.par_sum.reindex(pidx).to_numpy(dtype=float)
        p_n = self.par_count.reindex(pidx).to_numpy(dtype=float)

        k = self.shrink
        par_i = np.where(np.isfinite(p_s) & (p_n > 0),
                         (p_s - v + k * self.root) / (np.maximum(p_n - 1.0, 0.0) + k),
                         self.root)
        cell_i = np.where(np.isfinite(c_s) & (c_n > 0),
                          (c_s - v + k * par_i) / (np.maximum(c_n - 1.0, 0.0) + k),
                          par_i)
        return np.where(inside, cell_i, base)

    def transform_extra(self, df, name):
        s, root = self.extra[name]
        idx = pd.MultiIndex.from_frame(df[self.keys].astype(object))
        out = s.reindex(idx).to_numpy(dtype=float)
        return np.where(np.isfinite(out), out, root)


def build(train_df, view, settings):
    """The encoders this view is entitled to, as (feature name -> values) columns.

    C1 and C2 get the DSP encoder only. C3 and C4 get both, and it is the SECOND
    one that carries the SSP result.
    """
    cfg = settings.raw["encoders"]
    keys = cfg["keys"]["value"]
    k = cfg["shrink"]["value"]
    out = {}

    dsp = PriceEncoder(keys, k, "dsp", "bid_price", "won").fit(
        train_df, train_df["bid_price"].to_numpy(),
        weight_mask=mask_of("won", train_df))
    out["dsp"] = dsp

    if "winning_price" in train_df.columns:
        cleared = train_df["winning_price"].notna().to_numpy()
        won = train_df["won"].to_numpy() == 1

        # THE LOST-ROW CLEARING PRICE, added 16 August 2026 at stage 4.
        #
        # The all-clears encoder below is what training.yaml declares, and it is
        # unbiased in the sense that matters there: it is not selected on winning.
        # But for predicting OUR win it is diluted. In a first-price auction the
        # winning price on a row WE won is our own bid, so about 30 percent of its
        # mass is a number C1 already has in `bid_price`, and only the lost-sold
        # rows carry the competing bid that beat us.
        #
        # That remainder is precisely what SSP ownership reveals and no amount of
        # DSP data can recover: every view can already learn a win curve from its
        # own win/loss outcomes, because `won` is always visible, so a binary
        # outcome per cell is not what the SSP is adding. The price level on the
        # auctions we lost is.
        #
        # Kept as a SEPARATE feature rather than replacing the declared encoder,
        # so what training.yaml describes is still what is built.
        lost_sold = cleared & ~won
        if lost_sold.sum() >= 50:
            out["ssp_lost"] = PriceEncoder(
                keys, k, "ssp_lost", "winning_price", "lost_sold").fit(
                    train_df, train_df["winning_price"].to_numpy(),
                    weight_mask=lost_sold)

        extra = {}
        if "bid_density" in train_df.columns:
            # H9 is the other SSP column. Per-row it is realised at settlement and
            # cannot be a feature, but the CELL's historical mean density is known
            # before bidding and is exactly what SSP ownership tells you: how
            # crowded this kind of slot usually is.
            extra["density"] = train_df["bid_density"].to_numpy(dtype=float)
        ssp = PriceEncoder(keys, k, "ssp", "winning_price", "cleared").fit(
            train_df, train_df["winning_price"].to_numpy(),
            weight_mask=cleared, extra_values=extra)
        out["ssp"] = ssp

    if "min_winning_price" in train_df.columns:
        # THE FIFTH ENCODER, 22 August 2026, and it repairs at source what
        # `ssp_lost` was added to work around.
        #
        # Under first-price clearing, `winning_price` on a row WE won is OUR OWN
        # BID, a number C1 already holds. So the declared SSP encoder averages a
        # mixture and only its lost-sold rows carry a price the DSP could not
        # otherwise know. Measured at 1M seed 20250: `_enc_ssp_price` averages
        # 76.5 percent of rows and only 46.6 percent of them carry a genuine
        # rival price; `_enc_ssp_lost_price` averages 46.6 percent, all genuine.
        # This one averages 100 percent, all genuine.
        #
        # On a WON row `min_winning_price` is the runner-up threshold, which is
        # exactly what a DSP without SSP integration never learns. The one row
        # class where the old feed was worthless becomes the one where this
        # earns its place. No mask and no NaN: the column exists on every row.
        #
        # `ssp_lost` STAYS. The two are different estimators -- a selected mean
        # over lost-sold rows against an unselected mean over all rows -- so
        # retirement is decided by a paired ten-seed ablation at 1M, not by the
        # argument above being persuasive.
        out["ssp_minwin"] = PriceEncoder(
            keys, k, "ssp_minwin", "min_winning_price", "all").fit(
                train_df, train_df["min_winning_price"].to_numpy())
    return out


def apply(encoders, df, train_mask=None):
    """Add the encoder columns to a frame. Returns the new column names.

    `train_mask` marks the rows the encoders were FITTED on, and those rows get
    the leave-one-out estimate instead: their own value removed from their own
    cell and parent. Without it a training row reads a mean it is inside, which
    for `_enc_ssp_minwin_price` means reading a shrunk copy of its own label.
    See `PriceEncoder.transform_train` for the arithmetic and the size of it.

    Omitting it is the scoring path, and correct there: a valid or test row was
    never in the fit, so there is nothing of its own to remove. The eval pass
    and the bundle round trip both take that path.
    """
    names = []

    def put(col, enc, extra=None):
        if extra is not None:
            df[col] = enc.transform_extra(df, extra)
        elif train_mask is None:
            df[col] = enc.transform(df)
        else:
            m = np.asarray(train_mask, dtype=bool)
            out = enc.transform(df)
            out[m] = enc.transform_train(df)[m]
            df[col] = out
        names.append(col)

    put("_enc_dsp_price", encoders["dsp"])
    if "ssp" in encoders:
        put("_enc_ssp_price", encoders["ssp"])
        if "density" in encoders["ssp"].extra:
            # the extra is a mean of `bid_density`, not of a price, and it is
            # not an identity in anything. Left uncorrected rather than given a
            # second sufficient-statistic store for a feature that cannot leak a
            # label; noted here so the omission is a decision, not an oversight.
            put("_enc_ssp_density", encoders["ssp"], extra="density")
    if "ssp_lost" in encoders:
        put("_enc_ssp_lost_price", encoders["ssp_lost"])
    if "ssp_minwin" in encoders:
        put("_enc_ssp_minwin_price", encoders["ssp_minwin"])
    return names
