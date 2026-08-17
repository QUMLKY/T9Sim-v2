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


class PriceEncoder:
    """One empirical-Bayes cell mean, with a backoff parent and a global root."""

    def __init__(self, keys, shrink, name):
        self.keys = list(keys)
        self.shrink = float(shrink)
        self.name = name
        self.cell = None
        self.parent = None
        self.root = np.nan
        self.extra = {}

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

    dsp = PriceEncoder(keys, k, "dsp").fit(
        train_df, train_df["bid_price"].to_numpy(),
        weight_mask=(train_df["won"].to_numpy() == 1))
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
            out["ssp_lost"] = PriceEncoder(keys, k, "ssp_lost").fit(
                train_df, train_df["winning_price"].to_numpy(),
                weight_mask=lost_sold)

        extra = {}
        if "bid_density" in train_df.columns:
            # H9 is the other SSP column. Per-row it is realised at settlement and
            # cannot be a feature, but the CELL's historical mean density is known
            # before bidding and is exactly what SSP ownership tells you: how
            # crowded this kind of slot usually is.
            extra["density"] = train_df["bid_density"].to_numpy(dtype=float)
        ssp = PriceEncoder(keys, k, "ssp").fit(
            train_df, train_df["winning_price"].to_numpy(),
            weight_mask=cleared, extra_values=extra)
        out["ssp"] = ssp
    return out


def apply(encoders, df):
    """Add the encoder columns to a frame. Returns the new column names."""
    names = []
    df["_enc_dsp_price"] = encoders["dsp"].transform(df)
    names.append("_enc_dsp_price")
    if "ssp" in encoders:
        df["_enc_ssp_price"] = encoders["ssp"].transform(df)
        names.append("_enc_ssp_price")
        if "density" in encoders["ssp"].extra:
            df["_enc_ssp_density"] = encoders["ssp"].transform_extra(df, "density")
            names.append("_enc_ssp_density")
    if "ssp_lost" in encoders:
        df["_enc_ssp_lost_price"] = encoders["ssp_lost"].transform(df)
        names.append("_enc_ssp_lost_price")
    return names
