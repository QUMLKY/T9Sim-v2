# -*- coding: utf-8 -*-
"""The feature matrix for one view, and the temporal split.

Two jobs, and both are about what a model is ALLOWED to see.

The split is temporal and it is taken BEFORE censoring, on days: train 1-20,
valid 21-24, test 25-28. Identical in all four views, which is what makes the
comparison across views a comparison of data layers rather than of samples.

The leakage guard is the important half. A view can contain a column that a
particular head must not use, and the two cases are different in kind:

  OUTCOME LEAKAGE   the funnel columns are what Tier 1 predicts. Using `install`
                    to predict `is_payer` would score the gate identity, not the
                    model. Each head drops the outcomes at or below its own stage.

  SETTLEMENT LEAKAGE  `winning_price` and `bid_density` are realised AFTER the
                    auction clears. A bidder cannot know either at the moment it
                    bids, so neither is ever a per-row feature. They reach the
                    model only as CELL-LEVEL HISTORY through the SSP encoder, which
                    is what an SSP integration actually gives you: not this
                    auction's clearing price, but what this kind of slot has been
                    clearing at. That distinction is the whole C3 mechanism.

`won` is a feature nowhere. It is always visible, per specification 1.5, and Tier
2 predicts it; letting Tier 1 use it would import the market into the funnel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# the funnel outcomes, in stage order. A head may use nothing at or below its own
# stage, because the generator's gate identities make those deterministic.
FUNNEL_STAGES = [
    ("click", ["click", "click_timestamp"]),
    ("install", ["install", "install_timestamp"]),
    ("payer", ["is_payer"]),
    ("spend", ["ltv_value", "ltv_7d", "ltv_30d"]),
]
ALL_OUTCOME_COLS = [c for _, cols in FUNNEL_STAGES for c in cols]

# realised at settlement, never known at bid time. Cell-level history only.
#
# `min_winning_price` (H5) joins them 22 August 2026, for a stronger reason than
# the other two. It is not merely unknown at bid time, it is Tier 2's LABEL in
# disguise: `won = 1[bid_price >= min_winning_price]` holds on every row by
# construction, so a view carrying it as a feature is not predicting the auction,
# it is reading the answer.
#
# THE CENSORING MAP CANNOT CATCH THIS. H5 is C3/C4-visible by design, and C3 is
# exactly the view whose SSP contrast the identity would fabricate: C3's win head
# would score near 1.0, C1's would not, and the gap would be the leak rather than
# the data layer. That is the shape of v1's `bid_density` leak, which is how v1
# reached an SSP result v2 had to retract. H5 exists as the TARGET of the Tier-2
# price head and is never an input to anything.
SETTLEMENT_COLS = ["winning_price", "bid_density", "min_winning_price"]

# never a feature: Tier 2's label, and not the funnel's business
NEVER = ["won", "user_id", "timestamp"]

CATEGORICAL = ["region", "city", "os", "os_version", "device_type", "app_id",
               "app_category", "advertiser_id", "advertiser_scale", "campaign_id",
               "ad_genre", "ad_exchange", "slot_format", "_daypart"]


def day_of_run(df, settings):
    """The 1-based day within the 28-day window, from the timestamp.

    Derived rather than stored: D4 `week` is generator-internal and never a
    column, so the split has to be recovered from what the view actually carries.
    """
    import datetime as dt
    start = dt.datetime.fromisoformat(
        settings.raw["time"]["window_start_utc"]["value"]).replace(tzinfo=dt.timezone.utc)
    return ((df["timestamp"].to_numpy() - int(start.timestamp())) // 86400 + 1).astype(int)


def add_daypart(df):
    """`_daypart` is one of the 4 encoder keys and is not a stored column.

    Four blocks rather than 24 hours: an encoder cell keyed by the exact hour would
    be a quarter as populated and the shrinkage would swamp it.
    """
    h = df["hour_of_day"].to_numpy()
    out = np.full(len(df), "night", dtype=object)
    out[(h >= 6) & (h < 12)] = "morning"
    out[(h >= 12) & (h < 18)] = "afternoon"
    out[(h >= 18) & (h < 23)] = "evening"
    return out


def split(df, settings):
    """train / valid / test masks, by day. Same days in every view."""
    d = day_of_run(df, settings)
    sd = settings.raw["split_days"]
    lo_tr, hi_tr = sd["train"]
    lo_va, hi_va = sd["valid"]
    lo_te, hi_te = sd["test"]
    return {"train": (d >= lo_tr) & (d <= hi_tr),
            "valid": (d >= lo_va) & (d <= hi_va),
            "test": (d >= lo_te) & (d <= hi_te)}


def base_columns(view_df):
    """Everything in the view that any model may use as a per-row feature.

    Encoder outputs are excluded here even though they sit in the same frame. They
    are DERIVED features and must enter only by being asked for, through `extra`,
    so that which view gets which encoder stays a decision made in one place
    rather than a consequence of what happens to be in the frame.
    """
    banned = set(ALL_OUTCOME_COLS) | set(SETTLEMENT_COLS) | set(NEVER)
    return [c for c in view_df.columns
            if c not in banned and not c.startswith("_enc_")]


def tier1_features(view_df, stage, extra=None):
    """Per-row features for one Tier-1 head.

    A head may not use any outcome at or below its own stage. P(payer | install)
    may not read `install`, because install is 1 on every row it trains on, nor
    `is_payer`, which is its label.
    """
    names = [n for n, _ in FUNNEL_STAGES]
    if stage not in names:
        raise ValueError("unknown funnel stage %r" % stage)
    cols = base_columns(view_df)
    return cols + list(extra or [])


def tier2_features(view_df, extra=None):
    """Per-row features for the win classifier.

    `bid_price` is the feature the bidder sweeps, so it must be present and it
    carries the monotone constraint. Everything realised at settlement is absent.
    """
    cols = base_columns(view_df)
    if "bid_price" not in cols:
        raise RuntimeError("the win classifier needs bid_price and the view has none")
    return cols + list(extra or [])


def prepare(view_df, settings):
    """A working frame: the view, plus `_daypart`, with categoricals typed.

    XGBoost is configured with enable_categorical, so the categorical columns are
    handed over as pandas categories rather than one-hot expanded. app_id and
    campaign_id have hundreds of levels and one-hot would make a matrix wider than
    the training set is tall.
    """
    d = view_df.copy()
    d["_daypart"] = add_daypart(d)
    for c in CATEGORICAL:
        if c in d.columns:
            d[c] = d[c].astype("category")
    return d


def matrix(d, cols):
    """The frame XGBoost is given. Missing columns are an error, not a silent
    fill: a view that lacks a feature should say so."""
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise RuntimeError("features not in the frame: %s" % ", ".join(missing))
    return d[cols]
