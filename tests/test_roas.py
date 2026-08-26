# -*- coding: utf-8 -*-
"""The ROAS gate: the bidder can now decline a row, and v1's bidder could not.

The gate is `ev / b >= target` at the rung the argmax has already picked. It
DROPS a row or it does not; it never reprices one to meet the target. That
distinction is the whole design and most of what is tested here: a repriced bid
would make the ladder argmax mean one thing in a gated run and another in an
ungated one, and the two could no longer be compared.

THE DEFAULT TARGET WAS 1.0 UNTIL 25 AUGUST 2026, AND AT 1.0 IT DECLINED NOTHING
ON THE REPORTED BIDDER. That was the defect the target change fixed rather than a
property worth keeping: the learned head cannot predict an exact zero, so nothing
it valued ever failed `ev / b >= 1` and the gate was inert on the one bidder the
study reports. At 3.0 it is live on all three policies. The tests below measure
that rather than wishing for it, so a generator or model change that moves a
placed rate makes somebody look.

THE NUMBERS HERE ARE AT 100K, WHERE THE LEARNED HEAD IS WEAKEST. It places about
0.93 of rows at target 3, against 0.22 at 10M in C1. A worse model values more
rows above the price it would pay, so a small-scale placed rate is not a preview
of the reported one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from t9v2 import generate as G
from t9v2.core.config import load
from t9v2.train import bidders as B
from t9v2.train import metrics as M
from t9v2.train.runner import run_seed

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def s():
    return load("default")


# ------------------------------------------------------------------ the gate

def test_the_default_target_is_three():
    """3:1 is the requirement of a buyer at roughly a 33 percent margin.

    Break-even ROAS is one divided by the profit margin, so the target is a
    statement about the advertiser rather than a tuning knob. It moved from 1.0
    on 25 August 2026 because at 1.0 the gate never fired on the learned bidder,
    which made every reported economic figure describe a buyer with no return
    requirement at all. Ken's decision, and the argument is in
    docs/ROAS_Change_Background.md.
    """
    assert B.roas_target(load("default")) == 3.0


def test_no_target_places_every_row():
    """The ungated mode is real, not a default nothing uses.

    It is the only way to reproduce a v2 number, so it has to stay reachable.
    """
    prices = np.array([1.0, 2.0, 4.0])
    curve = np.tile(np.linspace(0.2, 1.0, 3), (3, 1))
    ev = np.array([100.0, 3.0, 0.001])          # the last is worth less than any rung
    _, _, _, placed = B.choose(ev, curve, prices, None)
    assert placed.all() and placed.dtype == bool


def test_a_row_worth_less_than_every_rung_is_declined():
    """The case the gate exists for. Every rung is loss-making, so it bids none.

    Without the gate the argmax still returns something: it takes the LEAST
    loss-making rung and bids it, and if the hurdle on that row happens to be
    low the policy buys value for more than it is worth.
    """
    prices = np.array([1.0, 2.0, 4.0])
    curve = np.array([[0.9, 0.95, 1.0]])
    b, profit, _, placed = B.choose(np.array([0.5]), curve, prices, 1.0)
    assert not placed[0], "0.5 of value cannot justify a 1.0 bid"
    assert profit[0] < 0, "and the argmax it declined was loss-making"
    assert b[0] == 1.0, "the bid is still reported; only the placement is refused"


def test_the_gate_never_reprices():
    """A placed row's bid is EXACTLY the bid an ungated run would make.

    This is the difference between a gate and a constrained search. A search
    that lifted a bid to satisfy the target would change what the argmax means
    and make gated and ungated runs incomparable.
    """
    rng = np.random.default_rng(0)
    prices = np.geomspace(0.1, 1200.0, 60)
    curve = np.sort(rng.random((500, 60)), axis=1)
    ev = rng.lognormal(1.0, 2.0, 500)
    for target in (0.0, 1.0, 2.0, 10.0):
        b, p, j, _ = B.choose(ev, curve, prices, target)
        b0, p0, j0, _ = B.choose(ev, curve, prices, None)
        assert np.array_equal(j, j0) and np.array_equal(b, b0)
        assert np.array_equal(p, p0)


def test_a_higher_target_declines_more_and_never_fewer():
    """Monotone in the target, which is what makes a sweep interpretable."""
    rng = np.random.default_rng(1)
    prices = np.geomspace(0.1, 1200.0, 60)
    curve = np.sort(rng.random((2000, 60)), axis=1)
    ev = rng.lognormal(1.0, 2.0, 2000)
    rates = [B.choose(ev, curve, prices, t)[3].mean()
             for t in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
    assert rates[0] == 1.0, "target 0 declines nothing"
    assert all(a >= b for a, b in zip(rates, rates[1:])), rates
    assert rates[-1] < rates[0], "a target of 50 must decline something here"


# ------------------------------------------------------ what it does downstream

def test_a_declined_row_cannot_win_even_at_a_zero_floor():
    """Why `placed` is passed to the economics rather than inferred from the bid.

    A declined row cannot be marked by bidding zero. About 14 percent of rows
    have a floor of exactly zero, so a zero bid clears the hurdle on them and
    the policy would be recorded as winning an auction it refused to enter.
    """
    bid = np.array([5.0, 5.0])
    value = np.array([100.0, 100.0])
    lu7 = np.zeros(2)
    floor = np.zeros(2)                       # the hurdle is zero on both rows
    r = M.economics(bid, None, value, lu7, floor, placed=np.array([True, False]))
    assert r["wins"] == 1 and r["placed_rate"] == 0.5
    none = M.economics(bid, None, value, lu7, floor,
                       placed=np.zeros(2, dtype=bool))
    assert none["wins"] == 0 and none["spend"] == 0.0 and none["profit"] == 0.0


def test_economics_without_placed_is_the_ungated_number():
    """The default reproduces v2 exactly, so nothing moved by adding the gate."""
    rng = np.random.default_rng(2)
    bid, value = rng.random(300) * 20, rng.random(300) * 30
    lu7, floor = rng.random(300) * 15, rng.random(300) * 5
    a = M.economics(bid, None, value, lu7, floor)
    b = M.economics(bid, None, value, lu7, floor, placed=np.ones(300, dtype=bool))
    assert a["placed_rate"] == 1.0
    assert {k: v for k, v in a.items()} == {k: v for k, v in b.items()}


def test_the_gate_can_only_raise_a_profit_never_lower_it():
    """Whatever it declines was losing money at the target, by construction.

    A declined row is one where `ev < target . b`, so at target 1 it is a row
    whose bid exceeds its own value. Declining it removes a loss if it would
    have won and changes nothing if it would not. The gate therefore cannot cost
    profit at target 1, and this holds for the oracle too.
    """
    rng = np.random.default_rng(3)
    prices = np.geomspace(0.1, 1200.0, 60)
    lu7, floor = rng.random(2000) * 5, np.zeros(2000)
    ev = rng.lognormal(0.0, 2.0, 2000)
    ev[rng.random(2000) < 0.3] = 0.0          # the inactive archetype's exact zeros
    curve = B.true_win_curve(lu7, floor, prices)
    b, _, _, placed = B.choose(ev, curve, prices, 1.0)
    on = M.economics(b, None, ev, lu7, floor, placed)
    off = M.economics(b, None, ev, lu7, floor, None)
    assert on["profit"] >= off["profit"]
    assert on["wins"] <= off["wins"], "it can only decline, never take more"


# -------------------------------------------------------- on the real pipeline

def test_the_gate_is_live_on_every_policy(s, tmp_path):
    """MEASURED, and it is the fact the target change was made to produce.

    At the old target of 1.0 this test asserted the opposite: `learned` placed
    every row, because the learned head cannot predict an exact zero and so
    nothing it valued ever failed `ev / b >= 1`. The other two policies declined
    about 74 percent of rows, since 30 percent of rows have `ev_truth` of exactly
    zero and no bid returns a target of 1 on a row worth nothing.

    At 3.0 the gate bites on all three. Pinned rather than described, so that a
    model change which moves a placed rate makes somebody look.

    THE BOUNDS ARE WIDE ON PURPOSE. What is asserted is that the gate fires and
    roughly where, not a number to four places. A tight bound here would fail on
    an unrelated model change and teach the next person to widen it.
    """
    p = tmp_path / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    master = pd.read_parquet(p)
    # sigma stated: C1 is censored and C3 is absent from this run. The ROAS gate
    # is what is measured, and the price head's scale cannot reach it.
    out = run_seed(master, s, seed=0, views=["C1", "C4"], quiet=True, sigma=1.0)
    for v in ("C1", "C4"):
        e = out[v]["economics"]
        assert out[v]["roas_target"] == 3.0
        assert e["learned"]["placed_rate"] < 1.0,             "%s: the gate declined nothing on the learned bidder, which is the "            "defect the target change fixed" % v
        assert 0.85 < e["learned"]["placed_rate"] < 0.98,             "%s: learned placed %.4f, measured 0.93 at this scale and seed"             % (v, e["learned"]["placed_rate"])
        for policy in ("truth_ev", "oracle"):
            assert 0.10 < e[policy]["placed_rate"] < 0.25,                 "%s/%s placed %.4f" % (v, policy, e[policy]["placed_rate"])


def test_the_gate_costs_total_profit_and_buys_margin(s, tmp_path):
    """What a binding gate does to the oracle, checked against an ungated run.

    THIS TEST ASSERTED THE OPPOSITE UNTIL 25 AUGUST 2026. At target 1.0 the gate
    was inert on the reported bidder, so the invariant worth pinning was that the
    headline numbers did not move. At 3.0 they must move, and in a direction:
    a higher target is a tighter constraint and not a better policy. It can only
    decline rows, so wins and total profit fall or hold, while the rows that
    survive are the profitable ones, so profit per win rises.

    Compared against a recomputed ungated run rather than a remembered number,
    which is what made the old version of this test survive the change of target
    without lying about what it checked.
    """
    p = tmp_path / "m.parquet"
    G.generate(scale="100K", seed=20250, out=str(p), quiet=True)
    master = pd.read_parquet(p)
    out = run_seed(master, s, seed=0, views=["C1"], quiet=True, sigma=1.0)

    from t9v2 import censor as CEN
    from t9v2.train import features as F
    d = F.prepare(CEN.censor(master, "C1", s), s)
    te = F.split(d, s)["test"]
    mt = master[te]
    prices = B.ladder(s)
    ev = B.to_price_unit(mt["ev_truth"].to_numpy(float))
    lu7 = mt["lu7_competing_bid"].to_numpy(float)
    floor = mt["floor_price"].to_numpy(float)
    curve = B.true_win_curve(lu7, floor, prices)
    b, _, _, _ = B.choose(ev, curve, prices, None)
    ungated = M.economics(b, None, ev, lu7, floor, None)

    gated = out["C1"]["economics"]["oracle"]
    assert gated["wins"] <= ungated["wins"], "the gate can only decline"
    assert gated["profit"] <= ungated["profit"],         "a tighter return requirement cannot raise total profit: %.2f -> %.2f"         % (ungated["profit"], gated["profit"])
    per = lambda e: 1000.0 * e["profit"] / e["wins"]
    assert per(gated) > per(ungated),         "the surviving rows must be the profitable ones: %.2f -> %.2f per 1k wins"         % (per(ungated), per(gated))
