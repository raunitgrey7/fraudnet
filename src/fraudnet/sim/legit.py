"""Genuine customer behaviour in bulk: everyday card purchases and transfers to friends.

These two streams are about 98 percent of the log, so they are generated with vectorised numpy draws from
each customer's own parameters (rate, spend level, favourite merchants, devices, hour-of-day profile). The
rarer genuine behaviours that look like fraud are generated as events in ``confounders``.
"""

from __future__ import annotations

import numpy as np

from fraudnet.sim.config import (
    CATEGORY_PROFILES,
    DAY,
    DAY_HOURS,
    DIGITAL,
    HOUR,
    NEVER,
    WEEKDAY_WEIGHTS,
    Channel,
)
from fraudnet.sim.world import World

EPOCH_WEEKDAY = 2  # day 0 of the simulation is Wednesday 2026-07-01
RESTAURANT = 1
APP_PRICES = np.array([0.99, 1.99, 2.99, 4.99, 9.99])
AMOUNT_MULT = np.array([p.amount_multiplier for p in CATEGORY_PROFILES])
ECOM_FRACTION = np.array([p.ecom_fraction for p in CATEGORY_PROFILES])
_DAY_CDF = np.cumsum(DAY_HOURS) / np.sum(DAY_HOURS)
_NIGHT_CDF = np.cumsum(np.roll(DAY_HOURS, 6)) / np.sum(DAY_HOURS)


def seconds_of_day(rng: np.random.Generator, night_owl: np.ndarray) -> np.ndarray:
    """Draw a time of day from the diurnal profile; night owls are shifted six hours later."""
    u = rng.random(night_owl.size)
    hour = np.where(night_owl, np.searchsorted(_NIGHT_CDF, u), np.searchsorted(_DAY_CDF, u))
    return np.minimum(hour, 23) * HOUR + rng.integers(0, HOUR, size=night_owl.size)


def row_inverse_cdf(cdf_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample one column per row from row-wise cumulative distributions."""
    return np.minimum((cdf_rows < u[:, None]).sum(axis=1), cdf_rows.shape[1] - 1)


def _activity_window(world: World) -> tuple[np.ndarray, int]:
    horizon = world.cfg.n_days * DAY
    return np.clip(world.first_active_ts, 0, horizon - 1), horizon


def _timestamps(world: World, rng: np.random.Generator, rate: np.ndarray, weekly: bool) -> tuple:
    """Poisson arrivals per account over its active window, with an optional weekday pattern."""
    start, horizon = _activity_window(world)
    oversample = max(WEEKDAY_WEIGHTS) if weekly else 1.0
    counts = rng.poisson(rate * (horizon - start) / DAY * oversample)
    acct = np.repeat(np.arange(rate.size), counts).astype(np.int32)
    day = ((start[acct] + rng.random(acct.size) * (horizon - start[acct])) // DAY).astype(np.int64)
    if weekly:
        weights = np.asarray(WEEKDAY_WEIGHTS)[(day + EPOCH_WEEKDAY) % 7] / oversample
        keep = rng.random(acct.size) < weights
        acct, day = acct[keep], day[keep]
    ts = day * DAY + seconds_of_day(rng, world.night_owl[acct])
    ts = np.clip(ts, start[acct], horizon - 1)
    return acct, ts


def _pick_devices(world: World, rng: np.random.Generator, acct: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Primary device most of the time, otherwise a second device, the family device or a workplace computer."""
    primary = np.where(
        ts >= world.device_change_ts[acct], world.replacement_device[acct], world.primary_device[acct]
    )
    r = rng.random(acct.size)
    secondary = world.secondary_device[acct]
    household = world.household_device[acct]
    workplace = world.workplace_device[acct]
    device = np.where((secondary >= 0) & (r < 0.25), secondary, primary)
    device = np.where((household >= 0) & (r >= 0.25) & (r < 0.37), household, device)
    return np.where((workplace >= 0) & (r >= 0.37) & (r < 0.43), workplace, device).astype(np.int32)


def _pick_merchants(world: World, rng: np.random.Generator, acct: np.ndarray) -> np.ndarray:
    """Favourite merchants most of the time, otherwise explore by category preference and popularity."""
    slot = row_inverse_cdf(world.favourite_cdf[acct], rng.random(acct.size))
    merchant = world.favourites[acct, slot]
    explore = (rng.random(acct.size) >= world.cfg.favourite_share) | (merchant < 0)
    cats = row_inverse_cdf(world.category_cdf[acct[explore]], rng.random(int(explore.sum())))
    picked = np.empty(cats.size, dtype=np.int32)
    for c in np.unique(cats):
        mask = cats == c
        slots = np.searchsorted(world.category_merchant_cdf[c], rng.random(int(mask.sum())), side="right")
        members = world.category_merchants[c]
        picked[mask] = members[np.minimum(slots, members.size - 1)]
    merchant[explore] = picked
    return merchant.astype(np.int32)


def legit_purchases(world: World, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Card purchases of every genuine customer over the whole horizon."""
    cfg = world.cfg
    acct, ts = _timestamps(world, rng, world.rate, weekly=True)
    merchant = _pick_merchants(world, rng, acct)
    cat = world.merchant_category[merchant]
    noise = rng.standard_normal(acct.size)
    amount = np.exp(world.amount_mu[acct] + np.log(AMOUNT_MULT[cat]) + world.amount_sigma[acct] * noise)
    big = rng.random(acct.size) < cfg.big_ticket_probability
    amount[big] *= rng.lognormal(np.log(6.0), 0.4, size=int(big.sum()))
    app = (cat == DIGITAL) & (rng.random(acct.size) < 0.5)
    amount[app] = APP_PRICES[rng.integers(0, APP_PRICES.size, size=int(app.sum()))]
    channel = np.where(rng.random(acct.size) < ECOM_FRACTION[cat], Channel.ECOM, Channel.POS)
    n = acct.size
    return {
        "ts": ts,
        "account": acct,
        "device": _pick_devices(world, rng, acct, ts),
        "merchant": merchant,
        "dst": np.full(n, -1),
        "amount": np.round(np.clip(amount, 0.5, 20_000.0), 2),
        "channel": channel,
        "pattern": np.zeros(n),
        "attack_id": np.full(n, -1),
        "report_ts": np.full(n, NEVER),
    }


def legit_transfers(world: World, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Account-to-account transfers to social contacts (splitting bills, paying friends back)."""
    degree = np.diff(world.contact_ptr)
    rate = np.where(degree > 0, world.cfg.transfer_rate, 0.0)
    acct, ts = _timestamps(world, rng, rate, weekly=False)
    offset = rng.integers(0, np.maximum(degree[acct], 1))
    dst = world.contact_idx[world.contact_ptr[acct] + offset]
    keep = world.signup_ts[dst] <= ts
    acct, ts, dst = acct[keep], ts[keep], dst[keep]
    amount = rng.lognormal(np.log(45.0), 0.9, size=acct.size)
    rounded = rng.random(acct.size) < 0.6
    amount[rounded] = np.maximum(5.0, np.round(amount[rounded] / 5.0) * 5.0)
    n = acct.size
    return {
        "ts": ts,
        "account": acct,
        "device": _pick_devices(world, rng, acct, ts),
        "merchant": np.full(n, -1),
        "dst": dst,
        "amount": np.round(np.clip(amount, 1.0, 5_000.0), 2),
        "channel": np.full(n, Channel.P2P),
        "pattern": np.zeros(n),
        "attack_id": np.full(n, -1),
        "report_ts": np.full(n, NEVER),
    }
