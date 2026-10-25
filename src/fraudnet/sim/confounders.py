"""Genuine behaviours that look like fraud, generated as events.

A detector is only as good as the genuine activity it has to tell fraud apart from. Each generator here is
the benign twin of one planted pattern, so no single feature separates the classes:

* trips and shopping sprees: bursts of larger purchases at unfamiliar merchants, at odd hours (twin of
  account takeover),
* in-app purchase bursts: several tiny charges at a digital merchant within an hour (twin of card testing),
* rent collection, landlords, forwarding and large one-off transfers: money in from several accounts and
  out again within hours, and big payments to first-time payees (twins of money mule chains),
* group payments: one person pays, friends pay back over two days.

Shared household and workplace devices (the benign twin of a fraud ring's devices) live in ``world``.
"""

from __future__ import annotations

import numpy as np

from fraudnet.sim.config import DAY, DIGITAL, ELECTRONICS, GIFT_CARD, HOUR, MINUTE, TRAVEL, UTILITIES, Channel
from fraudnet.sim.legit import AMOUNT_MULT, APP_PRICES, ECOM_FRACTION, seconds_of_day
from fraudnet.sim.rows import TxnRows
from fraudnet.sim.world import World

GROCERY, RESTAURANT, FUEL, RETAIL = 0, 1, 2, 3


def _established(world: World, rng: np.random.Generator, min_contacts: int = 0) -> tuple[int, np.ndarray]:
    """A customer who signed up before the window, with at least ``min_contacts`` established contacts."""
    for _ in range(200):
        a = int(rng.integers(0, world.n_legit_accounts))
        contacts = world.contacts(a)
        contacts = contacts[world.signup_ts[contacts] < 0]
        if world.signup_ts[a] < 0 and contacts.size >= min_contacts:
            return a, contacts
    raise RuntimeError("no established customer found; the world is too small")


def _day_time(
    world: World, rng: np.random.Generator, lo_hour: int, hi_hour: int, margin_days: int = 1
) -> int:
    day = int(rng.integers(1, world.cfg.n_days - margin_days))
    return day * DAY + int(rng.integers(lo_hour, hi_hour)) * HOUR + int(rng.integers(0, HOUR))


def _purchase(
    world: World, rng: np.random.Generator, rows: TxnRows, ts: int, account: int, category: int, scale: float
) -> None:
    """One purchase at a popular merchant of ``category`` on the customer's own device."""
    merchant = world.popular_merchant(rng, category)
    amount = float(np.exp(world.amount_mu[account])) * float(AMOUNT_MULT[category]) * scale
    channel = Channel.ECOM if rng.random() < ECOM_FRACTION[category] else Channel.POS
    rows.add(ts, account, world.device_at(account, ts), amount, channel, merchant=merchant)


def travel_trips(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """A booking, then several days of purchases at unfamiliar merchants in a different time zone."""
    horizon = world.cfg.n_days * DAY
    categories, weights = (
        [RESTAURANT, RETAIL, TRAVEL, FUEL, ELECTRONICS, GIFT_CARD],
        [0.4, 0.25, 0.15, 0.1, 0.05, 0.05],
    )
    for _ in range(world.cfg.n_trips):
        traveller, _ = _established(world, rng)
        start = int(rng.integers(3, world.cfg.n_days - 3))
        length = int(rng.integers(2, 7))
        shift = int(rng.choice([-6, -3, 3, 6, 9])) * HOUR
        booking = (start - int(rng.integers(1, 3))) * DAY + int(rng.integers(9, 22)) * HOUR
        _purchase(world, rng, rows, booking, traveller, TRAVEL, float(rng.uniform(4.0, 12.0)))
        for _ in range(int(rng.poisson(2.5 * length))):
            day = start + int(rng.integers(0, length))
            ts = day * DAY + int(seconds_of_day(rng, np.zeros(1, dtype=bool))[0]) + shift
            if 0 <= ts < horizon:
                category = int(rng.choice(categories, p=weights))
                _purchase(world, rng, rows, ts, traveller, category, float(rng.uniform(1.0, 2.5)))


def shopping_sprees(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Two to five larger retail or electronics purchases within a couple of hours."""
    for _ in range(world.cfg.n_sprees):
        shopper, _ = _established(world, rng)
        ts = _day_time(world, rng, 10, 21)
        for _ in range(int(rng.integers(2, 6))):
            ts += int(rng.exponential(20 * MINUTE))
            category = RETAIL if rng.random() < 0.6 else ELECTRONICS
            _purchase(world, rng, rows, ts, shopper, category, float(rng.uniform(1.5, 5.0)))


def app_bursts(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Several small in-app charges at one digital merchant within an hour, on the customer's phone."""
    for _ in range(world.cfg.n_app_bursts):
        player, _ = _established(world, rng)
        merchant = world.popular_merchant(rng, DIGITAL)
        ts = _day_time(world, rng, 8, 24)
        for _ in range(int(rng.integers(3, 9))):
            ts += 30 + int(rng.exponential(6 * MINUTE))
            amount = float(APP_PRICES[int(rng.integers(0, 4))])
            rows.add(ts, player, world.device_at(player, ts), amount, Channel.ECOM, merchant=merchant)


def forwarding(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Money collected for someone else and passed on within hours, often to a first-time payee."""
    for _ in range(world.cfg.n_forwards):
        middle, contacts = _established(world, rng, 1)
        sender = int(rng.choice(contacts))
        receiver = int(rng.choice(contacts)) if rng.random() < 0.6 else _established(world, rng)[0]
        if receiver in (middle, sender):
            continue
        ts = _day_time(world, rng, 8, 21)
        amount = float(rng.lognormal(np.log(400.0), 0.6))
        rows.add(ts, sender, world.device_at(sender, ts), amount, Channel.P2P, dst=middle)
        out = ts + int(rng.uniform(0.5, 12.0) * HOUR)
        rows.add(
            out,
            middle,
            world.device_at(middle, out),
            amount * float(rng.uniform(0.8, 1.0)),
            Channel.P2P,
            dst=receiver,
        )


def landlords(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Tenants pay monthly rent to a landlord, who pays a mortgage a few days later."""
    horizon = world.cfg.n_days * DAY
    n_landlords = max(1, int(world.cfg.landlord_fraction * world.n_legit_accounts))
    for _ in range(n_landlords):
        landlord, _ = _established(world, rng)
        tenants = rng.choice(world.n_legit_accounts, size=int(rng.integers(2, 7)), replace=False)
        rent = {int(t): float(round(rng.lognormal(np.log(900.0), 0.3), -1)) for t in tenants if t != landlord}
        pay_day = int(rng.integers(1, 26))
        for month in range(0, world.cfg.n_days, 30):
            total = 0.0
            for tenant, amount in rent.items():
                ts = (month + pay_day + int(rng.integers(-1, 3))) * DAY + int(rng.integers(8, 22)) * HOUR
                if 0 <= ts < horizon and world.signup_ts[tenant] <= ts:
                    rows.add(ts, tenant, world.device_at(tenant, ts), amount, Channel.P2P, dst=landlord)
                    total += amount
            ts = (month + pay_day + int(rng.integers(3, 6))) * DAY + int(rng.integers(9, 18)) * HOUR
            if total > 0 and ts < horizon:
                merchant = world.popular_merchant(rng, UTILITIES)
                rows.add(
                    ts, landlord, world.device_at(landlord, ts), 0.7 * total, Channel.ECOM, merchant=merchant
                )


def big_transfers(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """One-off large payments to someone new: a used car, a deposit, a builder."""
    for _ in range(world.cfg.n_big_transfers):
        payer, _ = _established(world, rng)
        payee, _ = _established(world, rng)
        if payer == payee:
            continue
        ts = _day_time(world, rng, 8, 21)
        amount = float(rng.lognormal(np.log(700.0), 0.7))
        rows.add(ts, payer, world.device_at(payer, ts), amount, Channel.P2P, dst=payee)


def rent_collections(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Flatmates pay their share in, then the organiser pays the bill: benign fan-in followed by fan-out."""
    horizon = world.cfg.n_days * DAY
    for _ in range(world.cfg.n_rent_collections):
        organiser, contacts = _established(world, rng, 2)
        payers = rng.choice(contacts, size=min(contacts.size, int(rng.integers(2, 5))), replace=False)
        t0 = _day_time(world, rng, 9, 20, margin_days=3)
        total, last = 0.0, t0
        for p in payers:
            ts = t0 + int(rng.random() * 20 * HOUR)
            share = round(float(rng.lognormal(np.log(350.0), 0.25)))
            rows.add(ts, int(p), world.device_at(int(p), ts), share, Channel.P2P, dst=organiser)
            total, last = total + share, max(last, ts)
        ts = min(horizon - 1, last + int(rng.uniform(1, 12) * HOUR))
        bill = total + float(rng.lognormal(np.log(350.0), 0.25))
        merchant = world.popular_merchant(rng, UTILITIES)
        rows.add(ts, organiser, world.device_at(organiser, ts), bill, Channel.ECOM, merchant=merchant)


def group_payments(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """One person pays for dinner or a trip, then friends transfer their share over the next two days."""
    horizon = world.cfg.n_days * DAY
    for _ in range(world.cfg.n_group_payments):
        organiser, contacts = _established(world, rng, 2)
        t0 = _day_time(world, rng, 11, 22, margin_days=2)
        travel = rng.random() < 0.3
        total = float(rng.lognormal(np.log(240.0), 0.5)) * (3.0 if travel else 1.0)
        channel = Channel.ECOM if travel else Channel.POS
        merchant = world.popular_merchant(rng, TRAVEL if travel else RESTAURANT)
        rows.add(t0, organiser, world.device_at(organiser, t0), total, channel, merchant=merchant)
        friends = rng.choice(contacts, size=min(contacts.size, int(rng.integers(2, 6))), replace=False)
        for f in friends:
            ts = min(horizon - 1, t0 + int(rng.uniform(1, 48) * HOUR))
            rows.add(
                ts,
                int(f),
                world.device_at(int(f), ts),
                round(total / (friends.size + 1), 2),
                Channel.P2P,
                dst=organiser,
            )


def all_confounders(world: World, rng: np.random.Generator, rows: TxnRows) -> None:
    """Run every generator in a fixed order."""
    for generate in (
        rent_collections,
        group_payments,
        travel_trips,
        shopping_sprees,
        app_bursts,
        forwarding,
        landlords,
        big_transfers,
    ):
        generate(world, rng, rows)
