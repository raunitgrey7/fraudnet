"""Planted fraud patterns: synthetic-identity rings, money mule chains, account takeover and card testing.

Each injector follows how the pattern works in practice, including how attackers hide, because the point of
the benchmark is to see which signals still catch a pattern once it stops being obvious:

* Rings open four to eight accounts over ten days, share two to four devices, shop like ordinary new
  customers for two to four weeks (sometimes paying each other small amounts), then bust out with large
  purchases, partly at a colluding merchant. Every ring transaction is fraud (the identities are fake). The
  bank finds out when the bust-out goes unpaid, 15 to 40 days later.
* Mule chains move a scam victim's transfer through two to four recruited accounts. Hops take minutes to
  hours, each mule keeps a cut, some hops reuse a payee the mule has paid before, some are made from a
  herder's device, and the money leaves through a crypto or gift card purchase or an exit account.
* Account takeover comes in two styles: smash-and-grab (a fast night-time burst of high-value purchases)
  and low-and-slow (a few moderately larger purchases spread over a day, at ordinary hours). The attacker
  device is reused across victims; a quarter of takeovers also push money into a mule chain.
* Card testing runs stolen cards through tiny charges at popular digital merchants: either a fast burst
  from one device or a slow drip rotated across a farm of devices and three merchants, at app-store prices.
  A quarter of the cards are cashed out later. Most tiny charges are never reported.
"""

from __future__ import annotations

import numpy as np

from fraudnet.sim.config import (
    CATEGORY_PROFILES,
    CRYPTO,
    DAY,
    DIGITAL,
    ELECTRONICS,
    GIFT_CARD,
    HOUR,
    MINUTE,
    NEVER,
    TRAVEL,
    Channel,
    Pattern,
)
from fraudnet.sim.legit import AMOUNT_MULT, APP_PRICES, ECOM_FRACTION
from fraudnet.sim.rows import TxnRows
from fraudnet.sim.world import World

RETAIL = 3
LEGIT_WEIGHTS = np.array([p.legit_weight for p in CATEGORY_PROFILES]) / sum(
    p.legit_weight for p in CATEGORY_PROFILES
)


def delay(rng: np.random.Generator, median_days: float, sigma: float) -> int:
    """Log-normal reporting delay in seconds."""
    return int(np.exp(rng.normal(np.log(median_days), sigma)) * DAY)


class FraudInjector:
    """Adds every planted attack to ``rows`` and allocates the fraudsters' devices, accounts and merchants."""

    def __init__(self, world: World, rng: np.random.Generator, rows: TxnRows) -> None:
        self.world = world
        self.rng = rng
        self.rows = rows
        self.horizon = world.cfg.n_days * DAY
        self.next_attack = 0
        self.herder_devices = [world.new_device() for _ in range(8)]
        self.ato_devices = [world.new_device() for _ in range(world.cfg.n_ato_devices)]
        self.test_devices = [world.new_device() for _ in range(6)]
        self.cashout_devices = [world.new_device() for _ in range(10)]
        mules = np.flatnonzero(world.mule_recruit_ts < NEVER)
        self.exit_accounts = [int(m) for m in rng.choice(mules, size=min(4, mules.size), replace=False)]
        self.mule_links: dict[int, list[int]] = {}

    def run_all(self) -> None:
        """Inject every pattern in a fixed order so the output depends only on the seed."""
        self.rings()
        self.mule_chains()
        self.account_takeovers()
        self.card_testing()

    def _attack(self) -> int:
        self.next_attack += 1
        return self.next_attack - 1

    def _stratified_ts(self, k: int, n: int, lo_day: float, hi_day: float) -> int:
        """Spread ``n`` attacks evenly over the horizon so every time split contains some."""
        day = lo_day + (k + self.rng.random()) * (hi_day - lo_day) / n
        return int(day * DAY)

    def _victim(self, ts: int, min_age_days: float = 30.0) -> int:
        """A genuine, established customer who is not a mule."""
        world = self.world
        for _ in range(1000):
            a = int(self.rng.integers(0, world.n_legit_accounts))
            if world.signup_ts[a] <= ts - min_age_days * DAY and world.mule_recruit_ts[a] == NEVER:
                return a
        raise RuntimeError("no eligible victim; the world is too small")

    def _mule(self, ts: int, exclude: set[int]) -> int | None:
        """A mule recruited before ``ts`` that this chain has not used yet (not an exit account)."""
        ready = np.flatnonzero(self.world.mule_recruit_ts <= ts)
        ready = np.array(
            [m for m in ready if int(m) not in exclude and int(m) not in self.exit_accounts], dtype=np.int64
        )
        if ready.size == 0:
            return None
        return int(self.rng.choice(ready))

    def _merchant(self, categories: list[int], weights: list[float]) -> int:
        return self.world.popular_merchant(self.rng, int(self.rng.choice(categories, p=weights)))

    def _add(
        self,
        ts: int,
        account: int,
        device: int,
        amount: float,
        channel: int,
        pattern: int,
        attack: int,
        report: int,
        **kw: int,
    ) -> None:
        if ts < self.horizon:
            self.rows.add(
                ts,
                account,
                device,
                amount,
                channel,
                pattern=pattern,
                attack_id=attack,
                report_ts=report,
                **kw,
            )

    # Fraud rings -----------------------------------------------------------------------------------------

    def rings(self) -> None:
        """Synthetic-identity rings that share devices, warm up like new customers, then bust out."""
        cfg = self.world.cfg
        for k in range(cfg.n_rings):
            self._ring(self._stratified_ts(k, cfg.n_rings, 1.0, cfg.n_days - 12.0))

    def _ring(self, start: int) -> None:
        rng, world, cfg = self.rng, self.world, self.world.cfg
        attack = self._attack()
        devices = [world.new_device() for _ in range(int(rng.integers(2, 5)))]
        warmup_end = start + int(rng.uniform(14, 28) * DAY)
        bust_end = warmup_end + int(rng.uniform(1, 3) * DAY)
        discovery = bust_end + int(rng.uniform(15, 40) * DAY)
        collusive = world.new_merchant(ELECTRONICS, start) if rng.random() < 0.5 else -1
        members = []
        for _ in range(int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))):
            signup = start + int(rng.random() * 10 * DAY)
            account = world.new_account(signup)
            own = [int(d) for d in rng.choice(devices, size=int(rng.integers(1, 3)), replace=False)]
            members.append((account, own, signup))
        for account, own, signup in members:
            self._ring_warmup(
                account, own, signup + int(rng.uniform(2, 30) * HOUR), warmup_end, attack, discovery
            )
            n_bust = int(rng.poisson(3.0 * (bust_end - warmup_end) / DAY)) + 2
            for ts in np.sort(rng.uniform(warmup_end, bust_end, size=n_bust)).astype(np.int64):
                if collusive >= 0 and rng.random() < 0.4:
                    merchant = collusive
                else:
                    merchant = self._merchant(
                        [ELECTRONICS, RETAIL, TRAVEL, GIFT_CARD], [0.35, 0.3, 0.15, 0.2]
                    )
                amount = float(rng.lognormal(np.log(250.0), 0.6))
                self._add(
                    int(ts),
                    account,
                    own[int(rng.integers(0, len(own)))],
                    amount,
                    Channel.ECOM,
                    Pattern.RING,
                    attack,
                    discovery + int(rng.random() * 2 * DAY),
                    merchant=merchant,
                )
        self._ring_transfers(members, warmup_end, attack, discovery)

    def _ring_warmup(
        self, account: int, own: list[int], first: int, warmup_end: int, attack: int, discovery: int
    ) -> None:
        """Ordinary-looking purchases at popular merchants, sized like a new customer's spend."""
        rng = self.rng
        mu = float(rng.normal(np.log(32.0), 0.5))
        n_warm = int(rng.poisson(0.4 * max(0.0, (warmup_end - first) / DAY)))
        for ts in np.sort(rng.uniform(first, warmup_end, size=n_warm)).astype(np.int64):
            category = int(rng.choice(len(LEGIT_WEIGHTS), p=LEGIT_WEIGHTS))
            amount = float(np.exp(mu + rng.normal(0.0, 0.6))) * float(AMOUNT_MULT[category])
            channel = Channel.ECOM if rng.random() < ECOM_FRACTION[category] else Channel.POS
            merchant = self.world.popular_merchant(rng, category)
            self._add(
                int(ts),
                account,
                own[int(rng.integers(0, len(own)))],
                amount,
                channel,
                Pattern.RING,
                attack,
                discovery + int(rng.random() * 2 * DAY),
                merchant=merchant,
            )

    def _ring_transfers(
        self, members: list[tuple[int, list[int], int]], warmup_end: int, attack: int, discovery: int
    ) -> None:
        """Small payments between ring members during warm-up, to look like friends."""
        rng = self.rng
        for account, own, signup in members:
            if rng.random() >= 0.3 or len(members) < 2:
                continue
            for _ in range(int(rng.integers(1, 3))):
                other = members[int(rng.integers(0, len(members)))][0]
                ts = int(rng.uniform(signup + DAY, max(signup + DAY + 1, warmup_end)))
                if other != account:
                    amount = float(rng.lognormal(np.log(60.0), 0.5))
                    self._add(
                        ts, account, own[0], amount, Channel.P2P, Pattern.RING, attack, discovery, dst=other
                    )

    # Money mule chains -----------------------------------------------------------------------------------

    def mule_chains(self) -> None:
        """Scam victims push money to a mule, who forwards it along a chain."""
        rng, world, cfg = self.rng, self.world, self.world.cfg
        for k in range(cfg.n_mule_chains):
            t0 = self._stratified_ts(k, cfg.n_mule_chains, 2.0, cfg.n_days - 1.0) // DAY * DAY
            t0 += int(rng.integers(9, 21)) * HOUR + int(rng.integers(0, HOUR))
            first = self._mule(t0, set())
            if first is None:
                continue
            victim = self._victim(t0)
            attack = self._attack()
            amount = float(round(rng.lognormal(np.log(700.0), 0.6)))
            reported = t0 + delay(rng, 1.5, 0.6)
            self._add(
                t0,
                victim,
                world.device_at(victim, t0),
                amount,
                Channel.P2P,
                Pattern.MULE,
                attack,
                reported,
                dst=first,
            )
            self.forward(first, t0, amount, int(rng.integers(2, 5)) - 1, attack, reported, {first})

    def _next_mule(self, mule: int, ts: int, used: set[int]) -> int | None:
        """Prefer a mule this one has paid before (so the payee is not new), otherwise any free mule."""
        known = [m for m in self.mule_links.get(mule, []) if m not in used]
        if known and self.rng.random() < 0.4:
            return int(self.rng.choice(known))
        return self._mule(ts, used)

    def forward(
        self, mule: int, t_in: int, amount: float, hops_left: int, attack: int, reported: int, used: set[int]
    ) -> None:
        """The mule passes the money on (sometimes split in two) or cashes it out."""
        rng, world = self.rng, self.world
        ts = t_in + 5 * MINUTE + int(rng.exponential(6 * HOUR))
        out = amount * float(rng.uniform(0.80, 0.97))
        herder = rng.random() < 0.4
        device = (
            self.herder_devices[int(rng.integers(0, len(self.herder_devices)))]
            if herder
            else world.device_at(mule, ts)
        )
        # The bank traces the chain after the victim's report, hop by hop: a hop is reported after both exist.
        report = max(reported, ts) + int(rng.uniform(0.5, 8.0) * DAY)
        nxt = self._next_mule(mule, ts, used) if hops_left > 0 else None
        if nxt is None:
            if rng.random() < 0.5 and self.exit_accounts:
                exit_account = self.exit_accounts[int(rng.integers(0, len(self.exit_accounts)))]
                self._add(ts, mule, device, out, Channel.P2P, Pattern.MULE, attack, report, dst=exit_account)
            else:
                merchant = world.popular_merchant(rng, CRYPTO if rng.random() < 0.6 else GIFT_CARD)
                self._add(
                    ts, mule, device, out, Channel.ECOM, Pattern.MULE, attack, report, merchant=merchant
                )
            return
        targets = [nxt]
        if len(used) == 1 and rng.random() < 0.3:
            extra = self._mule(ts, used | {nxt})
            if extra is not None:
                targets.append(extra)
        for i, target in enumerate(targets):
            used.add(target)
            self.mule_links.setdefault(mule, []).append(target)
            share = out / len(targets)
            self._add(
                ts + i * int(rng.integers(30, 300)),
                mule,
                device,
                share,
                Channel.P2P,
                Pattern.MULE,
                attack,
                report,
                dst=target,
            )
            self.forward(target, ts, share, hops_left - 1, attack, reported, used)

    # Account takeover ------------------------------------------------------------------------------------

    def account_takeovers(self) -> None:
        """Smash-and-grab bursts and low-and-slow sprees from reused attacker devices."""
        rng, cfg = self.rng, self.world.cfg
        for k in range(cfg.n_ato):
            day_start = self._stratified_ts(k, cfg.n_ato, 1.0, float(cfg.n_days)) // DAY * DAY
            smash = rng.random() < 0.4
            if smash:
                hour = int(rng.integers(0, 6)) if rng.random() < 0.6 else int(rng.integers(0, 24))
            else:
                hour = int(rng.integers(8, 21))
            self._takeover(day_start + hour * HOUR + int(rng.integers(0, HOUR)), smash)

    def _takeover(self, t0: int, smash: bool) -> None:
        rng, world = self.rng, self.world
        victim = self._victim(t0)
        device = self.ato_devices[int(rng.integers(0, len(self.ato_devices)))]
        attack = self._attack()
        if smash:
            n = int(rng.integers(3, 9))
            gaps = MINUTE + rng.exponential(12 * MINUTE, size=n)
            scale, mix = (3.0, 12.0), ([ELECTRONICS, GIFT_CARD, CRYPTO], [0.5, 0.3, 0.2])
        else:
            n = int(rng.integers(2, 6))
            gaps = 10 * MINUTE + rng.exponential(4 * HOUR, size=n)
            scale, mix = (1.5, 4.0), ([RETAIL, ELECTRONICS, GIFT_CARD], [0.4, 0.35, 0.25])
        times = t0 + np.cumsum(gaps).astype(np.int64)
        noticed = int(times[-1]) + delay(rng, 2.0, 0.7)
        mule = self._mule(t0, set()) if rng.random() < 0.25 else None
        transfer_at = int(rng.integers(0, n)) if mule is not None else -1
        typical = float(np.exp(world.amount_mu[victim]))
        for i, ts in enumerate(int(t) for t in times):
            report = noticed + int(rng.random() * 6 * HOUR)
            if i == transfer_at and mule is not None:
                amount = float(rng.uniform(500, 2500))
                self._add(ts, victim, device, amount, Channel.P2P, Pattern.ATO, attack, report, dst=mule)
                if ts < self.horizon:
                    self.forward(mule, ts, amount, int(rng.integers(1, 4)), self._attack(), noticed, {mule})
                continue
            merchant = self._merchant(*mix)
            category = int(world.all_merchant_category()[merchant])
            amount = min(6000.0, typical * float(AMOUNT_MULT[category]) ** 0.5 * float(rng.uniform(*scale)))
            self._add(
                ts, victim, device, amount, Channel.ECOM, Pattern.ATO, attack, report, merchant=merchant
            )

    # Card testing ----------------------------------------------------------------------------------------

    def card_testing(self) -> None:
        """Tiny charges on many stolen cards, as a fast burst or a slow drip across a device farm."""
        rng, cfg = self.rng, self.world.cfg
        digital = self.world.category_merchants[DIGITAL]
        targets = [int(m) for m in digital[: min(4, digital.size)]]
        for k in range(cfg.n_card_tests):
            t0 = self._stratified_ts(k, cfg.n_card_tests, 1.0, cfg.n_days - 0.5)
            rng.shuffle(targets)
            if rng.random() < 0.4:
                fresh = rng.random() < 0.6
                device = (
                    self.world.new_device()
                    if fresh
                    else self.test_devices[int(rng.integers(0, len(self.test_devices)))]
                )
                self._test_run(t0, [device], targets[:2], burst=True)
            else:
                farm = [self.world.new_device() for _ in range(int(rng.integers(6, 13)))]
                self._test_run(t0, farm, targets[:3], burst=False)

    def _test_run(self, t0: int, devices: list[int], merchants: list[int], burst: bool) -> None:
        rng, world = self.rng, self.world
        attack = self._attack()
        ready = np.flatnonzero(world.signup_ts <= t0)
        cards = rng.choice(ready, size=min(ready.size, int(rng.integers(10, 33))), replace=False)
        ts = t0
        for j, card in enumerate(int(c) for c in cards):
            device = devices[(j * len(devices)) // len(cards)]
            for _ in range(1 if rng.random() < 0.7 else 2):
                ts += 5 + int(rng.exponential(45)) if burst else 20 + int(rng.exponential(4 * MINUTE))
                if burst or rng.random() < 0.5:
                    amount = round(float(rng.uniform(0.5, 3.0 if burst else 5.0)), 2)
                else:
                    amount = float(APP_PRICES[int(rng.integers(0, 4))])
                report = ts + int(rng.uniform(5, 35) * DAY) if rng.random() < 0.45 else NEVER
                merchant = (
                    merchants[0]
                    if burst and rng.random() < 0.8
                    else merchants[int(rng.integers(0, len(merchants)))]
                )
                self._add(
                    ts,
                    card,
                    device,
                    amount,
                    Channel.ECOM,
                    Pattern.CARD_TESTING,
                    attack,
                    report,
                    merchant=merchant,
                )
            if rng.random() < 0.25:
                cash = ts + int(rng.uniform(6, 48) * HOUR)
                amount = min(6000.0, float(np.exp(world.amount_mu[card])) * float(rng.uniform(4, 10)))
                device = self.cashout_devices[int(rng.integers(0, len(self.cashout_devices)))]
                merchant = self._merchant([ELECTRONICS, GIFT_CARD], [0.6, 0.4])
                self._add(
                    cash,
                    card,
                    device,
                    amount,
                    Channel.ECOM,
                    Pattern.CARD_TESTING,
                    attack,
                    cash + delay(rng, 3.0, 0.6),
                    merchant=merchant,
                )
