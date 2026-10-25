"""The static part of the simulated world: customers, merchants, devices, households and social ties.

Behaviour is drawn per customer, not per transaction, because fraud detection is about deviations from a
customer's own habits: each account gets a spend level, a purchase rate, a set of favourite merchants, an
hour-of-day profile and one or more devices. Households share a family device and some colleagues share a
workplace computer: these are the benign versions of the device sharing that fraud rings show, so device
features cannot be a free lunch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fraudnet.sim.config import CATEGORY_PROFILES, DAY, NEVER, SimConfig

FAVOURITES_MAX = 14


@dataclass
class World:
    """Per-account and per-merchant parameters plus the id allocators used by the fraud injectors."""

    cfg: SimConfig
    signup_ts: np.ndarray
    first_active_ts: np.ndarray
    rate: np.ndarray
    amount_mu: np.ndarray
    amount_sigma: np.ndarray
    night_owl: np.ndarray
    category_cdf: np.ndarray
    favourites: np.ndarray
    favourite_cdf: np.ndarray
    primary_device: np.ndarray
    secondary_device: np.ndarray
    household_device: np.ndarray
    workplace_device: np.ndarray
    replacement_device: np.ndarray
    device_change_ts: np.ndarray
    contact_ptr: np.ndarray
    contact_idx: np.ndarray
    merchant_category: np.ndarray
    merchant_created_ts: np.ndarray
    category_merchants: list[np.ndarray]
    category_merchant_cdf: list[np.ndarray]
    mule_recruit_ts: np.ndarray
    n_devices: int
    extra_account_signup: list[int] = field(default_factory=list)
    extra_merchant_category: list[int] = field(default_factory=list)
    extra_merchant_created: list[int] = field(default_factory=list)

    @property
    def n_legit_accounts(self) -> int:
        """Accounts that belong to genuine customers (ring identities are appended after them)."""
        return int(self.signup_ts.shape[0])

    def new_device(self) -> int:
        """Allocate a device id that no customer owns (used by fraudsters)."""
        self.n_devices += 1
        return self.n_devices - 1

    def new_account(self, signup_ts: int) -> int:
        """Allocate a synthetic-identity account that signs up at ``signup_ts``."""
        self.extra_account_signup.append(int(signup_ts))
        return self.n_legit_accounts + len(self.extra_account_signup) - 1

    def new_merchant(self, category: int, created_ts: int) -> int:
        """Allocate a merchant controlled by fraudsters."""
        self.extra_merchant_category.append(int(category))
        self.extra_merchant_created.append(int(created_ts))
        return int(self.merchant_category.shape[0]) + len(self.extra_merchant_category) - 1

    def contacts(self, account: int) -> np.ndarray:
        """Social contacts of a genuine account (people it sends money to)."""
        return self.contact_idx[self.contact_ptr[account] : self.contact_ptr[account + 1]]

    def device_at(self, account: int, ts: int) -> int:
        """The device a genuine customer would use at time ``ts`` (accounts for a phone change)."""
        if ts >= self.device_change_ts[account]:
            return int(self.replacement_device[account])
        return int(self.primary_device[account])

    def popular_merchant(self, rng: np.random.Generator, category: int) -> int:
        """Draw a merchant of ``category`` by popularity."""
        merchants = self.category_merchants[category]
        slot = int(np.searchsorted(self.category_merchant_cdf[category], rng.random(), side="right"))
        return int(merchants[min(slot, merchants.size - 1)])

    def all_signup_ts(self) -> np.ndarray:
        """Signup time of every account, genuine first."""
        extra = np.asarray(self.extra_account_signup, dtype=np.int64)
        return np.concatenate([self.signup_ts, extra])

    def all_merchant_category(self) -> np.ndarray:
        """Category of every merchant, genuine first."""
        extra = np.asarray(self.extra_merchant_category, dtype=np.int8)
        return np.concatenate([self.merchant_category, extra])

    def all_merchant_created(self) -> np.ndarray:
        """Creation time of every merchant, genuine first."""
        extra = np.asarray(self.extra_merchant_created, dtype=np.int64)
        return np.concatenate([self.merchant_created_ts, extra])


def _merchants(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, list, list]:
    """Assign categories by share and a Zipf popularity within each category."""
    shares = np.array([p.merchant_share for p in CATEGORY_PROFILES])
    counts = np.maximum(1, np.round(shares / shares.sum() * cfg.n_merchants)).astype(int)
    counts[0] += cfg.n_merchants - counts.sum()
    category = np.repeat(np.arange(len(counts)), counts).astype(np.int8)
    rng.shuffle(category)
    members, cdfs = [], []
    for c in range(len(counts)):
        ids = np.flatnonzero(category == c)
        rng.shuffle(ids)
        weights = 1.0 / np.arange(1, ids.size + 1) ** 1.1
        members.append(ids)
        cdfs.append(np.cumsum(weights) / weights.sum())
    return category, members, cdfs


def _favourites(
    cfg: SimConfig, rng: np.random.Generator, category_cdf: np.ndarray, members: list, cdfs: list
) -> tuple:
    """Each customer keeps a short list of favourite merchants drawn from their category preferences."""
    n = cfg.n_accounts
    favourites = np.full((n, FAVOURITES_MAX), -1, dtype=np.int32)
    favourite_cdf = np.ones((n, FAVOURITES_MAX))
    sizes = rng.integers(6, FAVOURITES_MAX + 1, size=n)
    for a in range(n):
        cats = np.searchsorted(category_cdf[a], rng.random(sizes[a]), side="right")
        cats = np.minimum(cats, len(members) - 1)
        chosen = [
            members[c][min(int(np.searchsorted(cdfs[c], rng.random(), side="right")), members[c].size - 1)]
            for c in cats
        ]
        chosen = np.unique(chosen)
        favourites[a, : chosen.size] = chosen
        weights = rng.dirichlet(np.ones(chosen.size))
        favourite_cdf[a, : chosen.size] = np.cumsum(weights)
    return favourites, favourite_cdf


def _contacts(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Social ties with locality: most contacts come from the same neighbourhood block of 50 accounts."""
    n = cfg.n_accounts
    degree = rng.poisson(2.5, size=n) + 1
    lists: list[np.ndarray] = []
    for a in range(n):
        block = (a // 50) * 50
        local = rng.integers(block, min(block + 50, n), size=degree[a])
        far = rng.integers(0, n, size=degree[a])
        picks = np.where(rng.random(degree[a]) < 0.7, local, far)
        lists.append(np.unique(picks[picks != a]))
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([x.size for x in lists])
    idx = np.concatenate(lists).astype(np.int32) if lists else np.zeros(0, dtype=np.int32)
    return ptr, idx


def _devices(cfg: SimConfig, rng: np.random.Generator, signup_ts: np.ndarray) -> dict[str, np.ndarray | int]:
    """Personal devices, optional second devices, phone changes and shared household devices."""
    n = cfg.n_accounts
    primary = np.arange(n, dtype=np.int32)
    next_id = n
    secondary = np.full(n, -1, dtype=np.int32)
    has_second = rng.random(n) < cfg.second_device_fraction
    secondary[has_second] = np.arange(next_id, next_id + has_second.sum())
    next_id += int(has_second.sum())
    changes = rng.random(n) < cfg.device_change_fraction
    replacement = primary.copy()
    replacement[changes] = np.arange(next_id, next_id + changes.sum())
    next_id += int(changes.sum())
    change_ts = np.full(n, NEVER, dtype=np.int64)
    lo = np.maximum(signup_ts, 0)
    span = cfg.n_days * DAY - lo
    change_ts[changes] = lo[changes] + (rng.random(changes.sum()) * span[changes]).astype(np.int64)
    household, next_id = _shared_groups(
        rng, n, rng.permutation(n), int(cfg.household_fraction * n), (2, 5), next_id
    )
    established = rng.permutation(np.flatnonzero(signup_ts < 0))
    workplace, next_id = _shared_groups(
        rng, n, established, int(cfg.workplace_fraction * n), (5, 13), next_id
    )
    return {
        "primary": primary,
        "secondary": secondary,
        "replacement": replacement,
        "change_ts": change_ts,
        "household": household,
        "workplace": workplace,
        "n_devices": next_id,
    }


def _shared_groups(
    rng: np.random.Generator,
    n: int,
    order: np.ndarray,
    n_members: int,
    size_range: tuple[int, int],
    next_id: int,
) -> tuple[np.ndarray, int]:
    """Split the first ``n_members`` accounts of ``order`` into groups that share one device each."""
    shared = np.full(n, -1, dtype=np.int32)
    pos = 0
    while pos < min(n_members, order.size):
        size = int(rng.integers(*size_range))
        shared[order[pos : pos + size]] = next_id
        next_id += 1
        pos += size
    return shared, next_id


def build_world(cfg: SimConfig, rng: np.random.Generator) -> World:
    """Draw every static parameter of the world from ``rng``."""
    n = cfg.n_accounts
    horizon = cfg.n_days * DAY
    age_days = rng.lognormal(np.log(400.0), 1.0, size=n)
    signup_ts = (-age_days * DAY).astype(np.int64)
    new = rng.random(n) < cfg.new_account_fraction
    signup_ts[new] = (rng.random(new.sum()) * (horizon - 5 * DAY)).astype(np.int64)
    first_active = signup_ts + (rng.random(n) * 3 * DAY).astype(np.int64)
    category, members, cdfs = _merchants(cfg, rng)
    base = np.array([p.legit_weight for p in CATEGORY_PROFILES])
    prefs = rng.dirichlet(base * 20.0, size=n)
    category_cdf = np.cumsum(prefs, axis=1)
    favourites, favourite_cdf = _favourites(cfg, rng, category_cdf, members, cdfs)
    ptr, idx = _contacts(cfg, rng)
    dev = _devices(cfg, rng, signup_ts)
    mule = np.full(n, NEVER, dtype=np.int64)
    eligible = np.flatnonzero(~new)
    n_mules = max(3, int(cfg.mule_fraction * n))
    picks = rng.choice(eligible, size=n_mules, replace=False)
    mule[picks] = ((rng.random(n_mules) * 0.85 - 0.1) * horizon).astype(np.int64)
    return World(
        cfg=cfg,
        signup_ts=signup_ts,
        first_active_ts=first_active,
        rate=rng.lognormal(np.log(cfg.purchase_rate_median), cfg.purchase_rate_sigma, size=n),
        amount_mu=rng.normal(np.log(32.0), 0.5, size=n),
        amount_sigma=rng.uniform(0.45, 0.9, size=n),
        night_owl=rng.random(n) < cfg.night_owl_fraction,
        category_cdf=category_cdf,
        favourites=favourites,
        favourite_cdf=favourite_cdf,
        primary_device=dev["primary"],
        secondary_device=dev["secondary"],
        household_device=dev["household"],
        workplace_device=dev["workplace"],
        replacement_device=dev["replacement"],
        device_change_ts=dev["change_ts"],
        contact_ptr=ptr,
        contact_idx=idx,
        merchant_category=category,
        merchant_created_ts=np.full(cfg.n_merchants, -365 * DAY, dtype=np.int64),
        category_merchants=members,
        category_merchant_cdf=cdfs,
        mule_recruit_ts=mule,
        n_devices=int(dev["n_devices"]),
    )
