"""Simulator configuration and the fixed vocabularies shared by every module.

All behaviour parameters live in one frozen dataclass so a run is fully described by its config and its
seed, and so the benchmark can record the exact world it was measured on. The numeric defaults are chosen
to match published shapes rather than published values: purchase rates and amounts are log-normal across
customers (heavy-tailed spend), a handful of merchants take most of the volume (Zipf popularity), and the
fraud rate stays under one percent of transactions, as in card portfolios.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

DAY = 86_400
HOUR = 3_600
MINUTE = 60
NEVER = (1 << 62) - 1  # report time of a transaction whose fraud label never arrives

CATEGORIES: tuple[str, ...] = (
    "grocery",
    "restaurant",
    "fuel",
    "retail",
    "electronics",
    "travel",
    "digital",
    "gift_card",
    "crypto",
    "utilities",
)
TRANSFER_CATEGORY = len(CATEGORIES)  # merchant_category code used for account-to-account transfers
ELECTRONICS, TRAVEL, DIGITAL, GIFT_CARD, CRYPTO, UTILITIES = 4, 5, 6, 7, 8, 9
HIGH_RISK_CATEGORIES: frozenset[int] = frozenset({ELECTRONICS, GIFT_CARD, CRYPTO})


@dataclass(frozen=True)
class CategoryProfile:
    """How a merchant category behaves: share of merchants, share of legit spend, ticket size, channel."""

    merchant_share: float
    legit_weight: float
    amount_multiplier: float
    ecom_fraction: float


CATEGORY_PROFILES: tuple[CategoryProfile, ...] = (
    CategoryProfile(0.24, 0.30, 1.0, 0.10),  # grocery
    CategoryProfile(0.20, 0.22, 0.7, 0.15),  # restaurant
    CategoryProfile(0.08, 0.09, 1.3, 0.00),  # fuel
    CategoryProfile(0.15, 0.13, 1.6, 0.40),  # retail
    CategoryProfile(0.07, 0.03, 4.0, 0.60),  # electronics
    CategoryProfile(0.05, 0.02, 6.0, 0.85),  # travel
    CategoryProfile(0.08, 0.10, 0.35, 1.00),  # digital
    CategoryProfile(0.04, 0.02, 1.8, 0.70),  # gift_card
    CategoryProfile(0.03, 0.01, 3.0, 1.00),  # crypto
    CategoryProfile(0.06, 0.08, 2.5, 1.00),  # utilities
)

# Relative purchase volume by day of week (Monday first); weekends carry more discretionary spend.
WEEKDAY_WEIGHTS: tuple[float, ...] = (0.92, 0.94, 0.96, 0.98, 1.06, 1.14, 1.00)

# Hour-of-day profile of a typical customer: quiet nights, a lunch peak and an evening peak.
DAY_HOURS: tuple[float, ...] = (
    0.4, 0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 1.5, 2.6, 3.4, 4.0, 4.6,
    5.6, 5.0, 4.4, 4.2, 4.6, 5.4, 6.2, 6.0, 5.0, 3.6, 2.2, 1.0,
)  # fmt: skip


class Channel(IntEnum):
    """How the payment was made."""

    POS = 0
    ECOM = 1
    P2P = 2


class Pattern(IntEnum):
    """Ground-truth fraud pattern of a transaction; ``LEGIT`` for genuine activity."""

    LEGIT = 0
    RING = 1
    MULE = 2
    ATO = 3
    CARD_TESTING = 4


PATTERN_NAMES: dict[int, str] = {
    Pattern.RING: "fraud_ring",
    Pattern.MULE: "money_mule",
    Pattern.ATO: "account_takeover",
    Pattern.CARD_TESTING: "card_testing",
}


@dataclass(frozen=True)
class SimConfig:
    """Every knob of one simulated world. ``seed`` plus these values determine the output exactly."""

    seed: int = 42
    n_days: int = 90
    n_accounts: int = 5_000
    n_merchants: int = 400
    new_account_fraction: float = 0.12
    household_fraction: float = 0.15
    purchase_rate_median: float = 0.8
    purchase_rate_sigma: float = 0.6
    favourite_share: float = 0.85
    transfer_rate: float = 0.04
    second_device_fraction: float = 0.25
    device_change_fraction: float = 0.25
    night_owl_fraction: float = 0.05
    big_ticket_probability: float = 0.02
    workplace_fraction: float = 0.03
    n_rent_collections: int = 200
    n_group_payments: int = 300
    n_trips: int = 300
    n_sprees: int = 400
    n_app_bursts: int = 300
    n_forwards: int = 250
    landlord_fraction: float = 0.01
    n_big_transfers: int = 200
    n_rings: int = 12
    ring_size_min: int = 4
    ring_size_max: int = 8
    mule_fraction: float = 0.015
    n_mule_chains: int = 130
    n_ato: int = 110
    n_ato_devices: int = 30
    n_card_tests: int = 24
    unreported_fraction: float = 0.03

    @classmethod
    def small(cls, seed: int = 7) -> SimConfig:
        """A world small enough for unit tests (about 20 thousand transactions)."""
        return cls(
            seed=seed,
            n_days=40,
            n_accounts=700,
            n_merchants=120,
            n_rent_collections=25,
            n_group_payments=30,
            n_trips=30,
            n_sprees=40,
            n_app_bursts=30,
            n_forwards=30,
            n_big_transfers=25,
            n_rings=4,
            mule_fraction=0.03,
            n_mule_chains=24,
            n_ato=20,
            n_ato_devices=6,
            n_card_tests=6,
        )

    @classmethod
    def quick(cls, seed: int = 42) -> SimConfig:
        """The CI profile: about a third of the full world over 60 days, same behaviours and patterns."""
        return cls(
            seed=seed,
            n_days=60,
            n_accounts=2_500,
            n_merchants=250,
            n_rent_collections=70,
            n_group_payments=100,
            n_trips=100,
            n_sprees=130,
            n_app_bursts=100,
            n_forwards=80,
            n_big_transfers=70,
            n_rings=6,
            mule_fraction=0.02,
            n_mule_chains=50,
            n_ato=50,
            n_ato_devices=15,
            n_card_tests=10,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Plain dict for JSON reports."""
        return asdict(self)
