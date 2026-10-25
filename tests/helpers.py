"""Build tiny hand-written transaction logs for unit tests of feature semantics."""

from __future__ import annotations

import numpy as np

from fraudnet.sim import Dataset, SimConfig
from fraudnet.sim.config import NEVER
from fraudnet.sim.rows import COLUMN_DTYPES


def make_dataset(
    rows: list[dict], n_accounts: int = 20, n_merchants: int = 10, signup: int = -100 * 86_400
) -> Dataset:
    """A dataset from dicts with keys ts, account, device and optional merchant, dst, amount, channel, pattern, report_ts."""
    defaults = {
        "merchant": 0,
        "dst": -1,
        "amount": 10.0,
        "channel": 1,
        "pattern": 0,
        "attack_id": -1,
        "report_ts": NEVER,
    }
    full = [{**defaults, **r} for r in rows]
    for r in full:
        if r["dst"] >= 0:
            r["merchant"] = -1
            r["channel"] = 2
    full.sort(key=lambda r: r["ts"])
    txns = {name: np.array([r[name] for r in full], dtype=dtype) for name, dtype in COLUMN_DTYPES.items()}
    accounts = {
        "signup_ts": np.full(n_accounts, signup, dtype=np.int64),
        "is_ring": np.zeros(n_accounts, dtype=np.int8),
        "mule_recruit_ts": np.full(n_accounts, NEVER, dtype=np.int64),
    }
    merchants = {
        "category": np.arange(n_merchants, dtype=np.int8) % 10,
        "created_ts": np.zeros(n_merchants, dtype=np.int64),
    }
    return Dataset(SimConfig.small(), txns, accounts, merchants, n_devices=1000)
