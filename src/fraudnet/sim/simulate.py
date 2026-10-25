"""Orchestrates one simulation: world, genuine behaviour, look-alikes, planted fraud, label loss, ordering.

A single ``numpy.random.Generator`` seeded from the config drives every draw in a fixed order, which is what
makes the output byte-identical for a given seed (tested in ``tests/test_sim.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fraudnet.sim.config import DAY, NEVER, Pattern, SimConfig
from fraudnet.sim.confounders import all_confounders
from fraudnet.sim.dataset import Dataset, cache_path
from fraudnet.sim.legit import legit_purchases, legit_transfers
from fraudnet.sim.patterns import FraudInjector
from fraudnet.sim.rows import TxnRows, concat_columns
from fraudnet.sim.world import World, build_world

logger = logging.getLogger(__name__)


def _drop_unreported(cols: dict[str, np.ndarray], rng: np.random.Generator, fraction: float) -> None:
    """Some fraud is never reported (the customer never notices); its label never arrives."""
    fraud = np.flatnonzero(cols["pattern"] > Pattern.LEGIT)
    lost = fraud[rng.random(fraud.size) < fraction]
    cols["report_ts"][lost] = NEVER


def _account_table(world: World) -> dict[str, np.ndarray]:
    signup = world.all_signup_ts()
    is_ring = np.zeros(signup.size, dtype=np.int8)
    is_ring[world.n_legit_accounts :] = 1
    mule = np.concatenate([world.mule_recruit_ts, np.full(signup.size - world.n_legit_accounts, NEVER)])
    return {"signup_ts": signup, "is_ring": is_ring, "mule_recruit_ts": mule.astype(np.int64)}


def simulate(cfg: SimConfig) -> Dataset:
    """Generate the full world for ``cfg``; deterministic in ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    world = build_world(cfg, rng)
    purchases = legit_purchases(world, rng)
    transfers = legit_transfers(world, rng)
    events = TxnRows()
    all_confounders(world, rng, events)
    fraud = TxnRows()
    FraudInjector(world, rng, fraud).run_all()
    cols = concat_columns([purchases, transfers, events.columns(), fraud.columns()])
    _drop_unreported(cols, rng, cfg.unreported_fraction)
    inside = cols["ts"] < cfg.n_days * DAY
    order = np.lexsort((rng.permutation(int(inside.sum())), cols["ts"][inside]))
    cols = {name: values[inside][order] for name, values in cols.items()}
    merchants = {"category": world.all_merchant_category(), "created_ts": world.all_merchant_created()}
    return Dataset(cfg, cols, _account_table(world), merchants, world.n_devices)


def load_or_simulate(cfg: SimConfig, data_dir: Path | None) -> Dataset:
    """Return the cached dataset for ``cfg`` or simulate and cache it."""
    if data_dir is None:
        return simulate(cfg)
    path = cache_path(cfg, data_dir)
    if path.exists():
        return Dataset.load(path)
    dataset = simulate(cfg)
    dataset.save(path)
    logger.info("simulated dataset cached", extra={"path": str(path), "transactions": len(dataset)})
    return dataset
