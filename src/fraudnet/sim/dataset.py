"""The simulated dataset: the transaction log, reference tables, a checksum and an on-disk cache.

The cache key is a hash of the full config, so changing any simulator knob produces a new file instead of
silently reusing an old world, and the stored SHA-256 of every column is re-checked on load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fraudnet.sim.config import DAY, NEVER, PATTERN_NAMES, Pattern, SimConfig


@dataclass
class Dataset:
    """A world's transactions (sorted by time) plus the reference data a bank would hold."""

    config: SimConfig
    txns: dict[str, np.ndarray]
    accounts: dict[str, np.ndarray]
    merchants: dict[str, np.ndarray]
    n_devices: int

    def __len__(self) -> int:
        return int(self.txns["ts"].shape[0])

    @property
    def is_fraud(self) -> np.ndarray:
        """Ground truth: 1 for every planted fraud transaction."""
        return (self.txns["pattern"] > Pattern.LEGIT).astype(np.int8)

    def labels_as_of(self, t: int) -> np.ndarray:
        """Labels a bank would hold at time ``t``: fraud only once the report has arrived."""
        return ((self.txns["pattern"] > Pattern.LEGIT) & (self.txns["report_ts"] <= t)).astype(np.int8)

    @property
    def day(self) -> np.ndarray:
        """Day index of every transaction."""
        return self.txns["ts"] // DAY

    def checksum(self) -> str:
        """SHA-256 over every column, used by the determinism tests and the cache."""
        digest = hashlib.sha256()
        for table in (self.txns, self.accounts, self.merchants):
            for name in sorted(table):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(table[name]).tobytes())
        digest.update(str(self.n_devices).encode())
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        """Counts, fraud rate, per-pattern volumes and label delay quantiles."""
        fraud = self.is_fraud.astype(bool)
        pattern = self.txns["pattern"]
        report = self.txns["report_ts"]
        reported = fraud & (report < NEVER)
        delays = (report[reported] - self.txns["ts"][reported]) / DAY
        per_pattern = {}
        for code, name in PATTERN_NAMES.items():
            mask = pattern == code
            per_pattern[name] = {
                "transactions": int(mask.sum()),
                "attacks": int(np.unique(self.txns["attack_id"][mask]).size),
                "amount": round(float(self.txns["amount"][mask].sum()), 2),
            }
        return {
            "transactions": len(self),
            "accounts": int(self.accounts["signup_ts"].shape[0]),
            "devices": self.n_devices,
            "merchants": int(self.merchants["category"].shape[0]),
            "fraud_transactions": int(fraud.sum()),
            "fraud_rate": round(float(fraud.mean()), 6),
            "fraud_amount_share": round(
                float(self.txns["amount"][fraud].sum() / self.txns["amount"].sum()), 6
            ),
            "never_reported_share": round(float(1.0 - reported.sum() / max(1, fraud.sum())), 4),
            "label_delay_days": {
                q: round(float(np.quantile(delays, p)), 2) if delays.size else None
                for q, p in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
            },
            "patterns": per_pattern,
        }

    def save(self, path: Path) -> None:
        """Write a compressed archive with the config and checksum embedded."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"txn_{k}": v for k, v in self.txns.items()}
        arrays |= {f"acc_{k}": v for k, v in self.accounts.items()}
        arrays |= {f"mer_{k}": v for k, v in self.merchants.items()}
        meta = {"config": self.config.to_dict(), "n_devices": self.n_devices, "sha256": self.checksum()}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: Path) -> Dataset:
        """Read an archive written by ``save`` and verify its checksum."""
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            tables: dict[str, dict[str, np.ndarray]] = {"txn": {}, "acc": {}, "mer": {}}
            for key in archive.files:
                prefix, _, name = key.partition("_")
                if prefix in tables:
                    tables[prefix][name] = archive[key]
        dataset = cls(
            SimConfig(**meta["config"]), tables["txn"], tables["acc"], tables["mer"], int(meta["n_devices"])
        )
        if dataset.checksum() != meta["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")
        return dataset


def config_key(cfg: SimConfig) -> str:
    """Short stable hash of a config, used in cache file names."""
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def cache_path(cfg: SimConfig, data_dir: Path) -> Path:
    """Where the dataset for ``cfg`` is cached."""
    return data_dir / f"world_seed{cfg.seed}_{config_key(cfg)}.npz"
