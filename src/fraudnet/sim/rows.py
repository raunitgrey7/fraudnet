"""Column layout of the transaction log and a small row builder for event-driven generators.

The log is a set of aligned numpy columns rather than a dataframe: the feature store replays it row by row
and the evaluation slices it by time, and both are faster and simpler on plain arrays.
"""

from __future__ import annotations

import numpy as np

from fraudnet.sim.config import NEVER

COLUMN_DTYPES: dict[str, type] = {
    "ts": np.int64,
    "account": np.int32,
    "device": np.int32,
    "merchant": np.int32,
    "dst": np.int32,
    "amount": np.float64,
    "channel": np.int8,
    "pattern": np.int8,
    "attack_id": np.int32,
    "report_ts": np.int64,
}


def empty_columns() -> dict[str, np.ndarray]:
    """Zero-length columns with the canonical dtypes."""
    return {name: np.zeros(0, dtype=dtype) for name, dtype in COLUMN_DTYPES.items()}


def concat_columns(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate several column blocks, casting each column to its canonical dtype."""
    return {
        name: np.concatenate([np.asarray(p[name], dtype=dtype) for p in parts]).astype(dtype, copy=False)
        for name, dtype in COLUMN_DTYPES.items()
    }


class TxnRows:
    """Append-only buffer for generators that emit transactions one at a time."""

    def __init__(self) -> None:
        self._cols: dict[str, list] = {name: [] for name in COLUMN_DTYPES}

    def __len__(self) -> int:
        return len(self._cols["ts"])

    def add(
        self,
        ts: int,
        account: int,
        device: int,
        amount: float,
        channel: int,
        *,
        merchant: int = -1,
        dst: int = -1,
        pattern: int = 0,
        attack_id: int = -1,
        report_ts: int = NEVER,
    ) -> int:
        """Append one transaction and return its position in this buffer."""
        cols = self._cols
        cols["ts"].append(int(ts))
        cols["account"].append(int(account))
        cols["device"].append(int(device))
        cols["merchant"].append(int(merchant))
        cols["dst"].append(int(dst))
        cols["amount"].append(round(float(amount), 2))
        cols["channel"].append(int(channel))
        cols["pattern"].append(int(pattern))
        cols["attack_id"].append(int(attack_id))
        cols["report_ts"].append(int(report_ts))
        return len(cols["ts"]) - 1

    def set_report_ts(self, positions: list[int], report_ts: list[int]) -> None:
        """Fill in label report times once an attack's discovery time is known."""
        for pos, value in zip(positions, report_ts, strict=True):
            self._cols["report_ts"][pos] = int(value)

    def ts_of(self, position: int) -> int:
        """Timestamp of a row already added."""
        return int(self._cols["ts"][position])

    def columns(self) -> dict[str, np.ndarray]:
        """The buffer as typed numpy columns."""
        if not self._cols["ts"]:
            return empty_columns()
        return {name: np.asarray(values, dtype=COLUMN_DTYPES[name]) for name, values in self._cols.items()}
