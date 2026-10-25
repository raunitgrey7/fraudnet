"""Alert queue for analysts, stored in SQLite.

Every transaction scored as ``review`` or ``block`` becomes an alert. Analysts work the queue highest score
first and resolve each alert as fraud or genuine; a fraud resolution is fed back to the feature store as a
fraud report at the current clock, so later transactions on the same device or account see it (and earlier
ones never do). SQLite keeps the queue durable when a file path is configured and costs nothing in tests
(``:memory:``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import weakref
from typing import Any, Literal

Status = Literal["open", "fraud", "legit"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    account INTEGER NOT NULL,
    device INTEGER NOT NULL,
    merchant INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    amount REAL NOT NULL,
    score REAL NOT NULL,
    model TEXT NOT NULL,
    decision TEXT NOT NULL,
    reasons TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_ts INTEGER,
    true_fraud INTEGER
);
CREATE INDEX IF NOT EXISTS alerts_status_score ON alerts (status, score DESC);
"""
COLUMNS = (
    "alert_id", "txn_id", "ts", "account", "device", "merchant", "dst", "amount", "score", "model",
    "decision", "reasons", "status", "resolved_ts", "true_fraud",
)  # fmt: skip


class AlertQueue:
    """Thread-safe alert storage with the few queries the API needs."""

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._finalizer = weakref.finalize(self, self._conn.close)

    def close(self) -> None:
        """Close the database connection (also done automatically when the queue is garbage collected)."""
        self._finalizer()

    def add(self, record: dict[str, Any]) -> int:
        """Insert one alert and return its id."""
        values = (
            str(record["txn_id"]), int(record["ts"]), int(record["account"]), int(record["device"]),
            int(record["merchant"]), int(record["dst"]), float(record["amount"]), float(record["score"]),
            str(record["model"]), str(record["decision"]), json.dumps(record["reasons"]),
            None if record.get("true_fraud") is None else int(record["true_fraud"]),
        )  # fmt: skip
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO alerts (txn_id, ts, account, device, merchant, dst, amount, score, model, decision,"
                " reasons, true_fraud) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def _row(self, row: tuple) -> dict[str, Any]:
        item = dict(zip(COLUMNS, row, strict=True))
        item["reasons"] = json.loads(item["reasons"])
        return item

    def list(self, status: Status | None = "open", limit: int = 50) -> list[dict[str, Any]]:
        """Alerts with ``status`` (all when None), highest score first."""
        query = f"SELECT {', '.join(COLUMNS)} FROM alerts"
        params: tuple = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY score DESC, alert_id ASC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(query, (*params, limit)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, alert_id: int) -> dict[str, Any] | None:
        """One alert by id."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
        return self._row(row) if row else None

    def resolve(
        self, alert_id: int, label: Literal["fraud", "legit"], resolved_ts: int
    ) -> dict[str, Any] | None:
        """Close an open alert with the analyst's verdict; returns None for unknown or closed alerts."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET status = ?, resolved_ts = ? WHERE alert_id = ? AND status = 'open'",
                (label, resolved_ts, alert_id),
            )
            self._conn.commit()
            changed = cur.rowcount
        return self.get(alert_id) if changed else None

    def stats(self) -> dict[str, int]:
        """Counts by status plus how many alerts were planted fraud (known only for simulated replays)."""
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status").fetchall()
            truth = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(true_fraud), 0) FROM alerts WHERE true_fraud IS NOT NULL"
            ).fetchone()
        counts = {"open": 0, "fraud": 0, "legit": 0}
        counts.update({str(s): int(n) for s, n in rows})
        counts["replayed_alerts"] = int(truth[0])
        counts["replayed_alerts_true_fraud"] = int(truth[1])
        return counts
