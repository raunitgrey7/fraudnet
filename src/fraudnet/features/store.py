"""Point-in-time online feature store, shared by offline training and online scoring.

Training features are produced by replaying the transaction log through this same object, one event at a
time, so training and serving cannot drift apart. Three rules keep every feature point-in-time correct:

1. ``features`` is a pure read of state built from earlier events; ``ingest`` mutates state afterwards.
2. A fraud report is a scheduled event with its own timestamp. It reaches the counters only when the stream
   clock passes its report time, never at the time of the fraudulent transaction.
3. Graph features come from the last snapshot, built at a refresh boundary from events before it.

``tests/test_point_in_time.py`` proves the rules hold: replaying a prefix of the log (with only the reports
known at the cut) reproduces the full replay's features exactly, and the same check fails on leaky variants.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraudnet.features.graph import (
    GraphSnapshot,
    WindowEdges,
    build_snapshot,
    empty_snapshot,
    window_accounts,
)
from fraudnet.sim.config import DAY, HOUR, TRANSFER_CATEGORY

WEEK = 7 * DAY
SMALL_AMOUNT = 5.0
MERCHANT_PRIOR = 10
NO_HISTORY = math.log1p(30 * DAY)


@dataclass(frozen=True)
class StoreConfig:
    """Snapshot cadence and window of the batch graph features."""

    snapshot_interval: int = DAY
    graph_window: int = 30 * DAY


@dataclass(frozen=True, order=True)
class LabelEvent:
    """A fraud report: which transaction, the entities it touched, and when the report arrived."""

    report_ts: int
    txn_id: int
    account: int
    device: int
    merchant: int
    dst: int


class AccountState:
    """Rolling history of one account."""

    __slots__ = (
        "devices", "inflows", "known_fraud", "last_inflow_ts", "last_ts", "m2", "max_log", "mean",
        "merchants", "n", "payees", "recent", "senders", "signup_ts",
    )  # fmt: skip

    def __init__(self, signup_ts: int) -> None:
        self.signup_ts = signup_ts
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.max_log = 0.0
        self.last_ts = -1
        self.last_inflow_ts = -1
        self.recent: deque[tuple[int, float, int, bool]] = deque()
        self.inflows: deque[tuple[int, float]] = deque()
        self.senders: deque[tuple[int, int]] = deque()
        self.devices: dict[int, int] = {}
        self.merchants: set[int] = set()
        self.payees: set[int] = set()
        self.known_fraud = 0


class DeviceState:
    """Accounts seen on one device and its recent use."""

    __slots__ = ("accounts", "first_ts", "known_fraud", "recent")

    def __init__(self, first_ts: int) -> None:
        self.first_ts = first_ts
        self.accounts: set[int] = set()
        self.recent: deque[tuple[int, int]] = deque()
        self.known_fraud = 0


class MerchantState:
    """Last hour of activity at one merchant."""

    __slots__ = ("known_fraud", "n", "recent")

    def __init__(self) -> None:
        self.n = 0
        self.recent: deque[tuple[int, bool, bool]] = deque()
        self.known_fraud = 0


class EdgeLog:
    """Transactions inside the graph window, appended as plain lists and trimmed at each refresh."""

    def __init__(self) -> None:
        self.cols: dict[str, list] = {k: [] for k in ("ts", "account", "device", "merchant", "dst", "amount")}

    def append(self, ts: int, account: int, device: int, merchant: int, dst: int, amount: float) -> None:
        """Record one ingested transaction."""
        c = self.cols
        c["ts"].append(ts)
        c["account"].append(account)
        c["device"].append(device)
        c["merchant"].append(merchant)
        c["dst"].append(dst)
        c["amount"].append(amount)

    def window(self, start: int) -> WindowEdges:
        """Edges with ``ts >= start``; older edges are dropped from the log."""
        ts = np.asarray(self.cols["ts"], dtype=np.int64)
        keep = int(np.searchsorted(ts, start, side="left"))
        if keep:
            for values in self.cols.values():
                del values[:keep]
        arrays = {k: np.asarray(v) for k, v in self.cols.items()}
        return WindowEdges(
            ts=arrays["ts"].astype(np.int64),
            account=arrays["account"].astype(np.int64),
            device=arrays["device"].astype(np.int64),
            merchant=arrays["merchant"].astype(np.int64),
            dst=arrays["dst"].astype(np.int64),
            amount=arrays["amount"].astype(np.float64),
        )


def _prune(window: deque, now: int, horizon: int) -> None:
    while window and now - window[0][0] >= horizon:
        window.popleft()


class OnlineFeatureStore:
    """Streaming feature state for accounts, devices and merchants, plus the latest graph snapshot."""

    def __init__(
        self,
        signup_ts: np.ndarray,
        merchant_category: np.ndarray,
        config: StoreConfig | None = None,
        on_snapshot: Callable[[GraphSnapshot], None] | None = None,
    ) -> None:
        self.config = config or StoreConfig()
        self.signup_ref = signup_ts
        self.category_ref = merchant_category
        self.on_snapshot = on_snapshot
        self.accounts: dict[int, AccountState] = {}
        self.devices: dict[int, DeviceState] = {}
        self.merchants: dict[int, MerchantState] = {}
        self.fraud_accounts: set[int] = set()
        self.fraud_devices: set[int] = set()
        self.pending: list[LabelEvent] = []
        self.edges = EdgeLog()
        self.clock = -(1 << 62)
        self.snapshot = empty_snapshot(0)
        self.next_refresh: int | None = None
        self.labels_applied = 0
        self.snapshots_built = 0

    def __getstate__(self) -> dict[str, object]:
        """Pickle without the snapshot callback (it belongs to whoever owns the store at run time)."""
        state = dict(self.__dict__)
        state["on_snapshot"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

    # Reference data ----------------------------------------------------------------------------------------

    def signup_of(self, account: int, ts: int) -> int:
        """Signup time from the reference table; an account unknown to it counts as new at first sight."""
        if 0 <= account < self.signup_ref.size:
            return int(self.signup_ref[account])
        state = self.accounts.get(account)
        return state.signup_ts if state is not None else ts

    def category_of(self, merchant: int) -> int:
        """Merchant category code; transfers use ``TRANSFER_CATEGORY``."""
        if merchant < 0:
            return TRANSFER_CATEGORY
        if merchant < self.category_ref.size:
            return int(self.category_ref[merchant])
        return 0

    # Clock, labels and snapshots ---------------------------------------------------------------------------

    def schedule_label(self, event: LabelEvent) -> None:
        """Queue a fraud report; it becomes visible once the clock passes ``event.report_ts``."""
        heapq.heappush(self.pending, event)

    def advance_to(self, ts: int) -> None:
        """Move the clock to ``ts``: apply reports and refresh snapshots in time order.

        A report at exactly ``ts`` is not applied yet: features of an event may only use reports strictly
        earlier than the event.
        """
        if ts < self.clock:
            raise ValueError(f"event at {ts} is older than the store clock {self.clock}")
        if self.next_refresh is None:
            self.next_refresh = ts // self.config.snapshot_interval * self.config.snapshot_interval
        while self.next_refresh <= ts:
            self._apply_labels_before(self.next_refresh)
            self._refresh(self.next_refresh)
            self.next_refresh += self.config.snapshot_interval
        self._apply_labels_before(ts)
        self.clock = ts

    def _apply_labels_before(self, ts: int) -> None:
        while self.pending and self.pending[0].report_ts < ts:
            self._apply_label(heapq.heappop(self.pending))

    def _apply_label(self, event: LabelEvent) -> None:
        self.labels_applied += 1
        acct = self.accounts.get(event.account)
        if acct is not None:
            acct.known_fraud += 1
            self.fraud_accounts.add(event.account)
        dev = self.devices.get(event.device)
        if dev is not None:
            dev.known_fraud += 1
            self.fraud_devices.add(event.device)
        if event.merchant >= 0 and (mer := self.merchants.get(event.merchant)) is not None:
            mer.known_fraud += 1
        if event.dst >= 0 and (dst := self.accounts.get(event.dst)) is not None:
            dst.known_fraud += 1
            self.fraud_accounts.add(event.dst)

    def rebuild_snapshot(self, ts: int) -> GraphSnapshot:
        """Build the snapshot for ``ts`` from the current window now (used to time the daily graph job)."""
        self._refresh(ts)
        return self.snapshot

    def _refresh(self, boundary: int) -> None:
        edges = self.edges.window(boundary - self.config.graph_window)
        accounts = window_accounts(edges)
        signup = np.array([self.signup_of(int(a), boundary) for a in accounts], dtype=np.int64)
        self.snapshot = build_snapshot(
            boundary, edges, signup, self.fraud_accounts, self.fraud_devices, self.category_ref
        )
        self.snapshots_built += 1
        if self.on_snapshot is not None:
            self.on_snapshot(self.snapshot)

    # Features ----------------------------------------------------------------------------------------------

    def features(
        self, ts: int, account: int, device: int, merchant: int, dst: int, amount: float, channel: int
    ) -> list[float]:
        """Every feature of one transaction, in schema order, from state strictly before it."""
        a = self.accounts.get(account)
        d = self.devices.get(device)
        out = self._own(ts, a, account, device, merchant, dst, amount, channel)
        out += self._shared_device(ts, a, d)
        out += self._counterparty(ts, merchant, dst)
        out += self._risk(a, d, merchant, dst)
        out += self.snapshot.lookup(account, device, merchant, dst)
        return out

    def _own(
        self,
        ts: int,
        a: AccountState | None,
        account: int,
        device: int,
        merchant: int,
        dst: int,
        amount: float,
        channel: int,
    ) -> list[float]:
        """Amount, time, account, velocity, context, device-account and flow groups."""
        log_amt = math.log1p(amount)
        hour = (ts % DAY) / HOUR * (2.0 * math.pi / 24.0)
        age = math.log1p(max(0, ts - self.signup_of(account, ts)) / DAY)
        context = [float(channel), float(self.category_of(merchant))]
        if a is None or a.n == 0:
            history = [
                log_amt,
                0.0,
                0.0,
                math.sin(hour),
                math.cos(hour),
                age,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                NO_HISTORY,
            ]
            context += [
                1.0 if merchant >= 0 else 0.0,
                1.0 if a is None or device not in a.devices else 0.0,
                0.0,
            ]
        else:
            std = math.sqrt(a.m2 / a.n) if a.n > 1 else 0.0
            z = (log_amt - a.mean) / max(std, 0.25) if a.n >= 3 else 0.0
            c1, c24, c7, s24, n_devs = self._account_window(ts, a)
            history = [
                log_amt, z, log_amt - a.max_log, math.sin(hour), math.cos(hour), age, math.log1p(a.n),
                float(c1), float(c24), float(c7), math.log1p(s24), math.log1p(ts - a.last_ts),
            ]  # fmt: skip
            new_merchant = 1.0 if merchant >= 0 and merchant not in a.merchants else 0.0
            context += [new_merchant, 0.0 if device in a.devices else 1.0, float(n_devs)]
        return history + context + self._flow(ts, a, dst, amount)

    @staticmethod
    def _account_window(ts: int, a: AccountState) -> tuple[int, int, int, float, int]:
        """Counts in the last hour, day and week, spend in the last day, distinct devices in the week."""
        c1 = c24 = c7 = 0
        s24 = 0.0
        devs: set[int] = set()
        for t, amt, dev, _ in reversed(a.recent):
            gap = ts - t
            if gap >= WEEK:
                break
            c7 += 1
            devs.add(dev)
            if gap < DAY:
                c24 += 1
                s24 += amt
                if gap < HOUR:
                    c1 += 1
        return c1, c24, c7, s24, len(devs)

    @staticmethod
    def _flow(ts: int, a: AccountState | None, dst: int, amount: float) -> list[float]:
        """New payee, money in and out over the last day, pass-through ratio and time since money arrived."""
        new_payee = 1.0 if dst >= 0 and (a is None or dst not in a.payees) else 0.0
        if a is None:
            return [new_payee, 0.0, 0.0, 0.0, NO_HISTORY]
        inflow = 0.0
        for t, amt in reversed(a.inflows):
            if ts - t >= DAY:
                break
            inflow += amt
        outflow = 0.0
        for t, amt, _, is_transfer in reversed(a.recent):
            if ts - t >= DAY:
                break
            if is_transfer:
                outflow += amt
        passthrough = min(amount / inflow, 5.0) if inflow > 0 else 0.0
        since_in = math.log1p(ts - a.last_inflow_ts) if a.last_inflow_ts >= 0 else NO_HISTORY
        return [new_payee, math.log1p(inflow), math.log1p(outflow), passthrough, since_in]

    def _shared_device(self, ts: int, a: AccountState | None, d: DeviceState | None) -> list[float]:
        """Shared-device counts: how many accounts touch this device and this account's devices."""
        n_dev = float(len(a.devices)) if a is not None else 0.0
        neighbours = 0
        if a is not None:
            for dev in a.devices:
                neighbours += len(self.devices[dev].accounts) - 1
        if d is None:
            return [0.0, 0.0, 0.0, 0.0, n_dev, math.log1p(neighbours)]
        recent: set[int] = set()
        c1 = 0
        for t, acct in reversed(d.recent):
            gap = ts - t
            if gap >= DAY:
                break
            recent.add(acct)
            if gap < HOUR:
                c1 += 1
        age = math.log1p((ts - d.first_ts) / DAY)
        return [float(len(d.accounts)), float(len(recent)), float(c1), age, n_dev, math.log1p(neighbours)]

    def _counterparty(self, ts: int, merchant: int, dst: int) -> list[float]:
        """Last-hour activity at the merchant, or the payee's recent senders, age and inflow."""
        if merchant >= 0:
            m = self.merchants.get(merchant)
            if m is None:
                return [0.0] * 6
            n = small = new_dev = 0
            for t, is_small, is_new_dev in reversed(m.recent):
                if ts - t >= HOUR:
                    break
                n += 1
                small += is_small
                new_dev += is_new_dev
            return [float(n), float(small), new_dev / n if n else 0.0, 0.0, 0.0, 0.0]
        b = self.accounts.get(dst)
        age = math.log1p(max(0, ts - self.signup_of(dst, ts)) / DAY)
        if b is None:
            return [0.0, 0.0, 0.0, 0.0, age, 0.0]
        senders = {s for t, s in b.senders if ts - t < WEEK}
        inflow = sum(amt for t, amt in b.inflows if ts - t < DAY)
        return [0.0, 0.0, 0.0, float(len(senders)), age, math.log1p(inflow)]

    def _risk(self, a: AccountState | None, d: DeviceState | None, merchant: int, dst: int) -> list[float]:
        """Reported fraud on the device, the account, the counterparty and the account's other devices.

        For a merchant the counterparty value is a smoothed reported-fraud rate in percent, because raw counts
        mostly measure popularity; for a payee it is the log count of reports touching that account.
        """
        if merchant >= 0:
            m = self.merchants.get(merchant)
            cp = 100.0 * m.known_fraud / (m.n + MERCHANT_PRIOR) if m is not None else 0.0
        else:
            b = self.accounts.get(dst)
            cp = math.log1p(b.known_fraud) if b is not None else 0.0
        linked = sum(self.devices[x].known_fraud for x in a.devices) if a is not None else 0
        return [
            math.log1p(d.known_fraud) if d is not None else 0.0,
            math.log1p(a.known_fraud) if a is not None else 0.0,
            cp,
            math.log1p(linked),
        ]

    # Ingest ------------------------------------------------------------------------------------------------

    def ingest(self, ts: int, account: int, device: int, merchant: int, dst: int, amount: float) -> None:
        """Add one transaction to the state after its features were read."""
        a = self.accounts.get(account)
        if a is None:
            a = self.accounts[account] = AccountState(self.signup_of(account, ts))
        x = math.log1p(amount)
        a.n += 1
        delta = x - a.mean
        a.mean += delta / a.n
        a.m2 += delta * (x - a.mean)
        a.max_log = max(a.max_log, x)
        a.last_ts = ts
        a.recent.append((ts, amount, device, dst >= 0))
        _prune(a.recent, ts, WEEK)
        new_device = device not in a.devices
        if new_device:
            a.devices[device] = ts
        d = self.devices.get(device)
        if d is None:
            d = self.devices[device] = DeviceState(ts)
        d.accounts.add(account)
        d.recent.append((ts, account))
        _prune(d.recent, ts, DAY)
        if merchant >= 0:
            self._ingest_purchase(ts, a, merchant, amount, new_device)
        else:
            self._ingest_transfer(ts, a, account, dst, amount)
        self.edges.append(ts, account, device, merchant, dst, amount)

    def _ingest_purchase(
        self, ts: int, a: AccountState, merchant: int, amount: float, new_device: bool
    ) -> None:
        m = self.merchants.get(merchant)
        if m is None:
            m = self.merchants[merchant] = MerchantState()
        m.n += 1
        m.recent.append((ts, amount < SMALL_AMOUNT, new_device))
        _prune(m.recent, ts, HOUR)
        a.merchants.add(merchant)

    def _ingest_transfer(self, ts: int, a: AccountState, account: int, dst: int, amount: float) -> None:
        a.payees.add(dst)
        b = self.accounts.get(dst)
        if b is None:
            b = self.accounts[dst] = AccountState(self.signup_of(dst, ts))
        b.inflows.append((ts, amount))
        _prune(b.inflows, ts, DAY)
        b.senders.append((ts, account))
        _prune(b.senders, ts, WEEK)
        b.last_inflow_ts = ts

    def process(
        self, ts: int, account: int, device: int, merchant: int, dst: int, amount: float, channel: int
    ) -> list[float]:
        """Advance the clock, read features, then ingest: the online scoring path in one call."""
        self.advance_to(ts)
        row = self.features(ts, account, device, merchant, dst, amount, channel)
        self.ingest(ts, account, device, merchant, dst, amount)
        return row
