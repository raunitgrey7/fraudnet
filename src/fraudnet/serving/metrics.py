"""Prometheus collectors on a registry per application instance.

A registry per app avoids duplicate-registration errors when tests build several apps in one process.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Metrics:
    """Every collector the API exports."""

    registry: CollectorRegistry
    requests_total: Counter
    request_duration: Histogram
    score_latency: Histogram

    @classmethod
    def create(cls) -> Metrics:
        """Fresh collectors on a fresh registry."""
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests_total=Counter(
                "fraudnet_http_requests_total",
                "HTTP requests",
                ["method", "route", "status"],
                registry=registry,
            ),
            request_duration=Histogram(
                "fraudnet_http_request_duration_seconds",
                "HTTP request latency",
                ["method", "route"],
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
            score_latency=Histogram(
                "fraudnet_score_latency_seconds",
                "Feature lookup, scoring and reason codes",
                ["model"],
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
        )


class ServiceCollector(Collector):
    """Reads the scoring service's own counters at scrape time, so API and replay traffic are both counted."""

    def __init__(self, stats: Callable[[], dict[str, Any]]) -> None:
        self.stats = stats

    def collect(self) -> Iterator[Metric]:
        """Yield scored-transaction counters and store and queue gauges."""
        stats = self.stats()
        scored = CounterMetricFamily("fraudnet_scored", "Transactions scored", labels=["model", "decision"])
        for key, value in sorted(stats["scored"].items()):
            if ":" in key:
                model, decision = key.split(":", 1)
                scored.add_metric([model, decision], value)
        yield scored
        store = stats["store"]
        for name, key, text in (
            ("fraudnet_store_accounts", "accounts", "Accounts in the feature store"),
            ("fraudnet_store_devices", "devices", "Devices in the feature store"),
            ("fraudnet_labels_applied", "labels_applied", "Fraud reports applied to features"),
            ("fraudnet_labels_pending", "labels_pending", "Fraud reports scheduled for the future"),
            ("fraudnet_snapshots_built", "snapshots_built", "Graph snapshots built"),
        ):
            yield GaugeMetricFamily(name, text, value=float(store[key]))
        yield GaugeMetricFamily(
            "fraudnet_alerts_open",
            "Open alerts in the analyst queue",
            value=float(stats["alerts"].get("open", 0)),
        )
