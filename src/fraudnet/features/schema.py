"""Feature names, their groups and the reason-code text attached to each group.

Groups serve two purposes. The benchmark's ablation drops whole groups (for example every PageRank column)
to measure what each family of graph features adds, and the scoring API turns the groups that pushed a score
up the most into reason codes an analyst can read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureGroup:
    """A family of features that is ablated and explained as one unit."""

    name: str
    graph: bool
    batch: bool
    reason_code: str
    description: str
    features: tuple[str, ...]


GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup(
        "amount", False, False, "AMOUNT_UNUSUAL", "amount far above this account's usual spend",
        ("log_amount", "amount_z", "amount_vs_max"),
    ),
    FeatureGroup("time", False, False, "UNUSUAL_HOUR", "time of day", ("hour_sin", "hour_cos")),
    FeatureGroup("account", False, False, "NEW_ACCOUNT", "young or thin account history", ("acct_age_days", "acct_txn_count")),
    FeatureGroup(
        "velocity", False, False, "VELOCITY_SPIKE", "burst of transactions on this account",
        ("acct_cnt_1h", "acct_cnt_24h", "acct_cnt_7d", "acct_sum_24h", "log_secs_since_last"),
    ),
    FeatureGroup(
        "context", False, False, "RISKY_CONTEXT", "channel, merchant category or first visit to this merchant",
        ("channel", "merchant_category", "merchant_new_for_acct"),
    ),
    FeatureGroup(
        "device_account", False, False, "NEW_DEVICE", "device never seen on this account",
        ("device_new_for_acct", "acct_devices_7d"),
    ),
    FeatureGroup(
        "flow", False, False, "PASS_THROUGH_FUNDS", "money received and sent on within hours, or a new payee",
        ("payee_new", "acct_inflow_24h", "acct_outflow_24h", "passthrough_ratio", "log_secs_since_inflow"),
    ),
    FeatureGroup(
        "shared_device", True, False, "SHARED_DEVICE", "device shared with many other accounts",
        ("device_accts_total", "device_accts_24h", "device_cnt_1h", "device_age_days", "acct_devices_total",
         "acct_device_neighbours"),
    ),
    FeatureGroup(
        "counterparty", True, False, "COUNTERPARTY_ACTIVITY", "unusual activity at the merchant or payee",
        ("merchant_cnt_1h", "merchant_small_cnt_1h", "merchant_new_device_share_1h", "dst_senders_7d",
         "dst_age_days", "dst_inflow_24h"),
    ),
    FeatureGroup(
        "risk", True, False, "LINKED_TO_KNOWN_FRAUD", "device, account or counterparty with reported fraud",
        ("device_known_fraud", "acct_known_fraud", "cp_known_fraud", "acct_devices_known_fraud"),
    ),
    FeatureGroup(
        "degree", True, True, "HIGH_DEGREE", "many distinct merchants, payees or accounts in the last 30 days",
        ("acct_deg_merchants", "acct_deg_counterparties", "device_deg", "cp_deg"),
    ),
    FeatureGroup(
        "pagerank", True, True, "NETWORK_RISK", "close to known fraud in the device and transfer graph",
        ("acct_pagerank", "device_pagerank", "cp_pagerank", "acct_ppr_risk", "device_ppr_risk", "cp_ppr_risk"),
    ),
    FeatureGroup(
        "community", True, True, "SUSPICIOUS_COMMUNITY", "member of a dense cluster of linked accounts",
        ("comm_size", "comm_devices", "comm_known_fraud", "comm_new_acct_frac", "comm_max_device_deg"),
    ),
)  # fmt: skip

FEATURE_NAMES: tuple[str, ...] = tuple(f for g in GROUPS for f in g.features)
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)
CATEGORICAL_FEATURES: tuple[str, ...] = ("channel", "merchant_category")
GROUP_BY_NAME: dict[str, FeatureGroup] = {g.name: g for g in GROUPS}
TABULAR_GROUPS: tuple[str, ...] = tuple(g.name for g in GROUPS if not g.graph)
GRAPH_GROUPS: tuple[str, ...] = tuple(g.name for g in GROUPS if g.graph)
STREAMING_GROUPS: tuple[str, ...] = tuple(g.name for g in GROUPS if not g.batch)
BATCH_GROUPS: tuple[str, ...] = tuple(g.name for g in GROUPS if g.batch)


def columns_for(groups: tuple[str, ...] | list[str]) -> list[int]:
    """Column indices of every feature in ``groups``, in schema order."""
    wanted = set(groups)
    unknown = wanted - set(GROUP_BY_NAME)
    if unknown:
        raise KeyError(f"unknown feature groups: {sorted(unknown)}")
    return [FEATURE_INDEX[f] for g in GROUPS if g.name in wanted for f in g.features]
