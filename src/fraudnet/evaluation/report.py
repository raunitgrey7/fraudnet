"""Render benchmark results as Markdown, copy them into the README, and enforce the CI thresholds.

The README tables are generated from ``benchmarks/results.json`` and pasted between markers, and
``tests/test_docs_consistency.py`` fails when the two disagree, so a hand-edited number cannot survive.
Threshold paths use ``/`` as the separator because metric names such as ``recall_at_0.5pct_fpr`` contain dots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
MODEL_ORDER = ("rules", "gbdt_tabular", "gbdt_streaming", "gbdt", "graphsage", "gbdt+graphsage")
PATTERNS = ("fraud_ring", "money_mule", "account_takeover", "card_testing")


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _money(value: float) -> str:
    return f"{value:,.0f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _models(results: dict[str, Any]) -> list[str]:
    return [m for m in MODEL_ORDER if m in results["models"]]


def header_line(results: dict[str, Any]) -> str:
    """Provenance of a results file in one sentence."""
    m = results["meta"]
    return (
        f"Seed {m['seed']}, {m['profile']} profile, generated {m['generated_at']}, commit `{m['commit']}`, Python {m['python']}, "
        f"torch {m['torch']}, scikit-learn {m['scikit_learn']} on {m['machine']} with {m['threads']} threads. "
        f"Total benchmark time {m['seconds']['total']} s."
    )


def detection_table(results: dict[str, Any]) -> list[str]:
    """Headline detection metrics with day-block bootstrap intervals."""
    rows = []
    for name in _models(results):
        r = results["models"][name]
        mt, ci = r["metrics"], r["ci95"]
        rows.append([
            name,
            f"{_f(mt['pr_auc'])} [{_f(ci['pr_auc'][0], 3)}, {_f(ci['pr_auc'][1], 3)}]",
            _f(mt["roc_auc"]),
            _f(mt["recall_at_0.5pct_fpr"]),
            f"{_f(mt['recall_at_1pct_fpr'])} [{_f(ci['recall_at_1pct_fpr'][0], 3)}, {_f(ci['recall_at_1pct_fpr'][1], 3)}]",
            _f(mt["precision_at_10_per_day"], 3),
            _f(mt["precision_at_50_per_day"], 3),
        ])  # fmt: skip
    header = [
        "model",
        "PR-AUC [95% CI]",
        "ROC-AUC",
        "recall at 0.5% FPR",
        "recall at 1% FPR [95% CI]",
        "precision top-10/day",
        "precision top-50/day",
    ]
    return _table(header, rows)


def money_table(results: dict[str, Any]) -> list[str]:
    """Money saved at the validation-chosen threshold and with a fixed budget of alerts per day."""
    rows = []
    budget_key = next(k for k in results["models"][_models(results)[0]]["money"] if k.startswith("budget_"))
    budget = budget_key.split("_")[1]
    for name in _models(results):
        mo = results["models"][name]["money"]
        b = mo[budget_key]
        rows.append([
            name, _f(mo["alerts_per_day"], 1), _f(mo["alert_precision"], 3), _f(mo["fraud_recall"], 3), _money(mo["money_saved"]),
            _f(mo["share_of_loss_saved"], 3), _f(b["alert_precision"], 3), _f(b["fraud_recall"], 3), _money(b["money_saved"]),
        ])  # fmt: skip
    first = results["models"][_models(results)[0]]["money"]
    cost = results["meta"]["cost_model"]
    header = [
        "model", "alerts/day", "alert precision", "fraud recall", "money saved", "share of loss saved",
        f"precision at {budget}/day", f"recall at {budget}/day", f"money saved at {budget}/day",
    ]  # fmt: skip
    lines = _table(header, rows)
    lines.append(
        f"Cost model: {cost['review_cost']:.0f} per alert reviewed, {cost['false_alert_cost']:.0f} more when the alert is a genuine payment, "
        f"a missed fraud costs its amount plus a {cost['chargeback_fee']:.0f} chargeback fee. The first block uses the threshold that "
        f"maximised money saved on validation; the second gives analysts exactly {budget} alerts per day. Fraud loss with no model: "
        f"{_money(first['loss_without_model'])}; a perfect detector would save {_money(first['oracle_saved'])}."
    )
    return [*lines, ""]


def pattern_table(results: dict[str, Any]) -> list[str]:
    """Per-pattern transaction recall and attack detection at the 1 percent FPR threshold."""
    rows = []
    for name in _models(results):
        per = results["models"][name]["per_pattern_at_1pct_fpr"]
        rows.append(
            [name] + [f"{_f(per[p]['recall'], 3)} / {_f(per[p]['attacks_detected'], 2)}" for p in PATTERNS]
        )
    counts = results["models"][_models(results)[0]]["per_pattern_at_1pct_fpr"]
    lines = _table(["model", *PATTERNS], rows)
    sizes = ", ".join(
        f"{p} {counts[p]['transactions']} transactions in {counts[p]['attacks']} attacks" for p in PATTERNS
    )
    lines.append(
        f"Cells: transaction recall / share of attacks with at least one transaction caught. Test window: {sizes}."
    )
    return [*lines, ""]


def ablation_table(results: dict[str, Any]) -> list[str]:
    """Graph feature ablation of the boosted model."""
    rows = []
    for name, r in results["ablation"].items():
        pr = r["pattern_recall_at_1pct_fpr"]
        lo, hi = r["delta_vs_tabular_ci95"]
        change = f"{r['delta_pr_auc_vs_tabular']:+.4f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(
            [name, _f(r["pr_auc"]), change, _f(r["recall_at_1pct_fpr"])] + [_f(pr[p], 3) for p in PATTERNS]
        )
    lines = _table(
        ["variant", "PR-AUC", "change vs tabular [paired 95% CI]", "recall at 1% FPR", *PATTERNS], rows
    )
    lines.append(
        "Pattern columns are transaction recall at 1 percent FPR. Intervals resample test days and score both models on the same days."
    )
    return [*lines, ""]


def pit_table(results: dict[str, Any]) -> list[str]:
    """Point-in-time checker results for the honest and the leaky pipeline."""
    pit = results["point_in_time"]
    rows = []
    for pipeline in ("honest", "leaky_instant_labels"):
        for c in pit[pipeline]["checks"]:
            verdict = (
                "identical"
                if c["mismatched_values"] == 0
                else f"differs ({', '.join(c['mismatched_features'][:4])})"
            )
            rows.append(
                [
                    pipeline,
                    c["variant"],
                    str(c["rows"]),
                    f"{c['values']:,}",
                    f"{c['mismatched_values']:,}",
                    verdict,
                ]
            )
    lines = [
        f"First {pit['dataset_days']} days ({pit['transactions']:,} transactions), cut at day {pit['cut_day']:.2f}.",
        "",
    ]
    return lines + _table(
        ["pipeline", "check", "rows", "values compared", "mismatched values", "result"], rows
    )


def leakage_table(results: dict[str, Any]) -> list[str]:
    """What a label leak does to offline numbers and to production."""
    rows = [
        [name, _f(r["pr_auc"]), _f(r["recall_at_1pct_fpr"])] for name, r in results["leakage_cost"].items()
    ]
    return _table(["setting", "test PR-AUC", "recall at 1% FPR"], rows)


def latency_table(results: dict[str, Any]) -> list[str]:
    """Per-transaction latency in process and over HTTP."""
    lat = results["latency"]
    rows = [
        [
            name,
            _f(v["p50_ms"], 3),
            _f(v["p95_ms"], 3),
            _f(v["p99_ms"], 3),
            _f(v["max_ms"], 3),
            _f(v["throughput_per_s"], 1),
        ]
        for name, v in lat["in_process"].items()
    ]
    for side, label in (("server", "gbdt over HTTP, server side"), ("client", "gbdt over HTTP, client side")):
        h = lat["http"][side]
        rows.append(
            [
                label,
                _f(h["p50_ms"], 3),
                _f(h["p95_ms"], 3),
                _f(h["p99_ms"], 3),
                _f(h["max_ms"], 3),
                _f(h["throughput_per_s"], 1),
            ]
        )
    lines = _table(["path", "p50 ms", "p95 ms", "p99 ms", "max ms", "sequential txn/s"], rows)
    ref = lat["snapshot_refresh_s"]
    lines.append(
        f"{lat['transactions']} test transactions scored one at a time after {lat['warmup']} untimed warm-up transactions; "
        "each path covers feature lookup, scoring and reason codes unless marked no_reasons. "
        f"Online features equal the offline replay features (max absolute difference {lat['online_offline_feature_max_abs_diff']}). "
        f"A daily graph refresh over {ref['nodes']:,} nodes takes {ref['graph_features']} s plus {ref['gnn_embeddings']} s of GNN embedding."
    )
    return [*lines, ""]


def seeds_table(results: dict[str, Any]) -> list[str]:
    """Test PR-AUC per simulator seed."""
    seeds = results.get("seeds")
    if not seeds:
        return []
    models = [
        m for m in ("rules", "gbdt_tabular", "gbdt", "graphsage") if all(m in v for v in seeds.values())
    ]
    rows = [[seed] + [_f(v[m]["pr_auc"]) for m in models] for seed, v in seeds.items()]
    means = [f"{sum(v[m]['pr_auc'] for v in seeds.values()) / len(seeds):.4f}" for m in models]
    rows.append(["mean", *means])
    return _table(["seed", *models], rows)


def dataset_lines(results: dict[str, Any]) -> list[str]:
    """Dataset and split summary."""
    d = results["dataset"]
    lines = [
        f"{d['transactions']:,} transactions, {d['accounts']:,} accounts, {d['devices']:,} devices, {d['merchants']} merchants; "
        f"{d['fraud_transactions']:,} fraud ({d['fraud_rate'] * 100:.3f} percent); {d['never_reported_share'] * 100:.1f} percent of fraud is never reported; "
        f"median label delay {d['label_delay_days']['p50']} days.",
        "",
    ]
    rows = []
    for name in ("train", "val", "test"):
        s = d["split"][name]
        unl = s.get("fraud_unlabelled_at_build")
        rows.append(
            [
                name,
                f"{s['days'][0]} to {s['days'][1]}",
                f"{s['transactions']:,}",
                str(s["fraud"]),
                f"{s['fraud_rate'] * 100:.3f}%",
                "n/a" if unl is None else f"{unl * 100:.1f}%",
            ]
        )
    return lines + _table(
        ["window", "days", "transactions", "fraud", "fraud rate", "fraud not yet labelled at build"], rows
    )


def protocol_table(results: dict[str, Any]) -> list[str]:
    """With and without the label-maturity gap."""
    proto = results.get("protocol")
    if not proto:
        return []
    rows = []
    for model in ("gbdt", "graphsage"):
        for variant in ("with_gap", "no_gap"):
            r = proto[model][variant]
            rows.append(
                [
                    model,
                    variant,
                    _f(r["pr_auc"]),
                    _f(r["recall_at_1pct_fpr"]),
                    _f(r["ring_recall_at_1pct_fpr"], 3),
                    _money(r["money_saved"]),
                ]
            )
    s = proto["split"]
    note = (
        f"Without the gap, validation runs from day {s['val']['days'][0]} to the build day and {s['val']['fraud_unlabelled_at_build'] * 100:.1f} percent "
        "of its fraud is unlabelled at build time."
    )
    return [
        *_table(["model", "protocol", "PR-AUC", "recall at 1% FPR", "ring recall", "money saved"], rows),
        note,
        "",
    ]


SECTIONS = (
    ("Detection on the test window", detection_table),
    ("Money saved under the cost model", money_table),
    ("Per-pattern recall at 1 percent FPR", pattern_table),
    ("Graph feature ablation (gradient boosting)", ablation_table),
    ("Label-maturity gap", protocol_table),
    ("Point-in-time correctness", pit_table),
    ("Cost of a label leak", leakage_table),
    ("Scoring latency", latency_table),
    ("Seeds", seeds_table),
)


def render_markdown(results: dict[str, Any]) -> str:
    """The full results document."""
    lines = ["# Benchmark results", "", header_line(results), "", "## Dataset", "", *dataset_lines(results)]
    for title, fn in SECTIONS:
        body = fn(results)
        if body:
            lines += [f"## {title}", "", *body]
    return "\n".join(lines).rstrip() + "\n"


def readme_block(results: dict[str, Any]) -> str:
    """The README section: the same tables at heading level three."""
    lines = [header_line(results), "", *dataset_lines(results)]
    for title, fn in SECTIONS:
        body = fn(results)
        if body:
            lines += [f"### {title}", "", *body]
    return "\n".join(lines).rstrip()


def update_readme(path: Path, results: dict[str, Any]) -> None:
    """Replace the text between the RESULTS markers."""
    text = path.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    path.write_text(text[:start] + "\n" + readme_block(results) + "\n" + text[end:])


def _lookup(results: dict[str, Any], path: str) -> Any:
    node: Any = results
    for key in path.split("/"):
        node = node[key]
    return node


def check_thresholds(results: dict[str, Any], path: Path) -> list[str]:
    """Failures of the checks for this results file's profile (an empty list means the gate passes)."""
    spec = json.loads(path.read_text())
    failures = []
    for check in spec[results["meta"]["profile"]]:
        try:
            value = _lookup(results, check["path"])
        except (KeyError, TypeError):
            failures.append(f"{check['path']} missing from results")
            continue
        if "min" in check and not value >= check["min"]:
            failures.append(f"{check['path']} = {value} below minimum {check['min']}")
        if "max" in check and not value <= check["max"]:
            failures.append(f"{check['path']} = {value} above maximum {check['max']}")
        if "equals" in check and value != check["equals"]:
            failures.append(f"{check['path']} = {value} expected {check['equals']}")
    return failures
