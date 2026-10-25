"""End-to-end benchmark on the small world, report rendering, README update and the threshold gate."""

from __future__ import annotations

import json

import pytest

from fraudnet import bench
from fraudnet.evaluation.report import BEGIN, END, check_thresholds, render_markdown, update_readme


@pytest.fixture(scope="module")
def small_results() -> dict:
    return bench.run(7, "small", [], data_dir=None)


@pytest.mark.slow
def test_benchmark_produces_every_section(small_results):
    for key in (
        "meta",
        "dataset",
        "models",
        "ablation",
        "protocol",
        "leakage_cost",
        "point_in_time",
        "latency",
    ):
        assert key in small_results, key
    models = small_results["models"]
    assert set(models) == {"rules", "gbdt_tabular", "gbdt_streaming", "gbdt", "graphsage", "gbdt+graphsage"}
    for name, row in models.items():
        lo, hi = row["ci95"]["pr_auc"]
        assert lo <= hi, name
        assert set(row["per_pattern_at_1pct_fpr"]) == {
            "fraud_ring",
            "money_mule",
            "account_takeover",
            "card_testing",
        }
        assert row["money"]["money_saved"] <= row["money"]["oracle_saved"]
    json.dumps(small_results)


@pytest.mark.slow
def test_benchmark_proves_point_in_time_and_parity(small_results):
    pit = small_results["point_in_time"]
    assert pit["honest"]["passed"] is True
    assert pit["leaky_instant_labels"]["passed"] is False
    assert small_results["latency"]["online_offline_feature_max_abs_diff"] == 0.0


@pytest.mark.slow
def test_leak_inflates_offline_numbers(small_results):
    leak = small_results["leakage_cost"]
    assert leak["leaky_model_on_leaky_features"]["pr_auc"] > leak["leaky_model_on_online_features"]["pr_auc"]


@pytest.mark.slow
def test_markdown_and_readme_update(small_results, tmp_path):
    text = render_markdown(small_results)
    for heading in (
        "## Detection on the test window",
        "## Money saved",
        "## Graph feature ablation",
        "## Point-in-time correctness",
        "## Scoring latency",
    ):
        assert heading in text
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n\n{BEGIN}\nPLACEHOLDER-TABLES\n{END}\n\noutro\n")
    update_readme(readme, small_results)
    updated = readme.read_text()
    assert (
        "PLACEHOLDER-TABLES" not in updated
        and updated.startswith("intro")
        and updated.rstrip().endswith("outro")
    )
    assert "### Detection on the test window" in updated


def test_threshold_gate(tmp_path):
    results = {
        "meta": {"profile": "full"},
        "models": {"gbdt": {"metrics": {"pr_auc": 0.5}}},
        "point_in_time": {"honest": {"passed": True}},
    }
    spec = tmp_path / "thresholds.json"
    spec.write_text(json.dumps({"full": [
        {"path": "models/gbdt/metrics/pr_auc", "min": 0.4},
        {"path": "point_in_time/honest/passed", "equals": True},
    ]}))  # fmt: skip
    assert check_thresholds(results, spec) == []
    spec.write_text(json.dumps({"full": [
        {"path": "models/gbdt/metrics/pr_auc", "min": 0.6},
        {"path": "models/gbdt/metrics/pr_auc", "max": 0.45},
        {"path": "models/missing/metrics/pr_auc", "min": 0.1},
    ]}))  # fmt: skip
    failures = check_thresholds(results, spec)
    assert len(failures) == 3
    assert any("below minimum" in f for f in failures) and any("missing" in f for f in failures)


def test_committed_thresholds_cover_both_profiles():
    spec = json.loads((bench.BENCH_DIR / "thresholds.json").read_text())
    assert {"full", "quick"} <= set(spec)
    paths = {c["path"] for c in spec["full"]}
    assert "point_in_time/honest/passed" in paths and "models/gbdt/metrics/pr_auc" in paths
