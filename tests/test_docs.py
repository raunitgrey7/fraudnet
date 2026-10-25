"""The README must carry the benchmark output, and no prose or code may contain em or en dashes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results.json"
BANNED = (chr(0x2014), chr(0x2013))


def _results() -> dict:
    if not RESULTS.exists():
        pytest.skip("benchmarks/results.json has not been generated")
    return json.loads(RESULTS.read_text())


def test_readme_detection_table_matches_results():
    results = _results()
    text = (ROOT / "README.md").read_text()
    for model, row in results["models"].items():
        pattern = rf"^\| {re.escape(model)} \| ([0-9.]+) \[[0-9.]+, [0-9.]+\] \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \["
        match = re.search(pattern, text, flags=re.MULTILINE)
        assert match, f"README has no detection row for {model}"
        pr_auc, roc_auc, r05, r1 = (float(x) for x in match.groups())
        m = row["metrics"]
        assert (pr_auc, roc_auc, r05, r1) == (
            m["pr_auc"],
            m["roc_auc"],
            m["recall_at_0.5pct_fpr"],
            m["recall_at_1pct_fpr"],
        )


def test_readme_header_matches_results_provenance():
    results = _results()
    text = (ROOT / "README.md").read_text()
    assert f"commit `{results['meta']['commit']}`" in text
    assert f"Total benchmark time {results['meta']['seconds']['total']} s" in text


def test_readme_ablation_rows_match_results():
    results = _results()
    text = (ROOT / "README.md").read_text()
    for variant, row in results["ablation"].items():
        match = re.search(rf"^\| {re.escape(variant)} \| ([0-9.]+) \|", text, flags=re.MULTILINE)
        assert match, f"README has no ablation row for {variant}"
        assert float(match.group(1)) == row["pr_auc"]


def _values(node: object) -> list[float]:
    if isinstance(node, dict):
        return [v for item in node.values() for v in _values(item)]
    if isinstance(node, list):
        return [v for item in node for v in _values(item)]
    return [float(node)] if isinstance(node, int | float) and not isinstance(node, bool) else []


def test_numbers_in_readme_prose_come_from_results():
    """Every precise number in 'What the numbers say' must be a value in results.json, as printed."""
    results = _results()
    text = (ROOT / "README.md").read_text()
    prose = text[text.index("## What the numbers say") : text.index("## Models")]
    values = _values({k: v for k, v in results.items() if k != "meta"})
    decimals = re.findall(r"(?<![\d.])(\d+\.\d{3,})(?![\d])", prose)
    money = re.findall(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d,])", prose)
    assert decimals and money
    for token in decimals:
        places = len(token.split(".")[1])
        assert any(round(v, places) == float(token) for v in values), f"{token} is not in results.json"
    for token in money:
        assert any(round(v) == int(token.replace(",", "")) for v in values), f"{token} is not in results.json"


def test_report_block_is_generated_from_results():
    results = _results()
    report = (ROOT / "REPORT.md").read_text()
    assert f"commit `{results['meta']['commit']}`" in report
    assert f"{results['models']['gbdt']['metrics']['pr_auc']:.4f} [" in report


def _text_files() -> list[Path]:
    files = [ROOT / name for name in ("README.md", "REPORT.md", "Makefile", "Dockerfile", "pyproject.toml")]
    files += (
        sorted((ROOT / "docs").glob("*.md"))
        + sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "tests").rglob("*.py"))
    )
    files += sorted((ROOT / "src" / "fraudnet" / "serving" / "static").glob("*")) + sorted(
        (ROOT / "benchmarks").glob("*.md")
    )
    files += [ROOT / ".github" / "workflows" / "ci.yml"]
    return [f for f in files if f.exists() and f.is_file()]


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_or_en_dashes(path):
    text = path.read_text(encoding="utf-8")
    for char in BANNED:
        assert char not in text, f"{path} contains U+{ord(char):04X}"
