"""The console's ring graph must stay inside its panel.

The first layout let repulsion outgrow every other force, so each node of a ring ended up pinned to the
border of the drawing area. The layout is plain JavaScript in ``static/app.js``; this test runs that
function under Node on graphs shaped like real device-sharing clusters and checks the positions. It is
skipped where Node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "src" / "fraudnet" / "serving" / "static" / "app.js"
WIDTH, HEIGHT, MARGIN = 560, 380, 12

HARNESS = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const body = src.slice(src.indexOf("function layout("), src.indexOf("function svg("));
const layout = new Function(body + "; return layout;")();
const graph = JSON.parse(process.argv[2]);
const first = layout(graph.nodes, graph.edges, graph.width, graph.height).pos;
const second = layout(graph.nodes, graph.edges, graph.width, graph.height).pos;
console.log(JSON.stringify({ first, second }));
"""


def _cluster(components: int, accounts: int, devices: int) -> dict[str, list[dict[str, str]]]:
    """Several components, each a set of accounts that all log in from the same devices."""
    nodes, edges = [], []
    for c in range(components):
        account_ids = [f"a{c}_{i}" for i in range(accounts)]
        device_ids = [f"d{c}_{i}" for i in range(devices)]
        nodes += [{"id": i, "type": "account"} for i in account_ids]
        nodes += [{"id": i, "type": "device"} for i in device_ids]
        edges += [
            {"source": a, "target": device_ids[(k + j) % devices], "kind": "device"}
            for k, a in enumerate(account_ids)
            for j in range(min(2, devices))
        ]
    return {"nodes": nodes, "edges": edges}


def _layout(graph: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, float]]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    payload = json.dumps(graph | {"width": WIDTH, "height": HEIGHT})
    done = subprocess.run(
        [node, "-e", HARNESS, str(APP_JS), payload], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    ("components", "accounts", "devices"),
    [(1, 3, 1), (1, 6, 3), (4, 3, 2), (1, 33, 54), (9, 6, 4)],
    ids=["tiny", "small", "disconnected", "large-ring", "ninety-nodes"],
)
def test_ring_layout_stays_inside_the_panel(components: int, accounts: int, devices: int) -> None:
    graph = _cluster(components, accounts, devices)
    positions = _layout(graph)["first"]
    assert len(positions) == len(graph["nodes"])
    on_border = [
        p
        for p in positions
        if not (
            MARGIN + 0.5 < p["x"] < WIDTH - MARGIN - 0.5 and MARGIN + 0.5 < p["y"] < HEIGHT - MARGIN - 0.5
        )
    ]
    assert on_border == [], f"{len(on_border)} of {len(positions)} nodes sit on the border"


def test_ring_layout_is_deterministic() -> None:
    result = _layout(_cluster(3, 5, 3))
    assert result["first"] == result["second"]
