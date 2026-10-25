"""Command line entry points."""

from __future__ import annotations

import json

import pytest

from fraudnet import cli
from fraudnet.pipeline import load_artifacts
from fraudnet.sim import SimConfig


def test_parser_knows_every_command():
    parser = cli.build_parser()
    for command in ("simulate", "train", "serve", "leakage"):
        args = parser.parse_args([command])
        assert args.command == command and args.seed == 42
    assert parser.parse_args(["serve"]).port == 8772
    with pytest.raises(SystemExit):
        parser.parse_args(["fly"])


def test_simulate_caches_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRAUDNET_DATA_DIR", str(tmp_path))
    assert cli.main(["simulate", "--quick"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["transactions"] > 100_000 and len(out["sha256"]) == 64
    assert list(tmp_path.glob("world_seed42_*.npz"))


def test_train_writes_loadable_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRAUDNET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FRAUDNET_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(cli, "_config", lambda _args: SimConfig.small())
    assert cli.main(["train", "--no-gnn"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["models"] == ["rules", "gbdt"]
    bundle, store = load_artifacts(tmp_path / "artifacts")
    assert bundle.gnn is None and store.clock == bundle.split.build_ts
