"""Command line: simulate the world, train the serving bundle, serve the API, run the leakage check.

``fraudnet serve`` trains first when ``artifacts/`` is empty, so ``make serve`` works from a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from fraudnet.config import Settings
from fraudnet.logging_utils import configure_logging
from fraudnet.sim.config import DAY, HOUR, SimConfig


def _config(args: argparse.Namespace) -> SimConfig:
    return SimConfig.quick(args.seed) if args.quick else SimConfig(seed=args.seed)


def cmd_simulate(args: argparse.Namespace, settings: Settings) -> int:
    """Simulate (or load from cache) and print the dataset summary."""
    from fraudnet.sim.dataset import cache_path
    from fraudnet.sim.simulate import load_or_simulate

    cfg = _config(args)
    dataset = load_or_simulate(cfg, settings.data_dir)
    print(
        json.dumps(
            {
                "path": str(cache_path(cfg, settings.data_dir)),
                "sha256": dataset.checksum(),
                **dataset.summary(),
            },
            indent=2,
        )
    )
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    """Replay to the build day, fit the models, choose thresholds and write ``artifacts/``."""
    from threadpoolctl import threadpool_limits

    from fraudnet.pipeline import prepare, save_artifacts, train_bundle

    started = time.perf_counter()
    with threadpool_limits(limits=2):
        prep = prepare(_config(args), settings.data_dir, through_test=False)
        bundle = train_bundle(prep, with_gnn=not args.no_gnn)
    save_artifacts(bundle, prep.store_at_build, settings.artifact_dir)
    summary = {
        "artifact_dir": str(settings.artifact_dir),
        "models": bundle.models,
        "thresholds": bundle.thresholds,
        "fit_seconds": bundle.fit_seconds,
        "build_day": bundle.split.build_day,
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Serve the API with uvicorn, training the bundle first if it is missing."""
    import uvicorn

    from fraudnet.pipeline import artifacts_exist
    from fraudnet.serving.app import create_app

    if not artifacts_exist(settings.artifact_dir):
        print("no trained bundle found; training first", file=sys.stderr)
        cmd_train(args, settings)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None, access_log=False)
    return 0


def cmd_leakage(args: argparse.Namespace, settings: Settings) -> int:
    """Run the point-in-time checker on the first weeks of the dataset."""
    from fraudnet.features.leakage import check_point_in_time, known_at
    from fraudnet.features.replay import replay
    from fraudnet.sim.simulate import load_or_simulate

    dataset = known_at(load_or_simulate(_config(args), settings.data_dir), args.days * DAY)
    cut = int(args.days * 0.7) * DAY + 7 * HOUR
    reports = check_point_in_time(dataset, cut, lambda ds: replay(ds).table.X)
    for report in reports:
        print(json.dumps({**report.__dict__, "passed": report.passed}))
    return 0 if all(r.passed for r in reports) else 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with one sub-command per action."""
    parser = argparse.ArgumentParser(prog="fraudnet", description="Graph machine learning for payment fraud")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (
        ("simulate", "simulate the world and cache it under data/"),
        ("train", "fit the serving bundle into artifacts/"),
        ("serve", "serve the API (trains first when artifacts/ is empty)"),
        ("leakage", "run the point-in-time checker"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--quick", action="store_true", help="the smaller CI world")
        if name in ("train", "serve"):
            p.add_argument("--no-gnn", action="store_true", help="skip the GraphSAGE model")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8772)
        if name == "leakage":
            p.add_argument("--days", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``fraudnet`` script."""
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    handlers = {"simulate": cmd_simulate, "train": cmd_train, "serve": cmd_serve, "leakage": cmd_leakage}
    return handlers[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
