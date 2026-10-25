"""Build the serving state from trained artifacts: bundle, feature store as of build time, queue and feed."""

from __future__ import annotations

from fraudnet.config import Settings
from fraudnet.pipeline import artifacts_exist, load_artifacts
from fraudnet.serving.alerts import AlertQueue
from fraudnet.serving.service import ScoringService
from fraudnet.sim.config import SimConfig
from fraudnet.sim.simulate import load_or_simulate


def load_service(settings: Settings) -> ScoringService:
    """Load what ``fraudnet train`` wrote; the simulator's test window becomes the replay feed."""
    if not artifacts_exist(settings.artifact_dir):
        raise FileNotFoundError(f"no trained bundle in {settings.artifact_dir}; run `make train` first")
    bundle, store = load_artifacts(settings.artifact_dir)
    feed = load_or_simulate(SimConfig(**bundle.sim_config), settings.data_dir)
    return ScoringService(bundle, store, AlertQueue(settings.alert_db), feed, settings.default_model)
