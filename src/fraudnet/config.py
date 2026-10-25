"""Runtime settings read from the environment, with defaults that work from a fresh clone."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Where data and artifacts live, which model scores by default, and the optional API key."""

    data_dir: Path
    artifact_dir: Path
    api_key: str | None
    default_model: str
    seed: int
    log_level: str
    alert_db: str

    @classmethod
    def from_env(cls) -> Settings:
        """Read ``FRAUDNET_*`` variables and ``APP_API_KEY``."""
        artifact_dir = Path(os.environ.get("FRAUDNET_ARTIFACT_DIR", ROOT / "artifacts"))
        return cls(
            data_dir=Path(os.environ.get("FRAUDNET_DATA_DIR", ROOT / "data")),
            artifact_dir=artifact_dir,
            api_key=os.environ.get("APP_API_KEY") or None,
            default_model=os.environ.get("FRAUDNET_DEFAULT_MODEL", "gbdt"),
            seed=int(os.environ.get("FRAUDNET_SEED", "42")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            alert_db=os.environ.get("FRAUDNET_ALERT_DB", ":memory:"),
        )
