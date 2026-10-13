"""Environment-aware paths and experiment configuration."""

import os
from pathlib import Path

import yaml

ROOT = Path(os.getenv("CREDITRISK_PROJECT_ROOT", str(Path.cwd()))).resolve()
DATA = Path(os.getenv("CREDITRISK_DATA_DIR", str(ROOT / "data")))
ARTIFACTS = Path(os.getenv("CREDITRISK_ARTIFACT_DIR", str(ROOT / "artifacts")))


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs/default.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
