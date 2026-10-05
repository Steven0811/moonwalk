"""v1 reads only its own snapshot (train/CLAUDE.md version rule 4). Paths are relative to the version folder."""
from __future__ import annotations

from pathlib import Path

import yaml

VDIR = Path(__file__).resolve().parents[1]
SNAP = VDIR / "snapshot"
RUNS = VDIR / "runs"


def robot() -> dict:
    return yaml.safe_load((SNAP / "robot.yaml").read_text())


def gait() -> dict:
    return yaml.safe_load((SNAP / "gait.yaml").read_text())


def usd_path(variant: str) -> str:
    folders = sorted((SNAP / "usd").glob(f"*_{variant}_*g"))
    if len(folders) != 1:
        raise FileNotFoundError(f"expected one {variant} USD folder in {SNAP / 'usd'}, found {folders}")
    return str(next(folders[0].glob("*/*.usda")))
