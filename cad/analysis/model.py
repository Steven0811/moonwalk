"""Load config/robot.yaml and config/gait.yaml for the phase-1 analysis (plain Python, no CAD).

Everything here stays in the config's units (mm, g, deg). Torques are converted to N·m in exactly one
place: statics.G_NMM (g·mm -> N·m).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


def load_yaml(name: str) -> dict:
    with open(CONFIG / name) as f:
        return yaml.safe_load(f)


def load() -> tuple[dict, dict]:
    return load_yaml("robot.yaml"), load_yaml("gait.yaml")


def set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def get_path(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


def variant(robot: dict, gait: dict, **overrides) -> tuple[dict, dict]:
    """Deep-copied (robot, gait) with dotted-path overrides, e.g. {'robot.geometry.thigh': 85}."""
    r, g = copy.deepcopy(robot), copy.deepcopy(gait)
    for k, v in overrides.items():
        root, path = k.split(".", 1)
        set_path(r if root == "robot" else g, path, v)
    return r, g
