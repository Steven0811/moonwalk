"""Reads config/robot.yaml (and gait.yaml) for the CAD side. All lengths mm, masses g, angles deg.

Part code reads every dimension from here; nothing dimensional is typed in the part functions.
Derived foot quantities (pad recess, pivot) come from cad/analysis/kinematics.py so phase 1 and the CAD
share one derivation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "cad"
OUT = CAD / "out"
sys.path.insert(0, str(CAD / "analysis"))

import kinematics as _kin  # noqa: E402


class NS(dict):
    """dict with attribute access (recursively)."""

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return NS(v) if isinstance(v, dict) and not isinstance(v, NS) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def load_robot(path: Path | None = None) -> NS:
    with open(path or ROOT / "config" / "robot.yaml") as f:
        return NS(yaml.safe_load(f))


def load_gait() -> NS:
    with open(ROOT / "config" / "gait.yaml") as f:
        return NS(yaml.safe_load(f))


P = load_robot()


def foot_geom(variant: str | None = None):
    return _kin.foot_geom(dict(P), variant or P.foot_variant)


def leg_plane_x(side: str) -> float:
    """x of a leg's centre plane in the base frame (left = -X)."""
    return (-1.0 if side == "L" else 1.0) * P.geometry.hip_spacing / 2


def servo_bands() -> dict:
    """Lateral bands (distance from the leg/servo centre plane, mm) derived from the servo geometry.

    case     : |x| ≤ case/2
    horn     : case/2 … case/2 + horn protrusion (horn lives here)
    arm      : horn face - recess … horn face - recess + arm thickness (horn-mounted bracket plates)
    """
    g = P.servo.geometry
    half_span = (g.horn_face_to_case * 2 + g.case[2]) / 2           # horn face to horn face / 2 = 14.5
    s = P.structure
    a0 = half_span - P["print"]["horn_recess_depth"]
    return dict(case=g.case[2] / 2, horn_face=half_span, arm=(a0, a0 + s.arm_thickness))
