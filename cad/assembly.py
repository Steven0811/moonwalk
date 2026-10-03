"""Joint definitions and forward kinematics. The CAD assembly, the views, the checks and the URDF all use
these transforms, so the model and the URDF cannot disagree.

Joint origins come from robot.yaml geometry; axes and ranges from robot.yaml joints. Left axes are given in
robot.yaml; right axes are their mirror image (axial vector reflected about YZ: (ax, -ay, -az)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import Location

from params import P, leg_plane_x


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: tuple            # xyz in the parent frame, mm (zero pose: no rotation)
    axis: tuple
    lower: float | None      # deg
    upper: float | None
    kind: str = "revolute"   # revolute | continuous
    side: str = "L"
    key: str = ""            # robot.yaml joints key (hip_pitch, ...)


def _axis(key: str, side: str):
    a = P.joints[key]["axis"]
    return tuple(float(v) for v in a) if side == "L" else (float(a[0]), -float(a[1]), -float(a[2]))


def joints(variant: str | None = None) -> list[Joint]:
    variant = variant or P.foot_variant
    g = P.geometry
    out = []
    if P.hip_roll.enabled:
        raise NotImplementedError("hip_roll.enabled: the L_hip module is not modelled yet (see model_report.md)")
    for side in ("L", "R"):
        xc = leg_plane_x(side)
        r = P.joints
        out.append(Joint(f"{side}_hip_pitch", "base_link", f"{side}_thigh", (xc, 0.0, 0.0), _axis("hip_pitch", side),
                         *r.hip_pitch.range, side=side, key="hip_pitch"))
        out.append(Joint(f"{side}_knee_pitch", f"{side}_thigh", f"{side}_shank", (0.0, 0.0, -g.thigh),
                         _axis("knee_pitch", side), *r.knee_pitch.range, side=side, key="knee_pitch"))
        out.append(Joint(f"{side}_ankle_pitch", f"{side}_shank", f"{side}_foot", (0.0, 0.0, -g.shank),
                         _axis("ankle_pitch", side), *r.ankle_pitch.range, side=side, key="ankle_pitch"))
        if variant == "wheel":
            w = g.foot.wheel
            out.append(Joint(f"{side}_wheel_axle", f"{side}_foot", f"{side}_wheel",
                             (0.0, w.axle_y, -g.ankle_height + w.diameter / 2), _axis("wheel_axle", side),
                             None, None, kind="continuous", side=side, key="wheel_axle"))
    return out


def rot_axis(axis, deg: float) -> np.ndarray:
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def base_height() -> float:
    g = P.geometry
    return g.thigh + g.shank + g.ankle_height


def fk(q: dict | None = None, variant: str | None = None, base=None) -> dict:
    """World 4x4 transform (mm) of every link for joint angles q {joint name: deg}. Zero pose by default:
    base_link at z = thigh + shank + ankle height, feet flat on z = 0."""
    q = q or {}
    T = {"base_link": base if base is not None else _tf(np.eye(3), (0, 0, base_height()))}
    for j in joints(variant):
        R = rot_axis(j.axis, q.get(j.name, 0.0))
        T[j.child] = T[j.parent] @ _tf(R, j.origin)
    return T


def _tf(R, p) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def to_location(M: np.ndarray) -> Location:
    from OCP.gp import gp_Trsf
    t = gp_Trsf()
    t.SetValues(*M[0, :4], *M[1, :4], *M[2, :4])
    return Location(t)
