"""Sagittal-plane kinematics: foot contact geometry, foot rolling, leg IK/FK.

Conventions (root CLAUDE.md): Z up, +Y forward, +X right. Angles in the config are degrees; this module
works in radians internally and returns degrees for joint angles.
Rotation about +X by a:  (y, z) -> (y cos a - z sin a,  y sin a + z cos a).  Toe-down = negative a.
Foot world pitch = hip - knee + ankle (pelvis level).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

D2R = np.pi / 180.0


def rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class FootGeom:
    """Foot contact geometry in the foot frame (origin on the ankle axis, mm). Ground = z -h when flat."""

    variant: str
    h: float
    toe: float
    heel: float
    width: float
    pad_len: float
    face: float                  # rad
    switch: float                # rad (target switch-over angle)
    wheel_r: float = 0.0
    wheel_y: float = 0.0
    wheel_w: float = 0.0
    heel_clear: float = 0.0
    sole_front: float = 0.0      # ptfe only
    heel_split: float = 0.0
    pad_rear_recess: float = field(init=False)
    pivot1: np.ndarray = field(init=False)

    def __post_init__(self):
        yp = self.toe - self.pad_len
        if self.variant == "wheel":
            self.pivot1 = np.array([self.wheel_y, -self.h + self.wheel_r])     # foot rotates about the axle
        else:
            self.pivot1 = np.array([self.sole_front, -self.h])                 # about the sole front edge
        y1, z1 = self.pivot1
        th = self.switch
        # recess e such that rotating about pivot1 by `switch` brings the pad rear edge (yp, -h+e) to the ground
        self.pad_rear_recess = self.h + z1 + (-self.h - z1 + (yp - y1) * np.sin(th)) / np.cos(th)

    # --- named points in the foot frame (y, z) ---
    @property
    def pad_rear(self) -> np.ndarray:
        return np.array([self.toe - self.pad_len, -self.h + self.pad_rear_recess])

    @property
    def pad_tip(self) -> np.ndarray:
        return np.array([self.toe, -self.h + self.pad_rear_recess + self.pad_len * np.tan(self.face)])

    @property
    def track(self) -> float:
        return self.width - self.wheel_w

    def contact_features(self):
        """List of (kind, x_offsets, (y, z), radius). kind is used to name contacts in the report."""
        f = []
        hw = self.width / 2
        f.append(("pad", (-hw, hw), tuple(self.pad_rear), 0.0))
        f.append(("pad", (-hw, hw), tuple(self.pad_tip), 0.0))
        if self.variant == "wheel":
            t = self.track / 2
            f.append(("wheel", (-t, t), (self.wheel_y, -self.h + self.wheel_r), self.wheel_r))
            hb = hw - self.wheel_w - 1.0
            f.append(("heel", (-hb, hb), (-self.heel, -self.h + self.heel_clear), 0.0))
        else:
            f.append(("sole", (-hw, hw), (-self.heel, -self.h), 0.0))
            f.append(("sole", (-hw, hw), (self.sole_front, -self.h), 0.0))
        return f


def foot_geom(robot: dict, variant: str | None = None) -> FootGeom:
    g = robot["geometry"]
    ft = g["foot"]
    v = variant or robot["foot_variant"]
    pad = ft["toe_pad"]
    kw = dict(variant=v, h=g["ankle_height"], toe=ft["toe_length"], heel=ft["heel_length"], width=ft["width"],
              pad_len=pad["length"], face=pad["face_angle"] * D2R, switch=pad["switch_angle"] * D2R)
    if v == "wheel":
        w = ft["wheel"]
        kw.update(wheel_r=w["diameter"] / 2, wheel_y=w["axle_y"], wheel_w=w["width"], heel_clear=w["heel_clearance"])
    else:
        p = ft["ptfe"]
        kw.update(sole_front=ft["toe_length"] - pad["length"] - p["pad_gap"], heel_split=p["heel_split"])
    return FootGeom(**kw)


def foot_pose(fg: FootGeom, y_flat: float, theta: float) -> tuple[np.ndarray, float]:
    """Ankle (y, z) in the world and foot pitch for plantarflexion `theta` (rad, ≥ 0).

    The foot rolls without slipping: about pivot1 (axle / sole front edge) up to the switch angle, then about
    the pad rear edge up to the face angle, then about the pad tip. y_flat = ankle y with the foot flat.
    """
    ankle = np.array([y_flat, fg.h])
    stages = [(fg.pivot1, 0.0, fg.switch), (fg.pad_rear, fg.switch, fg.face), (fg.pad_tip, fg.face, np.inf)]
    pitch = 0.0
    for piv_local, lo, hi in stages:
        if theta <= lo:
            break
        d = min(theta, hi) - lo
        P = ankle + rot(pitch) @ piv_local
        ankle = P + rot(-d) @ (ankle - P)
        pitch -= d
    return ankle, pitch


def contacts(fg: FootGeom, x_center: float, ankle: np.ndarray, pitch: float, tol: float = 0.05):
    """World contact points (x, y, kind) of a foot posed at (ankle, pitch); also returns the lowest height."""
    R = rot(pitch)
    pts = []
    for kind, xs, yz, r in fg.contact_features():
        p = ankle + R @ np.array(yz)
        z = p[1] - r
        for xo in xs:
            pts.append((x_center + xo, p[0], z, kind))
    zmin = min(p[2] for p in pts)
    return [(x, y, k) for x, y, z, k in pts if z - zmin <= tol], zmin


@dataclass
class Leg:
    L1: float
    L2: float

    def ik(self, dy: float, dz: float) -> tuple[float, float] | None:
        """Hip pitch and knee (rad) placing the ankle at (dy, dz) relative to the hip axis. None if unreachable."""
        d2 = dy * dy + dz * dz
        c = (d2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        if c > 1.0 or c < -1.0:
            return None
        qk = np.arccos(c)                                  # ≥ 0: human-style flexion
        alpha = np.arctan2(dy, -dz)                         # hip->ankle line, from straight down toward +Y
        beta = np.arctan2(self.L2 * np.sin(qk), self.L1 + self.L2 * np.cos(qk))
        return alpha + beta, qk

    def fk(self, qh: float, qk: float) -> tuple[np.ndarray, np.ndarray]:
        """Knee and ankle (y, z) relative to the hip for hip pitch qh and knee qk (rad)."""
        down = np.array([0.0, -1.0])
        knee = rot(qh) @ down * self.L1
        ankle = knee + rot(qh - qk) @ down * self.L2
        return knee, ankle


def leg(robot: dict) -> Leg:
    g = robot["geometry"]
    return Leg(g["thigh"], g["shank"])
