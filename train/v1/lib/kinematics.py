"""Sagittal kinematics for the gait: foot rolling between the flat / switch / toe states and leg IK.

Same conventions as the phase-1 analysis (root CLAUDE.md): Z up, +Y forward, angles about +X, toe-down = negative
pitch, foot world pitch = hip - knee + ankle. Lengths in mm, angles in rad. This is v1's own copy (versions never
import cad/); it reads only snapshot/robot.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

D2R = math.pi / 180.0


def _rot(a, y, z):
    c, s = math.cos(a), math.sin(a)
    return y * c - z * s, y * s + z * c


@dataclass
class Foot:
    variant: str
    h: float                 # ankle height (flat)
    toe: float
    pad_len: float
    face: float              # rad
    switch: float            # rad
    pivot1: tuple            # (y, z) foot frame: wheel axle / sole front edge
    pad_rear: tuple
    pad_tip: tuple

    @classmethod
    def from_robot(cls, robot: dict, variant: str) -> "Foot":
        g = robot["geometry"]
        ft = g["foot"]
        pad = ft["toe_pad"]
        h = g["ankle_height"]
        face, sw = pad["face_angle"] * D2R, pad["switch_angle"] * D2R
        yp = ft["toe_length"] - pad["length"]
        if variant == "wheel":
            w = ft["wheel"]
            p1 = (w["axle_y"], -h + w["diameter"] / 2)
        else:
            p1 = (yp - ft["ptfe"]["pad_gap"], -h)
        y1, z1 = p1
        e = h + z1 + (-h - z1 + (yp - y1) * math.sin(sw)) / math.cos(sw)       # pad rear-edge recess
        rear = (yp, -h + e)
        tip = (ft["toe_length"], -h + e + pad["length"] * math.tan(face))
        return cls(variant, h, ft["toe_length"], pad["length"], face, sw, p1, rear, tip)

    def theta(self, state: str) -> float:
        return {"flat": 0.0, "switch": self.switch, "toe": self.face}[state]

    def pose(self, y_flat: float, theta: float) -> tuple[tuple, float]:
        """Ankle (y, z) and foot pitch for plantarflexion theta ≥ 0, rolling without slip from the flat pose."""
        ay, az = y_flat, self.h
        pitch = 0.0
        for piv, lo, hi in ((self.pivot1, 0.0, self.switch), (self.pad_rear, self.switch, self.face),
                            (self.pad_tip, self.face, math.inf)):
            if theta <= lo:
                break
            d = min(theta, hi) - lo
            py, pz = _rot(pitch, *piv)
            py, pz = ay + py, az + pz
            ry, rz = _rot(-d, ay - py, az - pz)
            ay, az = py + ry, pz + rz
            pitch -= d
        return (ay, az), pitch


@dataclass
class Leg:
    L1: float
    L2: float

    def ik(self, dy: float, dz: float) -> tuple[float, float]:
        """Hip pitch and knee (rad) putting the ankle at (dy, dz) mm from the hip axis (knee ≥ 0)."""
        c = (dy * dy + dz * dz - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        qk = math.acos(max(-1.0, min(1.0, c)))
        qh = math.atan2(dy, -dz) + math.atan2(self.L2 * math.sin(qk), self.L1 + self.L2 * math.cos(qk))
        return qh, qk
