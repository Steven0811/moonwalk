"""Contact points, support polygon and static margins from the simulated feet.

Candidate contact points are the bottoms of the collision primitives cad/ generated (wheel spheres, toe-pad box,
heel / sole boxes), posed with the measured body poses. A point counts as touching when it is within `tol` of the
ground AND the contact sensor on its body reports a normal force above `f_min`. The CoM projection is then
compared with the convex hull of the touching points: fore-aft margin along the Y line through the CoM, lateral
along the X line, as in the phase-1 report (cad/out/sizing_report.md) so the two can be compared.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import Foot


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FootContacts:
    """Bottom points of the foot collision primitives in the foot frame (m), by zone."""

    def __init__(self, robot: dict, variant: str):
        g = robot["geometry"]
        ft = g["foot"]
        f = Foot.from_robot(robot, variant)
        self.variant = variant
        hw = ft["width"] / 2 / 1000
        h = f.h / 1000
        a = f.face
        r = [f.pad_rear[0] / 1000, f.pad_rear[1] / 1000]
        t = [f.pad_tip[0] / 1000, f.pad_tip[1] / 1000]
        self.zones = {"toe_pad": np.array([[x, r[0], r[1]] for x in (-hw, hw)] + [[x, t[0], t[1]] for x in (-hw, hw)])}
        if variant == "wheel":
            w = ft["wheel"]
            self.wheel_r = w["diameter"] / 2000
            self.track = (ft["width"] - w["width"]) / 2 / 1000
            hb = robot["structure"]["foot"]["heel_block"]
            zc = -h + w["heel_clearance"] / 1000
            self.zones["heel"] = np.array([[x, y / 1000, zc] for x in (-hb["half_width"] / 1000, hb["half_width"] / 1000)
                                           for y in hb["y"]])
        else:
            heel, front = -ft["heel_length"] / 1000, f.pivot1[0] / 1000
            self.zones["sole"] = np.array([[x, y, -h] for x in (-hw, hw) for y in (heel, front)])

    def points(self, foot_pos, foot_quat, wheel_pos=None, side_sign=1.0):
        """{zone: (n, 3) world points}; wheels are the sphere bottoms below the wheel centres."""
        R = quat_to_R(foot_quat)
        out = {z: (R @ P.T).T + np.asarray(foot_pos) for z, P in self.zones.items()}
        if wheel_pos is not None:
            c = np.asarray(wheel_pos)
            ax = R @ np.array([1.0, 0.0, 0.0])
            out["wheel"] = np.array([c + s * self.track * ax - np.array([0, 0, self.wheel_r]) for s in (-1, 1)])
        return out


def hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 7), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 1e-12:
            lo.pop()
        lo.append(p)
    for p in pts[::-1]:
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 1e-12:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def margins(h: np.ndarray, p) -> tuple[float, float]:
    """(fore-aft, lateral) margin of p = (x, y) along the Y and X lines through it; negative = outside."""
    if len(h) < 3:
        return -1.0, -1.0
    res = []
    for axis in (1, 0):
        other = 1 - axis
        vals = []
        for i in range(len(h)):
            a_, b_ = h[i], h[(i + 1) % len(h)]
            lo_, hi_ = sorted((a_[other], b_[other]))
            if hi_ - lo_ < 1e-12 or not (lo_ <= p[other] <= hi_):
                continue
            u = (p[other] - a_[other]) / (b_[other] - a_[other])
            vals.append(a_[axis] + u * (b_[axis] - a_[axis]))
        res.append(-1.0 if not vals else min(p[axis] - min(vals), max(vals) - p[axis]))
    return res[0], res[1]
