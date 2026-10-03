"""Static analysis: CoM, support polygon margins, ground-reaction distribution and joint torques.

Units: mm, g, rad internally; torque converted once with G_NMM (g·mm·g0 -> N·m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from kinematics import rot

G0 = 9.81
G_NMM = G0 * 1e-6          # 1 g at 1 mm lever arm -> N·m

JOINTS = ("hip", "knee", "ankle")
AXIS_SIGN = {"hip": 1.0, "knee": -1.0, "ankle": 1.0}   # joint axis along ±X (robot.yaml joints.*.axis)


@dataclass
class MassModel:
    base: tuple[float, np.ndarray]           # mass, com (x, y, z) in base frame
    thigh: tuple[float, np.ndarray]
    shank: tuple[float, np.ndarray]
    foot: tuple[float, np.ndarray]

    @property
    def total(self) -> float:
        return self.base[0] + 2 * (self.thigh[0] + self.shank[0] + self.foot[0])


def electronics_mass(robot: dict, factor: float = 1.0, battery_dy: float = 0.0, battery_dz: float = 0.0):
    """Electronics mass (g) and centre of mass in the base frame. factor scales every placeholder mass."""
    m_tot, mom = 0.0, np.zeros(3)
    for name, e in robot["electronics"].items():
        if not isinstance(e, dict) or "mass" not in e:
            continue
        if e.get("enabled", True) is False:
            continue
        p = np.array(e["position"], float)
        if name == "battery":
            p = p + np.array([0.0, battery_dy, battery_dz])
        m = e["mass"] * factor
        m_tot += m
        mom += m * p
    return m_tot, mom / m_tot


def mass_model(robot: dict, variant: str, elec_factor: float = 1.0, battery_dy: float = 0.0,
               battery_dz: float = 0.0, link_override: dict | None = None) -> MassModel:
    me = link_override or robot["mass_estimate"]
    mb, cb = me["base_link"]["mass"], np.array(me["base_link"]["com"], float)
    me_, ce = electronics_mass(robot, elec_factor, battery_dy, battery_dz)
    base = (mb + me_, (mb * cb + me_ * ce) / (mb + me_))
    foot = me["foot_" + variant] if ("foot_" + variant) in me else me["foot"]
    return MassModel(base, (me["thigh"]["mass"], np.array(me["thigh"]["com"], float)),
                     (me["shank"]["mass"], np.array(me["shank"]["com"], float)),
                     (foot["mass"], np.array(foot["com"], float)))


@dataclass
class LegState:
    x: float                 # lateral position of the leg plane
    hip: np.ndarray          # (y, z) world
    qh: float
    qk: float
    qa: float
    ankle: np.ndarray        # (y, z)
    pitch: float
    contacts: list           # [(x, y, kind)]
    sliding: bool = False


def link_points(mm: MassModel, legs: list[LegState], L1: float, pelvis: np.ndarray):
    """[(mass, (x, y, z), owner)] for every link; owner = (leg index, segment) or ('base',)."""
    out = [(mm.base[0], np.array([mm.base[1][0], pelvis[0] + mm.base[1][1], pelvis[1] + mm.base[1][2]]), ("base",))]
    for i, lg in enumerate(legs):
        for seg, (m, c), origin, ang in (
            ("thigh", mm.thigh, lg.hip, lg.qh),
            ("shank", mm.shank, lg.hip + rot(lg.qh) @ np.array([0.0, -L1]), lg.qh - lg.qk),
            ("foot", mm.foot, lg.ankle, lg.pitch),
        ):
            yz = origin + rot(ang) @ c[1:]
            # com x is given for the left leg (x < 0); the right leg is its mirror image
            out.append((m, np.array([lg.x - c[0] * np.sign(lg.x), yz[0], yz[1]]), (i, seg)))
    return out


def com(points) -> np.ndarray:
    m = sum(p[0] for p in points)
    return sum(p[0] * p[1] for p in points) / m


# ---------------------------------------------------------------- support polygon
def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _ray_extent(hull: np.ndarray, p: np.ndarray, axis: int):
    """Interval of the line through p along `axis` (0 = x, 1 = y) that lies inside the hull."""
    other = 1 - axis
    vals = []
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        lo, hi = sorted((a[other], b[other]))
        if hi - lo < 1e-12:
            if abs(a[other] - p[other]) < 1e-9:
                vals += [a[axis], b[axis]]
            continue
        if lo - 1e-12 <= p[other] <= hi + 1e-12:
            t = (p[other] - a[other]) / (b[other] - a[other])
            vals.append(a[axis] + t * (b[axis] - a[axis]))
    if not vals:
        return None
    return min(vals), max(vals)


def margins(hull: np.ndarray, p: np.ndarray) -> dict:
    """Fore-aft and lateral margins of point p (x, y) along the Y and X lines through it (negative = outside),
    plus the Euclidean distance to the nearest edge (signed)."""
    out = {}
    for name, axis in (("lateral", 0), ("fore_aft", 1)):
        ext = _ray_extent(hull, p, axis) if len(hull) >= 3 else None
        out[name] = -1e3 if ext is None else min(p[axis] - ext[0], ext[1] - p[axis])
    if len(hull) >= 3:
        d = []
        n = len(hull)
        for i in range(n):
            a, b = hull[i], hull[(i + 1) % n]
            e = b - a
            nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)    # outward for CCW hull
            d.append(-np.dot(p - a, nrm))
        out["euclid"] = min(d)
    else:
        out["euclid"] = -1e3
    return out


# ---------------------------------------------------------------- joint torques
def joint_torques(mm: MassModel, legs: list[LegState], L1: float, pelvis: np.ndarray, mu_slide: float,
                  total_mass: float):
    """Minimum achievable max(|τ|/τ_rated) ground-force distribution (LP over contact-point weights).

    Ground forces: vertical w_k·m·g at each contact point (w ≥ 0, Σw = 1, Σw·p = CoM); a sliding foot also
    carries a forward friction force μ·N (opposing its backward slide), reacted by the other foot.
    Returns (best, worst): best = signed τ {(leg, joint): N·m} of the min-max distribution; worst = per-joint
    max |τ| over all admissible distributions (an upper bound however the load really splits).
    (None, None) if the CoM is outside the support polygon.
    """
    pts = link_points(mm, legs, L1, pelvis)
    c = com(pts)
    cps = [(i, np.array([x, y])) for i, lg in enumerate(legs) for (x, y, _k) in lg.contacts]
    n = len(cps)
    W = total_mass  # g; forces in g-force, converted by G_NMM
    # τ_joint = a + B @ w   (N·m) for each (leg, joint)
    rows, consts, keys = [], [], []
    for i, lg in enumerate(legs):
        knee = lg.hip + rot(lg.qh) @ np.array([0.0, -L1])
        jpos = {"hip": lg.hip, "knee": knee, "ankle": lg.ankle}
        distal = {"hip": ("thigh", "shank", "foot"), "knee": ("shank", "foot"), "ankle": ("foot",)}
        for j in JOINTS:
            r0 = jpos[j]
            a = 0.0
            for m, p, own in pts:
                if own[0] == i and own[1] in distal[j]:
                    a += -(p[1] - r0[0]) * m                       # gravity: τx = r_y·F_z, F_z = -m
            b = np.zeros(n)
            for k, (li, p) in enumerate(cps):
                ry, rz = p[1] - r0[0], 0.0 - r0[1]
                if li == i:
                    b[k] += ry * W                                 # vertical ground force
                    if lg.sliding:
                        b[k] += -rz * mu_slide * W                  # forward friction on the sliding foot
                else:
                    if legs[li].sliding:
                        b[k] += -rz * (-mu_slide) * W               # reaction carried by this (support) leg
            s = -AXIS_SIGN[j] * G_NMM                              # servo torque = −external moment
            rows.append(s * b)
            consts.append(s * a)
            keys.append((i, j))
    rows, consts = np.array(rows), np.array(consts)
    lim = 0.216
    # variables: w_0..w_{n-1}, t ; minimize t
    cvec = np.zeros(n + 1)
    cvec[-1] = 1.0
    A_ub = np.vstack([np.hstack([rows, -lim * np.ones((len(rows), 1))]),
                      np.hstack([-rows, -lim * np.ones((len(rows), 1))])])
    b_ub = np.concatenate([-consts, consts])
    A_eq = np.vstack([np.append(np.ones(n), 0.0),
                      np.append([p[0] for _, p in cps], 0.0),
                      np.append([p[1] for _, p in cps], 0.0)])
    b_eq = np.array([1.0, c[0], c[1]])
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(0, None)], method="highs")
    if not res.success:
        return None, None
    tau = consts + rows @ res.x[:n]
    best = {k: float(v) for k, v in zip(keys, tau)}
    # worst case: for each joint, the largest |τ| over every statically admissible distribution
    worst = {}
    for r_, c_, k in zip(rows, consts, keys):
        vals = []
        for sgn in (1.0, -1.0):
            rr = linprog(-sgn * r_, A_eq=A_eq[:, :n], b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
            if rr.success:
                vals.append(abs(c_ + r_ @ rr.x))
        worst[k] = max(vals)
    return best, worst
