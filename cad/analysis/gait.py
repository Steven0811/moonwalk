"""Gait sampling for the static analysis (reads the keyframes in config/gait.yaml).

Half cycle: leg A (support, toe-raised) stays at y_flat = 0, leg B slides. The second half is the mirror
(A on the other side). Everything is quasi-static: each sample is analysed as an equilibrium pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from kinematics import D2R, contacts, foot_geom, foot_pose, leg
from statics import LegState, com, convex_hull, joint_torques, link_points, margins, mass_model

KNEE_MIN = 5.0 * D2R          # stay away from the straight-knee singularity everywhere


@dataclass
class Sample:
    half: int
    t: float                  # fraction of the half cycle
    phase: str
    legs: list
    pelvis: np.ndarray        # (y, z) of the hip-axis midpoint
    com: np.ndarray           # (x, y, z)
    hull: np.ndarray
    margin: dict
    feasible: bool
    violation: str = ""
    tau: dict | None = None          # min-max distribution (signed, N·m)
    tau_worst: dict | None = None    # upper bound over admissible distributions (|τ|, N·m)


class GaitModel:
    def __init__(self, robot: dict, gait: dict, variant: str, elec_factor=1.0, battery_dy=0.0, battery_dz=0.0,
                 link_override=None):
        self.robot, self.gait, self.variant = robot, gait, variant
        self.fg = foot_geom(robot, variant)
        self.leg = leg(robot)
        self.H = gait["hip_height"][variant]
        self.s = gait["step_length"]
        self.w = robot["geometry"]["hip_spacing"]
        self.kf = gait["half_cycle"][variant]
        self.theta = {"flat": 0.0, "switch": self.fg.switch, "toe": self.fg.face}
        j = robot["joints"]
        self.ranges = {k: np.array(j[k + "_pitch"]["range"], float) * D2R for k in ("hip", "knee", "ankle")}
        self.mm = mass_model(robot, variant, elec_factor, battery_dy, battery_dz, link_override)
        self.mu = robot["friction"]["slide_design_value"]

    # pelvis keyframes: one value per keyframe except the last (derived)
    def default_pelvis(self):
        return np.array([k["pelvis_y"] for k in self.kf[:-1]], float)

    def _keys(self, pel):
        return np.append(pel, pel[0] - self.s / 2)

    def pose(self, t: float, pel: np.ndarray, half: int) -> Sample:
        kf = self.kf
        ts = [k["t"] for k in kf]
        i = min(int(np.searchsorted(ts, t, side="right")) - 1, len(kf) - 2)
        a, b = kf[i], kf[i + 1]
        u = (t - a["t"]) / (b["t"] - a["t"])
        keys = self._keys(pel)
        lerp = lambda x0, x1: x0 + u * (x1 - x0)
        By = lerp(a["B_y"], b["B_y"]) * self.s
        thA = lerp(self.theta[a["A"]], self.theta[b["A"]])
        thB = lerp(self.theta[a["B"]], self.theta[b["B"]])
        py = lerp(keys[i], keys[i + 1])
        pelvis = np.array([py, self.H])
        sideA = -1.0 if half == 0 else 1.0
        legs, ok, why = [], True, ""
        for x, yflat, th, sliding in ((sideA * self.w / 2, 0.0, thA, False),
                                      (-sideA * self.w / 2, By, thB, a["B_y"] != b["B_y"])):
            ankle, pitch = foot_pose(self.fg, yflat, th)
            cps, zmin = contacts(self.fg, x, ankle, pitch)
            assert abs(zmin) < 1e-6, zmin
            sol = self.leg.ik(ankle[0] - py, ankle[1] - self.H)
            if sol is None:
                return Sample(half, t, a["name"], [], pelvis, np.zeros(3), np.zeros((0, 2)), {}, False, "unreachable")
            qh, qk = sol
            qa = pitch - (qh - qk)
            for nm, q in (("hip", qh), ("knee", qk), ("ankle", qa)):
                lo, hi = self.ranges[nm]
                if q < lo - 1e-9 or q > hi + 1e-9:
                    ok, why = False, f"{nm} {q / D2R:.1f}° out of range"
            if qk < KNEE_MIN:
                ok, why = False, f"knee {qk / D2R:.1f}° < {KNEE_MIN / D2R:.0f}° (near singular)"
            legs.append(LegState(x, pelvis.copy(), qh, qk, qa, ankle, pitch, cps, sliding))
        c = com(link_points(self.mm, legs, self.leg.L1, pelvis))
        hull = convex_hull(np.array([[p[0], p[1]] for lg in legs for p in lg.contacts]))
        m = margins(hull, c[:2])
        return Sample(half, t, a["name"], legs, pelvis, c, hull, m, ok, why)

    def sample(self, pel, n=60, halves=(0, 1)):
        ts = np.linspace(0.0, 1.0, n + 1)[:-1]
        return [self.pose(t, pel, h) for h in halves for t in ts]

    def objective(self, pel, n=60):
        """Worst static margin over the cycle (same samples as the final check); infeasible poses are
        penalised in proportion to how far the knee is from the singular limit."""
        worst = np.inf
        for s in self.sample(pel, n):
            if not s.legs:
                return -1e4
            v = min(s.margin["fore_aft"], s.margin["lateral"])
            if not s.feasible:
                kmin = min(lg.qk for lg in s.legs)
                v = -100.0 - max(KNEE_MIN - kmin, 0.0) / D2R
            worst = min(worst, v)
        return worst

    def optimise_pelvis(self):
        """Pelvis keyframes maximising the worst static margin over the cycle."""
        pel = np.zeros(len(self.kf) - 1)
        # initial guess: CoM over the centre of the support polygon's fore-aft extent at each keyframe
        for _ in range(3):
            for i, k in enumerate(self.kf[:-1]):
                s = self.pose(k["t"], pel, 0)
                if not s.margin:
                    continue
                from statics import _ray_extent
                ext = _ray_extent(s.hull, s.com[:2], 1)
                if ext is not None:
                    pel[i] += 0.5 * (ext[0] + ext[1]) - s.com[1]
        res = minimize(lambda p: -self.objective(p), pel, method="Nelder-Mead",
                       options={"xatol": 0.05, "fatol": 0.01, "maxiter": 1500, "initial_simplex": None})
        return res.x, -res.fun

    def torques(self, samples):
        for s in samples:
            if s.legs:
                s.tau, s.tau_worst = joint_torques(self.mm, s.legs, self.leg.L1, s.pelvis, self.mu, self.mm.total)
        return samples
