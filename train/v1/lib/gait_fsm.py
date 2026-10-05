"""Four-phase moonwalk state machine (train/CLAUDE.md "Gait") from snapshot/gait.yaml keyframes.

Half cycle (gait.yaml half_cycle.<variant>): leg A is the toe-raised support leg (stays), leg B slides back.
Second half: same keyframes with A and B swapped. Each keyframe gives the foot states (flat / switch / toe),
B's flat position relative to A (units of step_length) and the pelvis y relative to A, linearly interpolated;
the feet roll without slipping between states (kinematics.Foot.pose). Joint targets come from leg IK.

Hard rule 12: the ankle target is recomputed every control step from the MEASURED pelvis pitch, hip and knee:
    ankle = foot_pitch_desired - pelvis_pitch - hip + knee

Start-up (not in gait.yaml): from the standing pose (both feet 'switch' side by side for the wheel foot, flat for
PTFE) to the cycle's 'slide_mid' keyframe (feet side by side, A toe-raised, B flat) in 1.5 s after a 0.5 s hold;
the cycle continues from 'slide_mid'. Built here from the same gait.yaml quantities; stated with every result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .kinematics import Foot, Leg

SIDES = ("L", "R")


@dataclass
class Key:
    t: float               # seconds from the start of the sequence
    A: str
    B: str
    By: float              # mm, B flat position relative to A
    py: float              # mm, pelvis y relative to A
    H: float               # mm, hip height
    name: str


class GaitFSM:
    def __init__(self, robot: dict, gait: dict, variant: str, cycle_time: float | None = None,
                 startup: bool = True, smooth: bool = True):
        g = robot["geometry"]
        # Interpolation between keyframes: linear gave velocity jumps at every keyframe, and the abrupt load shift
        # at each role swap rocked the robot onto foot edges (sim 2026-10-04: support polygon a triangle for ~0.1 s,
        # fore-aft margin down to -0.4 mm). Smoothstep keeps the same keyframes (same static analysis) with zero
        # velocity at each keyframe.
        self.smooth = smooth
        self.foot = Foot.from_robot(robot, variant)
        self.leg = Leg(g["thigh"], g["shank"])
        self.variant = variant
        self.T = cycle_time or gait["cycle_time"]
        self.s = gait["step_length"]
        self.H = gait["hip_height"][variant]
        kf = gait["half_cycle"][variant]
        p0 = kf[0]["pelvis_y"]
        # one half cycle in seconds
        self.half = [Key(k["t"] * self.T / 2, k["A"], k["B"], k["B_y"] * self.s,
                         k.get("pelvis_y", p0 - self.s / 2), self.H, k["name"]) for k in kf]
        # Start-up: standing pose -> the cycle's 'slide_mid' keyframe (feet side by side, A toe-raised, B flat), whose
        # static margin phase 1 verified. A first version that slid B backward into 'transfer' with A in the switch
        # posture tipped forward (sim, 2026-10-04): once B is flat it only touches on its axle line and the front of
        # the support polygon moves back to ≈ +8 mm while the CoM is still centred for the standing polygon.
        self.entry = [k for k in self.half if k.name == "slide_mid"][0]
        st = gait["standing"]
        stand_state = st["foot_state"][variant]
        Hs, pys = st["hip_height"][variant], st["pelvis_y"][variant]
        e = self.entry
        self.startup = [
            Key(0.0, stand_state, stand_state, 0.0, pys, Hs, "stand"),
            Key(0.5, stand_state, stand_state, 0.0, pys, Hs, "stand_hold"),
            Key(2.0, e.A, e.B, e.By, e.py, e.H, "startup_to_slide_mid"),
        ] if startup else []
        self.t_startup = self.startup[-1].t if startup else 0.0
        self.t_entry_in_half = e.t

    # ------------------------------------------------------------------ reference
    def _interp(self, keys: list[Key], t: float):
        for a, b in zip(keys[:-1], keys[1:]):
            if a.t <= t <= b.t:
                u = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
                if self.smooth:
                    u = u * u * (3.0 - 2.0 * u)          # smoothstep: zero velocity at every keyframe
                th = lambda s1, s2: self.foot.theta(s1) + u * (self.foot.theta(s2) - self.foot.theta(s1))
                return (th(a.A, b.A), th(a.B, b.B), a.By + u * (b.By - a.By), a.py + u * (b.py - a.py),
                        a.H + u * (b.H - a.H), a.name)
        k = keys[-1]
        return self.foot.theta(k.A), self.foot.theta(k.B), k.By, k.py, k.H, k.name

    def phase(self, time_s: float):
        """(half index 0/1, time within the half, phase name, in start-up?)"""
        if time_s < self.t_startup:
            return 0, time_s, None, True
        tc = time_s - self.t_startup + self.t_entry_in_half       # cycle time, starting at the entry keyframe
        half = int(tc // (self.T / 2))
        return half, tc - half * (self.T / 2), None, False

    def reference(self, time_s: float):
        """Per side: (ankle dy, dz relative to the hip, mm; desired foot pitch, rad); plus info dict."""
        half, th, _, start = self.phase(time_s)
        keys = self.startup if start else self.half
        thA, thB, By, py, H, name = self._interp(keys, th)
        support = "L" if half % 2 == 0 else "R"                     # leg A
        slider = "R" if support == "L" else "L"
        out = {}
        for side, yflat, theta in ((support, 0.0, thA), (slider, By, thB)):
            (ay, az), pitch = self.foot.pose(yflat, theta)
            out[side] = (ay - py, az - H, pitch)
        sliding = (not start and name in ("slide_start", "slide_mid"))
        return out, {"support": support, "slider": slider, "phase": name, "startup": start, "half": half,
                     "theta_A": thA, "theta_B": thB, "By": By, "py": py, "H": H, "sliding": sliding}

    def joint_targets(self, time_s: float, meas: dict | None = None, pelvis_pitch: float = 0.0,
                      ankle_mode: str = "full"):
        """Joint targets {joint: rad}. ankle_mode (hard rule 12 variants, compared in phase 3):
        'full'    ankle = pitch_des - pelvis_pitch_meas - (hip_meas - knee_meas)
        'hipknee' ankle = pitch_des - (hip_meas - knee_meas)       (pelvis assumed level)
        'planned' ankle = pitch_des - (hip_plan - knee_plan)       (no feedback)"""
        if ankle_mode == "planned":
            meas, pelvis_pitch = None, 0.0
        elif ankle_mode == "hipknee":
            pelvis_pitch = 0.0
        ref, info = self.reference(time_s)
        q = {}
        for side, (dy, dz, pitch) in ref.items():
            reach = math.hypot(dy, dz)
            if reach > 0.995 * (self.leg.L1 + self.leg.L2):
                raise ValueError(f"t={time_s:.2f}s {side}: target out of reach ({reach:.1f} mm)")
            qh, qk = self.leg.ik(dy, dz)
            hh = meas[f"{side}_hip_pitch"] if meas else qh
            kk = meas[f"{side}_knee_pitch"] if meas else qk
            q[f"{side}_hip_pitch"] = qh
            q[f"{side}_knee_pitch"] = qk
            q[f"{side}_ankle_pitch"] = pitch - pelvis_pitch - (hh - kk)
        return q, info
