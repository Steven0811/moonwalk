"""Passive foot wheel: rolling resistance and the one-way clutch (train/CLAUDE.md "Simulating the foot wheel").

The wheels get no commands. Each physics step the clutch reads the axle speed and chooses the axle damping:
    turning forward  (foot moving forward over the ground) -> lock damping   (clutch engaged, wheel held)
    turning backward (foot sliding backward)               -> free damping   (bearing + rolling resistance)

Why the damping is applied implicitly (PhysX joint drive) and not as an explicit torque: the wheel set's inertia
about the axle is ≈ 1.7e-8 kg·m² (two Ø12 TPU tyres + 3 mm steel axle). An explicit torque -D·ω integrated at
dt = 1 ms is only stable for D < 2·I/dt ≈ 3.4e-5 N·m·s/rad, far too weak to stop a wheel carrying ≈ 1 N per tyre.
The PhysX drive damping is solved implicitly and is stable at any value, so the asymmetric damping the spec
describes is written to the joint drive each physics step.

Rolling resistance and the axle's Coulomb friction are represented by an equivalent viscous damping at the design
slide speed (robot.yaml friction.wheel_rolling_resistance, wheel_axle_coulomb, wheel_axle_damping); an explicit
constant Coulomb torque chatters on this inertia for the same reason as above. The incline test measures the
resulting resistance; numbers are reported with that caveat.

Forward direction sign: a wheel rolling toward +Y turns about -X, so with the URDF axis +X 'forward' is a
negative joint speed (forward_sign = -1). Calibrated against a known case with tools/test_wheel_incline.py
(2026-10-04): with the sign set to +1 the robot was locked downhill-backward (+1.7 mm in 4 s) and rolled
downhill-forward (+77 mm), i.e. exactly reversed, so -1 is confirmed. (An earlier run that seemed to say +1 was
invalid: the test robot was tipping over and being auto-reset; see v1 README.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.actuators import ImplicitActuatorCfg


@dataclass
class ClutchParams:
    forward_sign: float          # sign of the axle speed when the wheel rolls forward (calibrated: -1)
    k_lock: float                # N·m/rad: ratchet spring when the wheel is forward of the ratchet angle
    d_lock: float                # N·m·s/rad, clutch engaged
    d_free: float                # N·m·s/rad, free rolling (bearing + rolling resistance equivalent)
    armature: float = 2e-6       # kg·m² on the axle: conditions the stiff lock for the PhysX solver (see below)


def clutch_params(robot: dict, scale_rr: float = 1.0) -> ClutchParams:
    fr = robot["friction"]
    g = robot["geometry"]
    r = g["foot"]["wheel"]["diameter"] / 2 / 1000.0                       # m
    m = robot["mass_estimate"]
    total = (m["base_link"]["mass"] + 2 * (m["thigh"]["mass"] + m["shank"]["mass"] + m["foot_wheel"]["mass"])
             + sum(e["mass"] for e in robot["electronics"].values() if isinstance(e, dict) and "mass" in e
                   and e.get("enabled", True))) / 1000.0                  # kg
    n_axle = 0.5 * total * 9.81                                          # N on one foot (both tyres, one axle)
    v_slide = 0.04                                                       # m/s: 40 mm per 1 s slide (gait.yaml design)
    w_slide = v_slide / r
    tau_rr = fr["wheel_rolling_resistance"] * n_axle * r + fr["wheel_axle_coulomb"]
    d_free = fr["wheel_axle_damping"] + scale_rr * tau_rr / w_slide
    return ClutchParams(forward_sign=-1.0, k_lock=10.0, d_lock=1.0, d_free=d_free)   # sign calibrated (incline test)


def wheel_actuator_cfg(cp: ClutchParams) -> ImplicitActuatorCfg:
    """Wheel axles: implicit drive; stiffness/damping/target switched by OneWayClutch each physics step."""
    # Armature: the bare wheel set (1.7e-8 kg·m²) with a lock spring has a natural frequency of ~24 000 rad/s; under
    # contact load PhysX's articulation solver then under-resolves the lock and the wheel creeps forward (measured:
    # 4.8 mm at the tyre on a 10° incline). 2e-6 kg·m² is ~15 % of the robot's own reflected rolling inertia
    # (m r² ≈ 1.4e-5 kg·m²), so it barely changes rolling but makes the lock solvable.
    return ImplicitActuatorCfg(joint_names_expr=[".*_wheel_axle"], stiffness=0.0, damping=cp.d_free,
                               armature=cp.armature)


class OneWayClutch:
    """Ratchet model of the needle clutch, per environment and wheel.

    u = forward_sign * axle angle (u grows when the wheel rolls forward). The ratchet angle u_ref follows the
    wheel when it rolls backward (u_ref = min(u_ref, u)). Whenever the wheel is forward of u_ref, an implicit
    spring (k_lock, d_lock) pulls it back to u_ref; otherwise only the free-rolling damping acts. Unlike a
    speed-triggered damper this cannot leak forward through repeated small back-and-forth motions.
    """

    def __init__(self, cp: ClutchParams, num_envs: int, num_wheels: int, device):
        self.cp = cp
        self.u_ref = torch.zeros(num_envs, num_wheels, device=device)
        self.theta_prev = torch.zeros(num_envs, num_wheels, device=device)
        self.theta_cont = torch.zeros(num_envs, num_wheels, device=device)
        self.locked = torch.zeros(num_envs, num_wheels, dtype=torch.bool, device=device)
        self._init = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.d_free_env = None                   # per-env free damping (robustness randomization), else cp.d_free

    def gains(self, wheel_pos: torch.Tensor):
        """(stiffness, damping, target position) for the wheel joints.

        The wheel is a continuous joint: its reported angle wraps at ±pi. The ratchet works on an unwrapped
        angle (sum of wrapped per-step increments); the drive target is written relative to the reported angle
        so it never sees the wrap.
        """
        if (~self._init).any():
            i = ~self._init
            self.theta_prev[i] = wheel_pos[i]
            self.theta_cont[i] = 0.0
            self.u_ref[i] = 0.0
            self._init[:] = True
        d_theta = torch.remainder(wheel_pos - self.theta_prev + math.pi, 2 * math.pi) - math.pi
        self.theta_prev = wheel_pos.clone()
        self.theta_cont = self.theta_cont + d_theta
        u = self.theta_cont * self.cp.forward_sign
        self.u_ref = torch.minimum(self.u_ref, u)
        self.locked = u > self.u_ref + 1e-6
        k = torch.where(self.locked, torch.full_like(u, self.cp.k_lock), torch.zeros_like(u))
        d_free = self.d_free_env[:, None].expand_as(u) if self.d_free_env is not None else torch.full_like(u, self.cp.d_free)
        d = torch.where(self.locked, torch.full_like(u, self.cp.d_lock), d_free)
        target = wheel_pos - (u - self.u_ref) * self.cp.forward_sign      # current reported angle, pulled back to the ratchet
        return k, d, target

    def reset(self, env_ids=None):
        if env_ids is None:
            self._init[:] = False
        else:
            self._init[env_ids] = False
