"""Feetech HD-1910 angle-servo model as a custom Isaac Lab actuator (train/CLAUDE.md "Servo model").

At every physics step (Isaac Lab calls compute() inside the decimation loop with PhysX):

    target_d  = target delayed by the command latency (whole physics steps, random in [min, max] per reset)
    e         = quantize(target_d) - quantize(theta)                 # 0.088° encoder / command resolution
    e_bl      = deadband(e, backlash / 2)                             # gear (+ linkage) backlash: no restoring torque inside the play
    tau_cmd   = Kp * e_bl - Kd * theta_dot
    tau_avail = tau_stall * (1 - |theta_dot| / omega_noload)  if tau_cmd and theta_dot share a sign, else tau_stall
    tau       = clip(tau_cmd, -tau_avail, +tau_avail)

Backlash is modelled as a dead band on the position error (the servo does not correct inside the play); this is
an approximation of a two-mass gear model and is stated with every result.

Logging: when `log` is enabled, applied torque and joint speed are recorded at every physics step (not once per
control step), and `calls` counts compute() invocations so the per-physics-step contract can be verified.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class ServoActuator(ActuatorBase):
    cfg: "ServoActuatorCfg"

    def __init__(self, cfg: "ServoActuatorCfg", joint_names, joint_ids, num_envs, device, **kwargs):
        kwargs.pop("stiffness", None)
        kwargs.pop("damping", None)
        super().__init__(cfg, joint_names, joint_ids, num_envs, device,
                         actuator_effort_limit=cfg.tau_stall, actuator_velocity_limit=cfg.omega_noload)
        n = self.num_joints
        self.kp = torch.full((num_envs, n), cfg.kp, device=device)
        self.kd = torch.full((num_envs, n), cfg.kd, device=device)
        self.half_backlash = torch.tensor([cfg.backlash_rad.get(j, 0.0) / 2 for j in self._match(joint_names)],
                                          device=device).repeat(num_envs, 1)
        self.q_res = cfg.resolution_rad
        self._hist = torch.zeros(cfg.max_delay + 1, num_envs, n, device=device)    # ring buffer of targets
        self._head = 0
        self._lag = torch.full((num_envs,), cfg.min_delay, dtype=torch.long, device=device)
        self._filled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.calls = 0
        self.log = False
        self.log_tau: list[torch.Tensor] = []
        self.log_vel: list[torch.Tensor] = []
        self.log_avail: list[torch.Tensor] = []
        self.tau_avail = torch.zeros(num_envs, n, device=device)

    def _match(self, joint_names):
        """robot.yaml-style keys (hip_pitch, ...) for each joint name (L_hip_pitch -> hip_pitch)."""
        return [j.split("_", 1)[1] for j in joint_names]

    def reset(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = slice(None)
        n = len(range(self._num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        self._lag[env_ids] = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), device=self._device)
        self._filled[env_ids] = False

    def _delayed(self, target: torch.Tensor) -> torch.Tensor:
        H = self._hist.shape[0]
        # environments just reset: fill the whole history with the current target (no stale commands)
        if (~self._filled).any():
            self._hist[:, ~self._filled] = target[~self._filled].unsqueeze(0)
            self._filled[:] = True
        self._head = (self._head + 1) % H
        self._hist[self._head] = target
        idx = (self._head - self._lag) % H
        return self._hist[idx, torch.arange(self._num_envs, device=self._device)]

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x / self.q_res) * self.q_res if self.q_res > 0 else x

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
                ) -> ArticulationActions:
        self.calls += 1
        target = self._delayed(control_action.joint_positions)
        e = self._quantize(target) - self._quantize(joint_pos)
        e_bl = torch.sign(e) * torch.clamp(e.abs() - self.half_backlash, min=0.0)
        tau_cmd = self.kp * e_bl - self.kd * joint_vel
        same = (tau_cmd * joint_vel) > 0
        avail = torch.where(same, self.cfg.tau_stall * torch.clamp(1.0 - joint_vel.abs() / self.cfg.omega_noload,
                                                                   min=0.0), torch.full_like(joint_vel, self.cfg.tau_stall))
        tau = torch.maximum(torch.minimum(tau_cmd, avail), -avail)
        self.computed_effort = tau_cmd
        self.applied_effort = tau
        self.tau_avail = avail
        if self.log:
            self.log_tau.append(tau.detach().clone())
            self.log_vel.append(joint_vel.detach().clone())
            self.log_avail.append(avail.detach().clone())
        control_action.joint_efforts = tau
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ServoActuatorCfg(ActuatorBaseCfg):
    class_type: type = ServoActuator
    stiffness: float | None = 0.0          # PhysX drive gains stay 0: torque comes only from compute()
    damping: float | None = 0.0
    kp: float = 4.0                        # N·m/rad (estimate until bench identification)
    kd: float = 0.04                       # N·m·s/rad
    tau_stall: float = 0.88                # N·m @ 4.8 V
    omega_noload: float = 7.64             # rad/s @ 4.8 V
    min_delay: int = 5                     # physics steps (1 ms each)
    max_delay: int = 10
    backlash_rad: dict = {}                # {"hip_pitch": rad, ...}
    resolution_rad: float = 2 * math.pi / 4096


def servo_cfg_from_robot(robot: dict, dt: float, joint_names_expr: list[str], latency_ms=None,
                         backlash_scale: float = 1.0) -> ServoActuatorCfg:
    """ServoActuatorCfg from snapshot/robot.yaml. The ankle gets the linkage backlash on top of the servo's."""
    sv = robot["servo"]
    lat = latency_ms if latency_ms is not None else sv["latency_ms"]
    bl = sv["backlash_deg"] * math.pi / 180
    link = robot["structure"]["linkage"]["backlash_deg"] * math.pi / 180
    return ServoActuatorCfg(
        joint_names_expr=joint_names_expr,
        kp=sv["kp_servo"], kd=sv["kd_servo"], tau_stall=sv["stall_torque"], omega_noload=sv["no_load_speed"],
        min_delay=int(round(lat[0] * 1e-3 / dt)), max_delay=int(round(lat[1] * 1e-3 / dt)),
        backlash_rad={"hip_pitch": bl * backlash_scale, "knee_pitch": bl * backlash_scale,
                      "ankle_pitch": (bl + link) * backlash_scale},
        resolution_rad=sv["resolution_deg"] * math.pi / 180,
        joint_effort_limit=sv["stall_torque"], joint_velocity_limit=sv["no_load_speed"], armature=sv["armature"],
    )
