"""DirectRLEnv for v1, used by the rule-based controller (and later by RL in a new version).

Action = absolute position targets (rad) for ACTIVE_JOINTS, held for one control step (10 physics steps).
Per physics step: the servo actuator computes torque (servo_model.py) and the one-way clutch sets the wheel axle
damping (foot_wheel.py). Reset = gait.yaml standing pose at the correct height (hard rule 2).
"""
from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv

from .env_cfg import ACTIVE_JOINTS, MoonwalkEnvCfg
from .foot_wheel import OneWayClutch, clutch_params
from .kinematics import Foot, Leg
from .snapshot import gait as load_gait, robot as load_robot


def standing_pose(robot: dict, gait: dict, variant: str):
    """Joint targets (rad, dict) and base position (m) of the gait.yaml standing pose."""
    g = robot["geometry"]
    foot = Foot.from_robot(robot, variant)
    leg = Leg(g["thigh"], g["shank"])
    st = gait["standing"]
    H, py = st["hip_height"][variant], st["pelvis_y"][variant]
    (ay, az), pitch = foot.pose(0.0, foot.theta(st["foot_state"][variant]))
    qh, qk = leg.ik(ay - py, az - H)
    qa = pitch - (qh - qk)
    q = {f"{s}_{k}": v for s in ("L", "R") for k, v in (("hip_pitch", qh), ("knee_pitch", qk), ("ankle_pitch", qa))}
    return q, (0.0, py / 1000.0, H / 1000.0)


class MoonwalkEnv(DirectRLEnv):
    cfg: MoonwalkEnvCfg

    def __init__(self, cfg: MoonwalkEnvCfg, render_mode=None, **kwargs):
        self.robot_yaml = load_robot()
        self.gait_yaml = load_gait()
        q_stand, base = standing_pose(self.robot_yaml, self.gait_yaml, cfg.variant)
        cfg.scene.robot.init_state.pos = base if not cfg.fix_root else (0.0, 0.0, 0.5)
        cfg.scene.robot.init_state.joint_pos = {**q_stand, **({".*_wheel_axle": 0.0} if cfg.variant == "wheel" else {})}
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.feet = self.scene["feet"]
        self.wheels = self.scene["wheels"] if cfg.variant == "wheel" else None
        jn = self.robot.joint_names
        self.active_ids = torch.tensor([jn.index(j) for j in ACTIVE_JOINTS], device=self.device)
        self.wheel_ids = [i for i, j in enumerate(jn) if j.endswith("wheel_axle")]
        self.q_stand = torch.tensor([q_stand[j] for j in ACTIVE_JOINTS], device=self.device)
        self.targets = self.q_stand.repeat(self.num_envs, 1)
        self.clutch = None
        if self.wheel_ids:
            self.clutch = OneWayClutch(clutch_params(self.robot_yaml), self.num_envs, len(self.wheel_ids), self.device)
            self._wheel_ids_t = torch.tensor(self.wheel_ids, device=self.device)
        self.servo = self.robot.actuators["servos"]
        self.physics_steps = 0

    def set_pose(self, q: dict, base_pos, env_ids=None):
        """Teleport to joint angles q {joint: rad} and base position (m, env frame); targets follow."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        r = self.robot
        jp = r.data.default_joint_pos.torch[env_ids].clone()
        for k, v in q.items():
            jp[:, r.joint_names.index(k)] = v
        pose = r.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = torch.tensor(base_pos, device=self.device) + self.scene.env_origins[env_ids]
        r.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=self.device),
                                           env_ids=env_ids)
        r.write_joint_position_to_sim_index(position=jp, env_ids=env_ids)
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=env_ids)
        self.targets[env_ids] = jp[:, self.active_ids]
        return self.targets[env_ids].clone()

    def randomize_robustness(self, gen: torch.Generator, friction=(0.7, 1.3), mass=(0.85, 1.15),
                             backlash_deg=(0.0, 1.0), rolling=(0.5, 1.5)) -> dict:
        """Per-environment robustness randomization (train/CLAUDE.md acceptance 'Robustness'); call once after
        creating the env. Servo latency 0-15 ms is set in the servo config (random per reset); electronics
        mass/position by the ElectronicsRandomizationCfg events. Returns the sampled values for the report."""
        import warp as wp
        n = self.num_envs
        u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*shape, generator=gen)
        view = self.robot.root_view
        ids = torch.arange(n, dtype=torch.int32)
        # friction: scale every shape's own static/dynamic friction (zones keep their ratio)
        mats = wp.to_torch(view.get_material_properties()).clone()
        fs = u(*friction, n)
        mats[:, :, 0:2] *= fs[:, None, None]
        view.set_material_properties(wp.from_torch(mats.contiguous(), dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))
        # mass: scale every body (inertia scaled with it)
        ms = u(*mass, n)
        masses = wp.to_torch(view.get_masses()).clone()
        inert = wp.to_torch(view.get_inertias()).clone()
        masses *= ms[:, None]
        inert *= ms[:, None, None]
        view.set_masses(wp.from_torch(masses.contiguous(), dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))
        view.set_inertias(wp.from_torch(inert.contiguous(), dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))
        # backlash: total per joint uniform in backlash_deg (replaces the nominal servo + linkage values)
        bl = u(*backlash_deg, n, self.servo.num_joints) * math.pi / 180
        self.servo.half_backlash = (bl / 2).to(self.device)
        # wheel rolling resistance
        rr = u(*rolling, n)
        if self.clutch is not None:
            self.clutch.cp_d_free = None
            self.clutch.d_free_env = (self.clutch.cp.d_free * rr).to(self.device)
        return {"friction_scale": fs.tolist(), "mass_scale": ms.tolist(), "rolling_scale": rr.tolist(),
                "backlash_deg_mean": (bl.mean(1) * 180 / math.pi).tolist()}

    def com_w(self) -> torch.Tensor:
        """Whole-robot centre of mass (num_envs, 3), env frame (m)."""
        d = self.robot.data
        m = d.default_mass.torch.to(self.device)                          # (N, B)
        p = d.body_com_pos_w.torch                                       # (N, B, 3)
        return (m.unsqueeze(-1) * p).sum(1) / m.sum(1, keepdim=True) - self.scene.env_origins

    def measured_com_offset(self, q: dict, base_pos) -> float:
        """Teleport to (q, base) and return CoM_y - base_y (m) measured after one physics step (body positions
        reported by the sim only refresh after a step)."""
        self.set_pose(q, base_pos)
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        return float(self.com_w()[0, 1]) - float(self.robot.data.root_link_pos_w.torch[0, 1]
                                                 - self.scene.env_origins[0, 1])

    # ------------------------------------------------------------------ hooks
    def _pre_physics_step(self, actions: torch.Tensor):
        self.targets = actions.clone()

    def _apply_action(self):
        if not hasattr(self, "targets"):
            return
        full = self.robot.data.default_joint_pos.torch.clone()
        full[:, self.active_ids] = self.targets
        if self.clutch is not None:
            k, d, tgt = self.clutch.gains(self.robot.data.joint_pos.torch[:, self._wheel_ids_t])
            full[:, self._wheel_ids_t] = tgt
            self.robot.write_joint_stiffness_to_sim_index(stiffness=k, joint_ids=self._wheel_ids_t)
            self.robot.write_joint_damping_to_sim_index(damping=d, joint_ids=self._wheel_ids_t)
        self.robot.actuators.target_command.set_position_index(value=full)
        self.physics_steps += 1

    def _get_observations(self) -> dict:
        d = self.robot.data
        obs = torch.cat([d.joint_pos.torch[:, self.active_ids], d.joint_vel.torch[:, self.active_ids],
                         d.projected_gravity_b.torch, d.root_lin_vel_b.torch, d.root_link_pos_w.torch[:, 2:3]], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def fallen(self) -> torch.Tensor:
        d = self.robot.data
        z = d.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        tilt = torch.acos(torch.clamp(-d.projected_gravity_b.torch[:, 2], -1.0, 1.0)) * 180.0 / math.pi
        return (z < self.cfg.fall_height) | (tilt > self.cfg.fall_tilt_deg)

    def _get_dones(self):
        died = self.fallen() if not self.cfg.fix_root else torch.zeros(self.num_envs, dtype=torch.bool,
                                                                         device=self.device)
        return died, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        r = self.robot
        pose = r.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        r.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        r.write_root_velocity_to_sim_index(root_velocity=r.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids)
        r.write_joint_position_to_sim_index(position=r.data.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids)
        r.write_joint_velocity_to_sim_index(velocity=r.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids)
        if hasattr(self, "targets"):                   # _reset_idx also runs inside DirectRLEnv.__init__
            self.targets[env_ids] = self.q_stand
        if getattr(self, "clutch", None) is not None:
            self.clutch.reset(env_ids)
