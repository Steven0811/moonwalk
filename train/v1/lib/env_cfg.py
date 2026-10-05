"""DirectRLEnvCfg for v1 (train/CLAUDE.md file structure): ground, robot, foot contact sensors, PhysX backend,
1 kHz physics, 100 Hz control, events (electronics randomization present from the start, off by default)."""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from .robot_cfg import robot_cfg
from .snapshot import robot as load_robot

PHYSICS_DT = 0.001          # ≥ 1 kHz
CONTROL_DT = 0.01           # 100 Hz (real robot bus rate target)
_R = load_robot()


def ground_material(mu: float | None = None) -> PhysxRigidBodyMaterialCfg:
    mu = _R["friction"]["floor"] if mu is None else mu
    return PhysxRigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, restitution=0.0,
                                     friction_combine_mode="min", restitution_combine_mode="min")


@configclass
class ElectronicsRandomizationCfg:
    """Electronics placeholders (models not chosen): mass ±50 %, position ±1 cm (train/CLAUDE.md task 12).
    Applied to base_link: mass add ±0.5·m_elec, CoM shift ±1 cm · m_elec / m_base_link."""

    elec_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-0.0255, 0.0255), "operation": "add"})
    elec_com = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.0034, 0.0034), "y": (-0.0034, 0.0034), "z": (-0.0034, 0.0034)}})


@configclass
class MoonwalkSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg(size=(400.0, 400.0), physics_material=ground_material()))
    robot: ArticulationCfg = robot_cfg("wheel", PHYSICS_DT)
    feet = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*/[LR]_foot", track_contact_points=True,
                            track_friction_forces=True, filter_prim_paths_expr=["/World/ground"],
                            max_contact_data_count_per_prim=16)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))


@configclass
class MoonwalkEnvCfg(DirectRLEnvCfg):
    variant: str = "wheel"
    fix_root: bool = False
    randomize_electronics: bool = False
    latency_ms: tuple | None = None           # None -> robot.yaml servo.latency_ms
    decimation: int = int(round(CONTROL_DT / PHYSICS_DT))
    episode_length_s: float = 60.0
    action_space: int = 6                     # absolute joint position targets (rad), order = ACTIVE_JOINTS
    observation_space: int = 6 + 6 + 3 + 3 + 1
    state_space: int = 0
    # use_newton_actuators defaults to True in Isaac Lab 3.0 even on PhysX; it would hand explicit actuators to a
    # native path instead of calling the Python servo model at every physics step.
    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT, render_interval=10, physics=PhysxCfg(),
                                       physics_material=ground_material(), use_newton_actuators=False)
    scene: MoonwalkSceneCfg = MoonwalkSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)
    events: object | None = None
    side_camera: bool = False                 # side-view camera sensor for headless GIF recording (hard rule 1)
    fall_height: float = 0.11                 # m: base below this = fallen
    fall_tilt_deg: float = 35.0

    def finalize(self):
        """Apply variant / options to the nested configs (call before creating the env)."""
        self.scene.robot = robot_cfg(self.variant, PHYSICS_DT, fix_root=self.fix_root, latency_ms=self.latency_ms)
        if self.variant == "wheel":
            self.scene.wheels = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*/[LR]_wheel",
                                                 track_contact_points=True, track_friction_forces=True,
                                                 filter_prim_paths_expr=["/World/ground"],
                                                 max_contact_data_count_per_prim=16)
        self.events = ElectronicsRandomizationCfg() if self.randomize_electronics else None
        if self.side_camera:
            from isaaclab.sensors import CameraCfg
            # side view from the robot's right (+X), looking toward -X, at hip height; forward (+Y) is to the right
            self.scene.camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/SideCamera", width=480, height=270, data_types=["rgb"],
                offset=CameraCfg.OffsetCfg(pos=(0.75, -0.15, 0.13), rot=(0.0, 0.0, 1.0, 0.0), convention="world"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)))
        return self


ACTIVE_JOINTS = ["L_hip_pitch", "L_knee_pitch", "L_ankle_pitch", "R_hip_pitch", "R_knee_pitch", "R_ankle_pitch"]
DEG = math.pi / 180.0
