"""ArticulationCfg for v1: snapshot USD, servo actuators on the six active joints, passive wheel axles, contact
sensors enabled (train/CLAUDE.md "Importing into Isaac Lab", step 3). Everything is read from snapshot/robot.yaml."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from .foot_wheel import clutch_params, wheel_actuator_cfg
from .servo_model import servo_cfg_from_robot
from .snapshot import robot as load_robot, usd_path

ACTIVE = [".*_hip_pitch", ".*_knee_pitch", ".*_ankle_pitch"]


def robot_cfg(variant: str, dt: float, prim_path: str = "{ENV_REGEX_NS}/Robot", fix_root: bool = False,
              init_pos=(0.0, 0.0, 0.18), init_joint_pos=None, latency_ms=None, backlash_scale: float = 1.0,
              rr_scale: float = 1.0) -> ArticulationCfg:
    robot = load_robot()
    actuators = {"servos": servo_cfg_from_robot(robot, dt, ACTIVE, latency_ms, backlash_scale)}
    if variant == "wheel":
        actuators["wheels"] = wheel_actuator_cfg(clutch_params(robot, rr_scale))
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path(variant),
            fix_root_link=fix_root,
            activate_contact_sensors=True,
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False,
                                                     solver_position_iteration_count=8,
                                                     solver_velocity_iteration_count=1)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=init_pos, joint_pos=init_joint_pos or {".*": 0.0}),
        actuators=actuators,
    )
