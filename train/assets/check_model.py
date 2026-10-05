"""Post-import model checks (train/CLAUDE.md "Importing into Isaac Lab", step 4). Any failure blocks simulation.

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/assets/check_model.py --variant wheel [--usd <folder>]

Backend: Isaac Sim PhysX (stated explicitly), dt = 1 ms. Two copies of the robot are spawned through an
ArticulationCfg: a fixed-root copy hanging in the air (joint directions, velocity-limit unit) and a free copy on
the ground in the gait.yaml standing pose with every joint held by a stiff PD (5-second stand test). The PD here
is only a test fixture; the versions use the custom servo actuator.

Checks:
  1 total and per-link mass = cad/out/model_report.md
  2 joint limits = robot.yaml
  3 every collision body is a primitive (no meshes / convex hulls); zoned materials bound, combine mode min
  4 joint positive directions (hip + → foot forward, knee + → heel back, ankle − → toe down), both legs
  5 one joint driven flat out reaches ≈ 7.64 rad/s (proves joint_velocity_limit is read as rad/s)
  6 standing pose, joints held, 5 s on the ground: no explosion, no sinking, no bouncing
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(__file__).resolve().parent / "build"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="wheel", choices=["wheel", "ptfe"])
parser.add_argument("--usd", default=None, help="build folder (default: newest post-processed one of the variant)")
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"

import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402

D2R = math.pi / 180.0
RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}", flush=True)


# ------------------------------------------------------------------ inputs
def build_folder() -> Path:
    if args.usd:
        return Path(args.usd)
    cands = [p for p in sorted(BUILD.glob(f"*_{args.variant}_*g")) if (p / "POSTPROCESSED.txt").exists()]
    if not cands:
        raise FileNotFoundError("no post-processed build folder; run import_urdf.py and postprocess_usd.py")
    return cands[-1]


def report_masses(variant: str) -> tuple[float, dict]:
    """Per-link masses (g) from the model_report table of this foot variant."""
    txt = (ROOT / "cad" / "out" / "model_report.md").read_text()
    sec = txt.split(f"### {variant} foot — total **")[1]
    total = float(sec.split(" g**")[0])
    links = {}
    for line in sec.split("\n\n")[1].splitlines():
        m = re.match(r"\| (\w+) \| ([0-9.]+) \|", line)
        if m:
            links[m.group(1)] = float(m.group(2))
    return total, links


def standing_pose(robot: dict, gait: dict, variant: str) -> tuple[dict, tuple]:
    """Joint angles (rad) and base position (m) of the gait.yaml standing pose (IK, same convention as cad/)."""
    g = robot["geometry"]
    h, L1, L2 = g["ankle_height"], g["thigh"], g["shank"]
    st = gait["standing"]
    H, py = st["hip_height"][variant], st["pelvis_y"][variant]
    if st["foot_state"][variant] == "switch":       # rolled about the wheel axle by the switch angle
        th = g["foot"]["toe_pad"]["switch_angle"] * D2R
        w = g["foot"]["wheel"]
        piv = (w["axle_y"], w["diameter"] / 2)
        ay, az = 0.0 - piv[0], h - piv[1]
        ankle = (piv[0] + ay * math.cos(-th) - az * math.sin(-th), piv[1] + ay * math.sin(-th) + az * math.cos(-th))
        pitch = -th
    else:
        ankle, pitch = (0.0, h), 0.0
    dy, dz = ankle[0] - py, ankle[1] - H
    c = (dy * dy + dz * dz - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    qk = math.acos(max(-1.0, min(1.0, c)))
    qh = math.atan2(dy, -dz) + math.atan2(L2 * math.sin(qk), L1 + L2 * math.cos(qk))
    qa = pitch - (qh - qk)
    q = {}
    for s in ("L", "R"):
        q.update({f"{s}_hip_pitch": qh, f"{s}_knee_pitch": qk, f"{s}_ankle_pitch": qa})
    return q, (0.0, py / 1000.0, H / 1000.0)


# ------------------------------------------------------------------ USD-level checks
def check_usd(usd_path: Path, robot: dict):
    from pxr import Usd, UsdPhysics, UsdShade      # only after Kit is up: importing pxr first breaks Kit's USD
    st = Usd.Stage.Open(str(usd_path))
    bad, unbound, n = [], [], 0
    prim_types = {"Cube", "Sphere", "Capsule", "Cylinder"}
    for p in st.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        n += 1
        if p.GetTypeName() not in prim_types or p.HasAPI(UsdPhysics.MeshCollisionAPI):
            bad.append(f"{p.GetName()}:{p.GetTypeName()}")
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        if not mat:
            unbound.append(p.GetName())
            continue
        cm = mat.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get()
        if cm != "min":
            unbound.append(f"{p.GetName()} (combine {cm})")
    record("Collision bodies are primitives; zoned materials bound, friction combine min",
           not bad and not unbound and n > 0,
           f"{n} collision prims; non-primitive: {bad or 'none'}; unbound / wrong combine: {unbound or 'none'}")


# ------------------------------------------------------------------ scene
def robot_cfg(usd: str, prim: str, fixed: bool, pos, joint_pos, robot: dict) -> ArticulationCfg:
    sv = robot["servo"]
    active = [".*_hip_pitch", ".*_knee_pitch", ".*_ankle_pitch"]
    acts = {
        # test fixture only: stiff PD holding / moving the joints, servo limits from robot.yaml
        "servos": ImplicitActuatorCfg(joint_names_expr=active, stiffness=5.0, damping=0.05,
                                      joint_effort_limit=sv["stall_torque"], joint_velocity_limit=sv["no_load_speed"],
                                      armature=sv["armature"]),
    }
    if args.variant == "wheel":
        acts["wheels"] = ImplicitActuatorCfg(joint_names_expr=[".*_wheel_axle"], stiffness=0.0, damping=0.0)
    return ArticulationCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(usd_path=usd, fix_root_link=fixed, activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, joint_pos=joint_pos),
        actuators=acts,
    )


def main():
    robot = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
    gait = yaml.safe_load((ROOT / "config" / "gait.yaml").read_text())
    folder = build_folder()
    usd = next(folder.glob("*/*.usda"))
    print(f"[check_model] variant {args.variant}, USD {usd}, backend Isaac Sim PhysX, dt 1 ms", flush=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.001, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        check_usd(usd, robot)
        sim = sim_utils.SimulationContext(sim_cfg)
        ground = sim_utils.GroundPlaneCfg(physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=robot["friction"]["floor"], dynamic_friction=robot["friction"]["floor"],
            restitution=0.0, friction_combine_mode="min", restitution_combine_mode="min"))
        ground.func("/World/ground", ground)
        q_stand, base_stand = standing_pose(robot, gait, args.variant)
        fixed = Articulation(robot_cfg(str(usd), "/World/Fixed", True, (-0.6, 0.0, 0.5),
                                       {".*": 0.0}, robot))
        free = Articulation(robot_cfg(str(usd), "/World/Free", False,
                                      (0.6, base_stand[1], base_stand[2]), q_stand, robot))
        sim.reset()
        dt = sim.get_physics_dt()
        # sim.reset() leaves the joints at 0; the init_state values are only defaults until written to the sim
        for a in (fixed, free):
            a.write_root_pose_to_sim_index(root_pose=a.data.default_root_pose.torch.clone())
            a.write_root_velocity_to_sim_index(root_velocity=a.data.default_root_vel.torch.clone())
            a.write_joint_position_to_sim_index(position=a.data.default_joint_pos.torch.clone())
            a.write_joint_velocity_to_sim_index(velocity=a.data.default_joint_vel.torch.clone())
            a.reset()
            # hold the initial pose from the first step on (the free copy keeps standing while the fixed copy is
            # tested; an unset target is 0 = straight legs, which would drive its feet into the floor)
            a.actuators.target_command.set_position_index(value=a.data.default_joint_pos.torch.clone())

        def step(n=1):
            for _ in range(n):
                for a in (fixed, free):
                    a.write_data_to_sim()
                sim.step()
                for a in (fixed, free):
                    a.update(dt)

        # ---------------- 1 masses
        tot_g, link_g = report_masses(args.variant)
        m = fixed.data.default_mass.torch[0].cpu() * 1000.0
        names = fixed.body_names
        diffs = {n: float(m[i]) - link_g[n] for i, n in enumerate(names) if n in link_g}
        missing = [n for n in link_g if n not in names]
        worst = max(abs(v) for v in diffs.values())
        record("Total and per-link mass = model_report.md",
               abs(float(m.sum()) - tot_g) < 0.2 and worst < 0.05 and not missing,
               f"total {float(m.sum()):.2f} g vs report {tot_g:.1f} g; max per-link |Δ| {worst:.3f} g over "
               f"{len(diffs)} links; missing {missing or 'none'}")

        # ---------------- 2 joint limits
        lim = fixed.data.joint_pos_limits.torch[0].cpu()
        jn = fixed.joint_names
        errs = []
        for i, n in enumerate(jn):
            key = n.split("_", 1)[1]
            if key == "wheel_axle":
                continue
            lo, hi = robot["joints"][key]["range"]
            e = max(abs(float(lim[i][0]) / D2R - lo), abs(float(lim[i][1]) / D2R - hi))
            errs.append((n, e))
        we = max(e for _, e in errs)
        record("Joint limits = robot.yaml", we < 0.01,
               f"{len(errs)} active joints, max |Δ| {we:.4f}°; order {jn}")

        # ---------------- 4 joint directions (fixed-root copy)
        bi = {n: i for i, n in enumerate(fixed.body_names)}

        def settle_to(targets: dict, n=600):
            tgt = torch.zeros(1, fixed.num_joints, device=sim.device)
            for k, v in targets.items():
                tgt[0, jn.index(k)] = v
            fixed.actuators.target_command.set_position_index(value=tgt)
            step(n)

        def foot_state(side):
            p = fixed.data.body_link_pos_w.torch[0, bi[f"{side}_foot"]].cpu()
            qx, qy, qz, qw = fixed.data.body_link_quat_w.torch[0, bi[f"{side}_foot"]].cpu().tolist()
            pitch = 2 * math.atan2(qx, qw)                     # rotation about +X (quaternion x, y, z, w)
            return p, pitch

        rows, ok_dir = [], True
        for s in ("L", "R"):
            settle_to({})
            p0, a0 = foot_state(s)
            settle_to({f"{s}_hip_pitch": 10 * D2R})
            p1, _ = foot_state(s)
            settle_to({f"{s}_knee_pitch": 10 * D2R})
            p2, _ = foot_state(s)
            settle_to({f"{s}_ankle_pitch": -10 * D2R})
            _, a3 = foot_state(s)
            hip_ok, knee_ok, ankle_ok = p1[1] > p0[1] + 1e-3, p2[1] < p0[1] - 1e-3, a3 < a0 - 5 * D2R
            ok_dir &= hip_ok and knee_ok and ankle_ok
            rows.append(f"{s}: hip+10 foot Δy {1000 * (p1[1] - p0[1]):+.1f} mm, knee+10 foot Δy "
                        f"{1000 * (p2[1] - p0[1]):+.1f} mm (heel back), ankle−10 foot pitch {(a3 - a0) / D2R:+.1f}°")
        record("Joint positive directions (hip+ forward, knee+ heel back, ankle− toe down)", ok_dir, "; ".join(rows))

        # ---------------- 5 velocity limit unit (fixed-root copy, L_hip_pitch flat out)
        j = jn.index("L_hip_pitch")
        settle_to({"L_hip_pitch": -30 * D2R}, 800)
        stiff = fixed.data.joint_stiffness.torch.clone()
        damp = fixed.data.joint_damping.torch.clone()
        s2, d2 = stiff.clone(), damp.clone()
        s2[0, j], d2[0, j] = 0.0, 10.0
        fixed.write_joint_stiffness_to_sim_index(stiffness=s2)
        fixed.write_joint_damping_to_sim_index(damping=d2)
        vt = torch.zeros(1, fixed.num_joints, device=sim.device)
        vt[0, j] = 50.0
        fixed.actuators.target_command.set_velocity_index(value=vt)
        vmax = 0.0
        for _ in range(250):
            step()
            vmax = max(vmax, abs(float(fixed.data.joint_vel.torch[0, j])))
            if float(fixed.data.joint_pos.torch[0, j]) > 80 * D2R:
                break
        lim_v = robot["servo"]["no_load_speed"]
        record("joint_velocity_limit is rad/s (one joint flat out reaches ≈ 7.64 rad/s)",
               abs(vmax - lim_v) / lim_v < 0.05,
               f"L_hip_pitch peak {vmax:.3f} rad/s = {vmax / D2R:.0f}°/s with limit {lim_v} rad/s and a 50 rad/s "
               f"velocity target (a deg/s reading would cap at {lim_v * D2R:.3f} rad/s)")
        fixed.write_joint_stiffness_to_sim_index(stiffness=stiff)
        fixed.write_joint_damping_to_sim_index(damping=damp)
        fixed.actuators.target_command.set_velocity_index(value=torch.zeros_like(vt))

        # ---------------- 6 stand 5 s, joints held (free copy), restarted from the standing pose
        free.write_root_pose_to_sim_index(root_pose=free.data.default_root_pose.torch.clone())
        free.write_root_velocity_to_sim_index(root_velocity=free.data.default_root_vel.torch.clone())
        free.write_joint_position_to_sim_index(position=free.data.default_joint_pos.torch.clone())
        free.write_joint_velocity_to_sim_index(velocity=free.data.default_joint_vel.torch.clone())
        free.reset()
        tgt = torch.zeros(1, free.num_joints, device=sim.device)
        for k, v in q_stand.items():
            tgt[0, free.joint_names.index(k)] = v
        free.actuators.target_command.set_position_index(value=tgt)
        z, vel, tilt = [], [], []
        t0 = time.time()
        for i in range(5000):
            step()
            if i % 10 == 0:
                rp = free.data.root_link_pos_w.torch[0].cpu()
                qx, qy, qz, qw = free.data.root_link_quat_w.torch[0].cpu().tolist()
                z.append(float(rp[2]))
                vel.append(float(free.data.root_link_lin_vel_w.torch[0].norm()))
                tilt.append(2 * math.asin(min(1.0, math.sqrt(qx * qx + qy * qy))) / D2R)
        z = torch.tensor(z)
        finite = bool(torch.isfinite(z).all())
        z_settled = z[100:]                                   # after 1 s
        sink = base_stand[2] - float(z.min())
        p2p = float(z_settled.max() - z_settled.min())
        ok = finite and abs(sink) < 0.003 and p2p < 0.001 and max(vel[100:]) < 0.01 and max(tilt) < 2.0
        record("Standing pose, joints held, 5 s on the ground: no explosion, sinking or bouncing", ok,
               f"base start z {1000 * base_stand[2]:.1f} mm, min {1000 * float(z.min()):.1f} mm (drop "
               f"{1000 * sink:.2f} mm), z peak-to-peak after 1 s {1000 * p2p:.3f} mm, max speed after 1 s "
               f"{100 * max(vel[100:]):.3f} cm/s, max tilt {max(tilt):.2f}°, finite {finite} "
               f"(stand pose hip {q_stand['L_hip_pitch'] / D2R:.1f}° knee {q_stand['L_knee_pitch'] / D2R:.1f}° "
               f"ankle {q_stand['L_ankle_pitch'] / D2R:.1f}°; wall {time.time() - t0:.0f} s)")

        # written inside the launch context: leaving it shuts Kit down together with this process
        out = folder / f"CHECK_MODEL_{args.variant}.txt"
        passed = all(r[1] for r in RESULTS) and len(RESULTS) == 6
        out.write_text(f"check_model {time.strftime('%Y-%m-%d %H:%M:%S')} variant {args.variant} backend Isaac Sim "
                       f"PhysX dt 1 ms: {'PASS' if passed else 'FAIL'}\n" +
                       "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}" for n, ok, d in RESULTS) + "\n")
        print(f"[check_model] {'ALL PASS' if passed else 'FAILED'} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
