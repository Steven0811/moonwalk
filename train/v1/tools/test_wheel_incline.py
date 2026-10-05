"""Step 3: one-way foot wheel on an incline (wheel variant). Gravity is tilted instead of the floor.

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/test_wheel_incline.py

Pose: both feet flat on their wheels, one 30 mm ahead and one 30 mm behind, hip height 140 mm (a four-wheeled 'cart'; a single
foot on its wheels alone tips over its axle line), joints held by the servo model, pelvis over the middle.
Runs (each 4 s, 1 env, no harness, Isaac Sim PhysX dt 1 ms):
  A downhill backward 2°   -> must roll backward; terminal speed gives the equivalent rolling resistance
  B downhill forward 2°    -> forward locked: clutch slip (wheel rotation forward of the ratchet) < 1 mm at the tyre
  C downhill forward 10°   -> forward locked: clutch slip < 1 mm at the tyre
  A also checks the configured rolling resistance: terminal speed within 10 % of the prediction from it.
The wheel axles (not the pelvis, which leans on the compliant servos) are measured. Net backward travel in B/C is
reported separately: a body that rocks on a one-way clutch can ratchet backward.
"""
import math
import time

from _boot import VDIR, finish_args, parser

p = parser(__doc__.splitlines()[0])
a = finish_args(p)

from isaaclab.app import launch_simulation  # noqa: E402

import torch  # noqa: E402

from lib.env import MoonwalkEnv  # noqa: E402
from lib.env_cfg import MoonwalkEnvCfg  # noqa: E402
from lib.foot_wheel import clutch_params  # noqa: E402
from lib.kinematics import Foot, Leg  # noqa: E402
from lib.snapshot import robot as load_robot  # noqa: E402

D = math.pi / 180
STAGGER = 30.0      # mm: feet at ±30 (axles 60 mm apart); with H = 140 the CoM (≈0.10 m high) shifts ≈18 mm at 10°, inside the 30 mm margin
G = 9.81


def cart_pose(robot, py_mm):
    """Joint angles and base position for both feet flat at y = +20 / -20 mm with the pelvis at py_mm."""
    g = robot["geometry"]
    foot, leg = Foot.from_robot(robot, "wheel"), Leg(g["thigh"], g["shank"])
    H = 140.0          # lower than the gait height so both legs reach the stance with bent knees
    q = {}
    for s, yf in (("L", STAGGER), ("R", -STAGGER)):
        (ay, az), pitch = foot.pose(yf, 0.0)
        reach = math.hypot(ay - py_mm, az - H)
        if reach > 0.97 * (leg.L1 + leg.L2):
            raise ValueError(f"cart pose unreachable for {s}: hip-ankle {reach:.1f} mm vs leg {leg.L1 + leg.L2:.0f} mm")
        qh, qk = leg.ik(ay - py_mm, az - H)
        q.update({f"{s}_hip_pitch": qh, f"{s}_knee_pitch": qk, f"{s}_ankle_pitch": pitch - (qh - qk)})
    return q, (0.0, py_mm / 1000, H / 1000)


def centred_cart(env, robot, axle_mid_mm):
    """Move the pelvis relative to the (fixed) feet until the measured CoM is over the axle midline."""
    py = axle_mid_mm
    for _ in range(6):
        q, base = cart_pose(robot, py)
        off = env.measured_com_offset(q, base) * 1000              # CoM_y - pelvis_y, mm
        err = (py + off) - axle_mid_mm
        if abs(err) < 0.2:
            break
        py -= err / 0.6                                          # moving the pelvis moves the CoM by ~0.6 x as much
    q, base = cart_pose(robot, py)
    return env.set_pose(q, base), py


def run(case, slope_deg, direction):
    cfg = MoonwalkEnvCfg()
    cfg.variant = "wheel"
    s = math.sin(slope_deg * D) * direction            # +1: gravity pulls toward +Y (downhill forward)
    cfg.sim.gravity = (0.0, G * s, -G * math.cos(slope_deg * D))
    cfg.episode_length_s = 30.0
    cfg.finalize()
    return cfg


def main():
    robot = load_robot()
    out = VDIR / "runs" / "wheel_incline"
    out.mkdir(parents=True, exist_ok=True)
    cases = [("A backward 2°", 2.0, -1), ("B forward 2°", 2.0, +1), ("C forward 10°", 10.0, +1)]
    lines = [f"wheel incline test {time.strftime('%Y-%m-%d %H:%M:%S')}: wheel variant, staggered flat stance ±30 mm, hip height 140 mm, "
             f"servo model holding the joints, gravity tilted, Isaac Sim PhysX dt 1 ms, 1 env, no harness, 4 s per case"]
    cp = clutch_params(robot)
    lines.append(f"clutch (ratchet): forward sign {cp.forward_sign:+.0f}, lock spring {cp.k_lock} N·m/rad + damping {cp.d_lock} N·m·s/rad, free damping "
                 f"{cp.d_free:.3e} N·m·s/rad (equivalent of rolling resistance {robot['friction']['wheel_rolling_resistance']} "
                 f"at 40 mm/s + axle friction)")
    ok_all = True
    # one simulation app, one env per case (gravity is a sim-level setting): run the cases sequentially
    cfg0 = run(*cases[0][:1], cases[0][1], cases[0][2])
    with launch_simulation(cfg0.sim, a):
        for name, slope, direction in cases:
            cfg = run(name, slope, direction)
            env = MoonwalkEnv(cfg)
            env.reset()
            axle_mid = (0.5 * (20.0 - 20.0) + robot["geometry"]["foot"]["wheel"]["axle_y"]) / 1000
            act, py = centred_cart(env, robot, axle_mid * 1000)      # CoM (not the pelvis) midway between the axles
            env.step(act)
            com0 = float(env.com_w()[0, 1]) * 1000                  # measured after a real step
            for _ in range(30):                          # settle 0.3 s
                env.step(act)
            r = env.robot
            wb = [r.body_names.index(n) for n in ("L_wheel", "R_wheel")]
            axle_y = lambda: [float(r.data.body_link_pos_w.torch[0, b, 1]) for b in wb]
            y0 = axle_y()
            ys, fwd_max = [], [0.0, 0.0]
            resets, pitch_max, slip_max = 0, 0.0, 0.0
            fb = [r.body_names.index(n) for n in ("L_foot", "R_foot")]
            fpitch = [[], []]
            for i in range(400):
                _, _, term, trunc, _ = env.step(act)
                if bool(term[0]) or bool(trunc[0]):
                    resets += 1
                qx, qy, qz, qw = r.data.root_link_quat_w.torch[0].tolist()
                pitch_max = max(pitch_max, abs(2 * math.atan2(qx, qw)) / D)
                slip_max = max(slip_max, float((env.clutch.theta_cont * env.clutch.cp.forward_sign
                                                - env.clutch.u_ref).max()) * 6.0)        # mm at the tyre
                for k_, b_ in enumerate(fb):
                    fx, _, _, fw = r.data.body_link_quat_w.torch[0, b_].tolist()
                    fpitch[k_].append(2 * math.atan2(fx, fw) / D)
                y = axle_y()
                ys.append(sum(y) / 2)
                fwd_max = [max(f, (yi - y0i) * 1000) for f, yi, y0i in zip(fwd_max, y, y0)]
            dy = (ys[-1] - sum(y0) / 2) * 1000           # mean axle displacement (wheels on the ground)
            v_end = (ys[-1] - ys[-51]) / 0.5 * 1000      # mm/s over the last 0.5 s
            fell = resets > 0 or bool(env.fallen()[0])
            if direction < 0:
                m = sum(float(x) for x in env.robot.data.default_mass.torch[0])
                rw = robot["geometry"]["foot"]["wheel"]["diameter"] / 2000
                f_drive = m * G * math.sin(slope * D)
                v_pred = f_drive * rw * rw / (2 * cp.d_free) * 1000          # mm/s: 2 axles, viscous equivalent
                f40 = 2 * cp.d_free * 0.040 / rw / rw
                good = dy < -5.0 and not fell and abs(abs(v_end) - v_pred) / v_pred < 0.10
                lines.append(f"[{'PASS' if good else 'FAIL'}] {name}: rolled {dy:+.1f} mm in 4 s (negative = backward), "
                             f"terminal speed {abs(v_end):.1f} mm/s vs {v_pred:.1f} mm/s predicted from the configured "
                             f"resistance (must agree within 10 %); configured resistance at 40 mm/s ≙ coefficient "
                             f"{f40 / (m * G):.3f} (rolling 0.02 + axle Coulomb + axle damping, viscous equivalent)")
            else:
                good = slip_max < 1.0 and not fell
                lines.append(f"[{'PASS' if good else 'FAIL'}] {name}: clutch slip forward of the ratchet "
                             f"{slip_max:.3f} mm at the tyre (must be < 1 mm); axle travel forward L {fwd_max[0]:+.2f} / "
                             f"R {fwd_max[1]:+.2f} mm (includes the foot tilting about the tyre contact with the wheel "
                             f"locked), net {dy:+.2f} mm")
            lines[-1] += (f"; start CoM y {com0:+.1f} mm vs axle midline {axle_mid * 1000:+.1f} mm; falls/resets "
                          f"{resets}, max base pitch {pitch_max:.1f}°; max clutch slip forward of the ratchet "
                          f"{slip_max:.3f} mm at the tyre; foot pitch range L {min(fpitch[0]):+.1f}…{max(fpitch[0]):+.1f}°, "
                          f"R {min(fpitch[1]):+.1f}…{max(fpitch[1]):+.1f}°")
            ok_all &= good
            env.close()
        lines.append("RESULT: " + ("PASS" if ok_all else "FAIL"))
        (out / "result.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
