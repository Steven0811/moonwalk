"""Run the rule-based moonwalk in the Isaac Lab env (no viewer by default) and log a full trace.

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/run_gait.py --variant wheel --seconds 60

Logged per control step (100 Hz): time, phase, support side, base pose, joint positions and targets, foot and
wheel body positions, clutch state; per physics step (1 kHz): servo torque, available torque, joint speed.
Writes runs/gait/<variant>_<stamp>/trace.npz and summary.txt (with the measurement conditions).
Sign convention: backward distance = -(Δy of the base) > 0 when the robot moves backward (toward -Y).
"""
import json
import math
import time

from _boot import VDIR, finish_args, parser

p = parser(__doc__.splitlines()[0])
p.add_argument("--seconds", type=float, default=60.0, help="gait duration after the 2 s start-up")
p.add_argument("--cycle", type=float, default=None, help="cycle time (s); default gait.yaml (4 s)")
p.add_argument("--gif", action="store_true", help="record a side-view animated GIF from a camera (headless)")
p.add_argument("--realtime", action="store_true", help="pace the loop to wall-clock time (for watching with --viz kit)")
p.add_argument("--ankle-mode", default="hipknee", choices=["full", "hipknee", "planned"])
p.add_argument("--interp", default="smooth", choices=["smooth", "linear"])
p.add_argument("--tag", default="")
a = finish_args(p)

from isaaclab.app import launch_simulation  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.env import MoonwalkEnv  # noqa: E402
from lib.env_cfg import ACTIVE_JOINTS, MoonwalkEnvCfg  # noqa: E402
from lib.gait_fsm import GaitFSM  # noqa: E402
from lib.snapshot import gait as load_gait, robot as load_robot  # noqa: E402

D = math.pi / 180


def quat_pitch(q):
    qx, qy, qz, qw = q
    return 2 * math.atan2(qx, qw)


def main():
    robot, gait = load_robot(), load_gait()
    cfg = MoonwalkEnvCfg()
    cfg.variant = a.variant
    cfg.episode_length_s = a.seconds + 10
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = VDIR / "runs" / "gait" / f"{a.variant}_{stamp}{'_' + a.tag if a.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    fsm = GaitFSM(robot, gait, a.variant, a.cycle, smooth=a.interp == "smooth")
    cfg.side_camera = a.gif
    cfg.finalize()
    # the whole env cfg goes to the launcher so it sees the camera sensor and starts the renderer
    with launch_simulation(cfg, a):
        env = MoonwalkEnv(cfg)
        base_env = env
        env.reset()
        # live window: look at the robot from its right side (forward = right of the screen)
        if hasattr(env.sim, "set_camera_view"):
            o = env.scene.env_origins[0].tolist()
            env.sim.set_camera_view([o[0] + 0.7, o[1] - 0.2, 0.25], [o[0], o[1] - 0.2, 0.12])
        frames, wall0 = [], time.time()
        r = base_env.robot
        bi = {n: i for i, n in enumerate(r.body_names)}
        servo = base_env.servo
        servo.log, servo.log_tau, servo.log_vel, servo.log_avail = True, [], [], []
        jn = r.joint_names
        log = {k: [] for k in ("t", "base", "pitch", "q", "q_tgt", "feet", "wheels", "support", "phase", "fallen",
                               "locked", "com", "foot_quat", "f_feet", "f_wheels", "theta_A", "theta_B")}
        fs_ids = [base_env.feet.find_sensors(f"{s}_foot")[0][0] for s in ("L", "R")]
        ws_ids = [base_env.wheels.find_sensors(f"{s}_wheel")[0][0] for s in ("L", "R")] if base_env.wheels else []
        n = int((fsm.t_startup + a.seconds) / 0.01)
        t_fall = None
        for i in range(n):
            t = i * 0.01
            qm = {j: float(r.data.joint_pos.torch[0, jn.index(j)]) for j in ACTIVE_JOINTS}
            pp = quat_pitch(r.data.root_link_quat_w.torch[0].tolist())
            tgt, info = fsm.joint_targets(t, qm, pp, a.ankle_mode)
            act = torch.tensor([[tgt[j] for j in ACTIVE_JOINTS]], device=base_env.device)
            _, _, term, _, _ = env.step(act)
            if a.gif:                                           # camera follows the robot fore-aft
                b = r.data.root_link_pos_w.torch[0]
                eye = torch.tensor([[float(b[0]) + 0.62, float(b[1]), 0.13]], device=base_env.device)
                tgt_ = torch.tensor([[float(b[0]), float(b[1]), 0.115]], device=base_env.device)
                env.scene["camera"].set_world_poses_from_view(eye, tgt_)
            if a.gif and i % 5 == 0:                           # 20 frames per simulated second
                rgb = env.scene["camera"].data.output["rgb"][0]
                frames.append(rgb.torch[..., :3].cpu().numpy() if hasattr(rgb, "torch") else rgb[..., :3].cpu().numpy())
            if a.realtime:
                lag = (i + 1) * 0.01 - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
            base = (r.data.root_link_pos_w.torch[0] - base_env.scene.env_origins[0]).tolist()
            log["t"].append(t)
            log["base"].append(base)
            log["pitch"].append(pp)
            log["q"].append([qm[j] for j in ACTIVE_JOINTS])
            log["q_tgt"].append([tgt[j] for j in ACTIVE_JOINTS])
            log["feet"].append([r.data.body_link_pos_w.torch[0, bi[f"{s}_foot"]].tolist() for s in ("L", "R")])
            if a.variant == "wheel":
                log["wheels"].append([r.data.body_link_pos_w.torch[0, bi[f"{s}_wheel"]].tolist() for s in ("L", "R")])
                log["locked"].append(base_env.clutch.locked[0].tolist())
            log["com"].append((base_env.com_w()[0] + base_env.scene.env_origins[0]).tolist())   # world frame, like the feet
            log["foot_quat"].append([r.data.body_link_quat_w.torch[0, bi[f"{s}_foot"]].tolist() for s in ("L", "R")])
            fn = base_env.feet.data.net_normal_forces_w.torch[0]
            log["f_feet"].append([float(fn[k].norm()) for k in fs_ids])
            if ws_ids:
                wn = base_env.wheels.data.net_normal_forces_w.torch[0]
                log["f_wheels"].append([float(wn[k].norm()) for k in ws_ids])
            log["theta_A"].append(info["theta_A"])
            log["theta_B"].append(info["theta_B"])
            log["support"].append(info["support"])
            log["phase"].append(info["phase"] if not info["startup"] else "startup")
            log["fallen"].append(bool(term[0]))
            if bool(term[0]) and t_fall is None:
                t_fall = t
                break
        tau = torch.stack(servo.log_tau)[:, 0].cpu().numpy()
        vel = torch.stack(servo.log_vel)[:, 0].cpu().numpy()
        avail = torch.stack(servo.log_avail)[:, 0].cpu().numpy()
        order = [servo.joint_names.index(j) for j in ACTIVE_JOINTS]
        np.savez_compressed(out / "trace.npz", t=np.array(log["t"]), base=np.array(log["base"]),
                            pitch=np.array(log["pitch"]), q=np.array(log["q"]), q_tgt=np.array(log["q_tgt"]),
                            feet=np.array(log["feet"]), wheels=np.array(log["wheels"]) if log["wheels"] else np.zeros(0),
                            locked=np.array(log["locked"]) if log["locked"] else np.zeros(0),
                            support=np.array(log["support"]), phase=np.array(log["phase"]),
                            com=np.array(log["com"]), foot_quat=np.array(log["foot_quat"]),
                            f_feet=np.array(log["f_feet"]),
                            f_wheels=np.array(log["f_wheels"]) if log["f_wheels"] else np.zeros(0),
                            theta_A=np.array(log["theta_A"]), theta_B=np.array(log["theta_B"]),
                            tau=tau[:, order], vel=vel[:, order], avail=avail[:, order],
                            t_startup=fsm.t_startup, cycle=fsm.T, step=fsm.s,
                            origin=base_env.scene.env_origins[0].cpu().numpy())
        base = np.array(log["base"])
        t_arr = np.array(log["t"])
        gi = t_arr >= fsm.t_startup
        back = -(base[-1, 1] - base[gi][0, 1]) * 1000 if gi.any() else 0.0
        dur = t_arr[-1] - fsm.t_startup
        meta = {"variant": a.variant, "ankle_mode": a.ankle_mode, "interp": a.interp, "cycle_s": fsm.T, "step_mm": fsm.s, "gait_seconds_requested": a.seconds,
                "survived_s": float(dur if t_fall is None else t_fall - fsm.t_startup),
                "fell": t_fall is not None, "backward_distance_mm": float(back),
                "conditions": "1 env, no harness, no viewer, Isaac Sim PhysX dt 1 ms, control 100 Hz, servo model "
                              "(latency 5-10 ms random per reset, backlash per robot.yaml, 0.088° quantisation), "
                              "zoned friction materials from robot.yaml, no randomization, rule-based controller v1"}
        (out / "summary.json").write_text(json.dumps(meta, indent=1))
        print(f"[run_gait] {json.dumps(meta)}\n[run_gait] trace -> {out}", flush=True)
        if a.gif and frames:
            from PIL import Image
            imgs = [Image.fromarray(f.astype("uint8")) for f in frames]
            gif = out / f"walk_{a.variant}.gif"
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=50, loop=0, optimize=True)
            print(f"[run_gait] GIF ({len(imgs)} frames, real time) -> {gif}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
