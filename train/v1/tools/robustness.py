"""Robustness acceptance (and the vectorized-vs-single cross-check of hard rule 3).

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/robustness.py --variant wheel --envs 64

Each env runs the same rule-based gait clock; the ankle correction (hard rule 12, hip/knee mode) uses that env's
measured joints. With randomization (default): friction ×U(0.7,1.3) per robot shape, all body masses ×U(0.85,1.15),
electronics mass ±50 % / position ±1 cm (base CoM), servo latency 0–15 ms (per reset), backlash 0–1° per joint,
wheel rolling resistance ×U(0.5,1.5). Pass: ≥ 90 % of envs stand for 60 s. Falls are tallied by direction
(forward / backward / left / right from the base tilt at the fall).
--no-random: identical nominal envs; their survival and backward distance must match a single-env run (hard rule 3).
Floor check (hard rule 6): envs × spacing + travel must fit the 400 m ground plane.
"""
import json
import math
import time

from _boot import VDIR, finish_args, parser

p = parser(__doc__.splitlines()[0])
p.add_argument("--envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=60.0)
p.add_argument("--cycle", type=float, default=None)
p.add_argument("--no-random", action="store_true")
p.add_argument("--seed", type=int, default=0)
a = finish_args(p)

from isaaclab.app import launch_simulation  # noqa: E402

import torch  # noqa: E402

from lib.env import MoonwalkEnv  # noqa: E402
from lib.env_cfg import ACTIVE_JOINTS, MoonwalkEnvCfg  # noqa: E402
from lib.gait_fsm import GaitFSM  # noqa: E402
from lib.snapshot import gait as load_gait, robot as load_robot  # noqa: E402


def main():
    robot, gait = load_robot(), load_gait()
    cfg = MoonwalkEnvCfg()
    cfg.variant = a.variant
    cfg.episode_length_s = a.seconds + 10
    cfg.scene.num_envs = a.envs
    cfg.scene.env_spacing = 1.0
    if not a.no_random:
        cfg.randomize_electronics = True
        cfg.latency_ms = (0, 15)
    cfg.finalize()
    side = math.ceil(math.sqrt(a.envs)) * cfg.scene.env_spacing
    assert side + 2.0 < 400.0, "floor too small (hard rule 6)"
    fsm = GaitFSM(robot, gait, a.variant, a.cycle)
    tag = "nominal" if a.no_random else "random"
    out = VDIR / "runs" / "robustness" / f"{a.variant}_{tag}_{a.envs}env_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg.sim, a):
        env = MoonwalkEnv(cfg)
        gen = torch.Generator().manual_seed(a.seed)
        sampled = {} if a.no_random else env.randomize_robustness(gen)
        env.reset()
        r = env.robot
        jn = r.joint_names
        ids = {j: jn.index(j) for j in ACTIVE_JOINTS}
        n = int((fsm.t_startup + a.seconds) / 0.01)
        fell_at = torch.full((a.envs,), float("nan"))
        fall_dir = [""] * a.envs
        y0 = None
        for i in range(n):
            t = i * 0.01
            tgt, info = fsm.joint_targets(t, None, 0.0, "planned")      # IK targets (same for every env)
            act = torch.tensor([[tgt[j] for j in ACTIVE_JOINTS]], device=env.device).repeat(a.envs, 1)
            q = r.data.joint_pos.torch
            for s in ("L", "R"):                                        # hard rule 12, hip/knee mode, per env
                ref, _ = fsm.reference(t)
                pitch = ref[s][2]
                k = ACTIVE_JOINTS.index(f"{s}_ankle_pitch")
                act[:, k] = pitch - (q[:, ids[f"{s}_hip_pitch"]] - q[:, ids[f"{s}_knee_pitch"]])
            fallen_now = env.fallen().cpu()
            g = r.data.projected_gravity_b.torch.cpu()
            for e in torch.nonzero(fallen_now & torch.isnan(fell_at)).flatten().tolist():
                fell_at[e] = t
                # gravity in the body frame: tipping forward (top toward +Y) gives g_b = (0, +sin, -cos); tipping
                # right (top toward +X) gives g_b.x = +sin. Derived; checked against a known fall in the README.
                gx, gy = float(g[e, 0]), float(g[e, 1])
                fall_dir[e] = ("forward" if gy > 0 else "backward") if abs(gy) >= abs(gx) else ("right" if gx > 0 else "left")
            env.step(act)
            if i == int(fsm.t_startup / 0.01):
                y0 = (r.data.root_link_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]).cpu()
        y1 = (r.data.root_link_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]).cpu()
        alive = torch.isnan(fell_at)
        back = -(y1 - y0) * 1000
        dirs = {d: fall_dir.count(d) for d in ("forward", "backward", "left", "right")}
        rate = float(alive.float().mean())
        res = {"variant": a.variant, "mode": tag, "envs": a.envs, "cycle_s": fsm.T, "seconds": a.seconds,
               "standing_rate": rate, "falls_by_direction": dirs,
               "fall_times_s": [round(float(x) - fsm.t_startup, 2) for x in fell_at[~alive]],
               "backward_mm_alive_mean": float(back[alive].mean()) if alive.any() else None,
               "backward_mm_alive_min_max": [float(back[alive].min()), float(back[alive].max())] if alive.any() else None,
               "conditions": f"{a.envs} envs, no harness, no viewer, Isaac Sim PhysX dt 1 ms, control 100 Hz, ankle "
                             f"mode hipknee, smoothstep keyframes, " + ("nominal parameters" if a.no_random else
                             "randomized: friction ±30 %, mass ±15 %, electronics ±50 %/±1 cm, latency 0–15 ms, "
                             "backlash 0–1°, rolling resistance ±50 %"),
               "pass": rate >= 0.9 if not a.no_random else None, "sampled": sampled}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        print("[robustness] " + json.dumps({k: v for k, v in res.items() if k != "sampled"}), flush=True)


if __name__ == "__main__":
    main()
