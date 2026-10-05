"""Step 4: standing test. Reset from the gait.yaml standing pose and hold it for 30 s with the servo model.

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/stand_test.py --variant wheel

Pass (train/CLAUDE.md task 4): no fall for 30 s, joint tracking error < 2° (after a 1 s settle), no torque
saturation (|tau| never ≥ 95 % of the available torque from the torque-speed curve). Conditions are written
with the result: 1 env, no harness, no viewer, Isaac Sim PhysX dt 1 ms, control 100 Hz, nominal latency
range from robot.yaml (random per reset), backlash from robot.yaml, no randomization.
"""
import math
import time

from _boot import VDIR, finish_args, parser

p = parser(__doc__.splitlines()[0])
p.add_argument("--seconds", type=float, default=30.0)
a = finish_args(p)

from isaaclab.app import launch_simulation  # noqa: E402

import torch  # noqa: E402

from lib.env import MoonwalkEnv  # noqa: E402
from lib.env_cfg import ACTIVE_JOINTS, MoonwalkEnvCfg  # noqa: E402

D = math.pi / 180


def main():
    cfg = MoonwalkEnvCfg()
    cfg.variant = a.variant
    cfg.episode_length_s = a.seconds + 5
    cfg.finalize()
    out = VDIR / "runs" / "stand"
    out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg.sim, a):
        env = MoonwalkEnv(cfg)
        env.reset()
        act = env.q_stand.unsqueeze(0)
        servo = env.servo
        r = env.robot
        n_ctrl = int(a.seconds / 0.01)
        errs, falls, z = [], 0, []
        for i in range(n_ctrl):
            if i == 100:                                    # after the 1 s settle: log every physics step
                servo.log, servo.log_tau, servo.log_vel, servo.log_avail = True, [], [], []
            _, _, term, trunc, _ = env.step(act)
            if bool(term[0]):
                falls += 1
            if i >= 100:
                q = r.data.joint_pos.torch[0, env.active_ids]
                errs.append((q - act[0]).abs() / D)
                z.append(float(r.data.root_link_pos_w.torch[0, 2]))
        tau = torch.stack(servo.log_tau)[:, 0]
        avail = torch.stack(servo.log_avail)[:, 0]
        names = servo.joint_names
        order = [names.index(j) for j in ACTIVE_JOINTS]
        tau, avail = tau[:, order], avail[:, order]
        e = torch.stack(errs).cpu()
        e_max = e.max(0).values
        sat = (tau.abs() >= 0.95 * avail).float().mean(0) * 100
        tmax = tau.abs().max(0).values
        trms = tau.pow(2).mean(0).sqrt()
        ok = falls == 0 and float(e_max.max()) < 2.0 and float(sat.max()) == 0.0
        lines = [f"stand test {time.strftime('%Y-%m-%d %H:%M:%S')}: variant {a.variant}, {a.seconds:.0f} s, 1 env, no "
                 f"harness, no viewer, Isaac Sim PhysX dt 1 ms, control 100 Hz, servo latency {cfg.scene.robot.actuators['servos'].min_delay}-"
                 f"{cfg.scene.robot.actuators['servos'].max_delay} ms, backlash per robot.yaml, no randomization",
                 f"falls {falls}; base height {1000 * min(z):.1f}…{1000 * max(z):.1f} mm",
                 "joint            max|err|°   max|tau| N·m   rms tau N·m   saturated %"]
        for k, j in enumerate(ACTIVE_JOINTS):
            lines.append(f"{j:16s} {float(e_max[k]):9.2f}   {float(tmax[k]):12.4f}   {float(trms[k]):11.4f}   {float(sat[k]):10.2f}")
        lines.append(f"RESULT: {'PASS' if ok else 'FAIL'} (criteria: no fall, tracking error < 2°, no saturation)")
        (out / f"result_{a.variant}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
