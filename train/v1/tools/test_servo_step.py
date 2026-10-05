"""Step 2: single-joint step response of the servo model (fixed base, zero gravity = no-load speed).

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/test_servo_step.py

Checks (results to runs/servo_step/result.txt):
  - actuator compute() runs once per physics step (call count == physics steps)
  - peak speed of a large step ≈ 7.64 rad/s (no-load speed; the linear torque-speed curve makes it asymptotic,
    criterion within 5 %)
  - applied torque never exceeds 0.88 N·m and reaches it at the start of the step (stall)
  - command latency = configured physics steps
"""
import time

from _boot import VDIR, finish_args, parser

p = parser(__doc__.splitlines()[0])
p.add_argument("--joint", default="L_hip_pitch")
a = finish_args(p)

from isaaclab.app import launch_simulation  # noqa: E402

import math  # noqa: E402

import torch  # noqa: E402

from lib.env import MoonwalkEnv  # noqa: E402
from lib.env_cfg import ACTIVE_JOINTS, MoonwalkEnvCfg  # noqa: E402

D = math.pi / 180


def main():
    cfg = MoonwalkEnvCfg()
    cfg.variant = a.variant
    cfg.fix_root = True
    cfg.latency_ms = (7, 7)                         # fixed latency so it can be measured
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.finalize()
    out = VDIR / "runs" / "servo_step"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    with launch_simulation(cfg.sim, a):
        env = MoonwalkEnv(cfg)
        env.reset()
        j = ACTIVE_JOINTS.index(a.joint)
        lo, hi = {"hip": (-30, 90), "knee": (0, 120), "ankle": (-35, 30)}[a.joint.split("_")[1]]
        start, goal = (lo + 5) * D, (hi - 5) * D
        act = env.q_stand.clone().unsqueeze(0)
        act[0, j] = start
        for _ in range(100):
            env.step(act)
        servo = env.servo
        servo.log, servo.log_tau, servo.log_vel = True, [], []
        calls0, steps0 = servo.calls, env.physics_steps
        act[0, j] = goal
        for _ in range(60):                              # 0.6 s
            env.step(act)
        sj = [i for i, n in enumerate(servo.joint_names) if n == a.joint][0]
        tau = torch.stack(servo.log_tau)[:, 0, sj].cpu()
        vel = torch.stack(servo.log_vel)[:, 0, sj].cpu()
        calls, steps = servo.calls - calls0, env.physics_steps - steps0
        moved = (tau.abs() > 1e-6).nonzero()
        lag = int(moved[0]) if len(moved) else -1
        vmax, tmax = float(vel.abs().max()), float(tau.abs().max())
        t_start = float(tau[lag:lag + 3].abs().max()) if lag >= 0 else 0.0
        ok_calls = calls == steps
        ok_v = abs(vmax - 7.64) / 7.64 < 0.05
        ok_t = tmax <= 0.88 + 1e-6 and t_start > 0.85
        ok_lag = lag == 7
        lines += [
            f"servo step test {time.strftime('%Y-%m-%d %H:%M:%S')}: joint {a.joint}, variant {a.variant}, fixed base, "
            f"zero gravity, Isaac Sim PhysX dt 1 ms, control 100 Hz, latency set to 7 ms, 1 env, no harness",
            f"[{'PASS' if ok_calls else 'FAIL'}] actuator calls {calls} over {steps} physics steps",
            f"[{'PASS' if ok_v else 'FAIL'}] peak speed {vmax:.3f} rad/s ({vmax / D:.0f}°/s), spec 7.64 rad/s",
            f"[{'PASS' if ok_t else 'FAIL'}] peak applied torque {tmax:.4f} N·m (limit 0.88), torque in the first "
            f"steps after the command arrives {t_start:.4f} N·m (stall)",
            f"[{'PASS' if ok_lag else 'FAIL'}] torque starts {lag} physics steps after the step command (7 ms set)",
        ]
        (out / "trace.csv").write_text("step,tau_Nm,vel_rad_s\n" + "\n".join(
            f"{i},{float(t):.5f},{float(v):.5f}" for i, (t, v) in enumerate(zip(tau, vel))))
        allok = ok_calls and ok_v and ok_t and ok_lag
        lines.append("RESULT: " + ("PASS" if allok else "FAIL"))
        (out / f"result_{a.joint}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
