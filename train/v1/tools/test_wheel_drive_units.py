"""General rule 1: measure what the wheel-axle drive gains written at runtime actually do (fixed base, wheels
in the air, zero gravity, Isaac Sim PhysX dt 1 ms, 1 env).

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/v1/tools/test_wheel_drive_units.py

1 damping: spin the wheel to w0 with stiffness 0 and damping D written via write_joint_damping_to_sim_index;
  w(t) = w0 exp(-D_eff t / I)  ->  D_eff = I * ln(w0 / w(t)) / t, compared with the D written.
2 stiffness: hold target 0, start the wheel at 0.2 rad, stiffness K, small damping; the oscillation period gives
  K_eff = I (2 pi / T)^2, compared with the K written.
I is the wheel link's inertia about the axle as reported by the simulation.
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


def main():
    cfg = MoonwalkEnvCfg()
    cfg.variant = "wheel"
    cfg.fix_root = True
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.finalize()
    out = VDIR / "runs" / "wheel_drive_units"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"wheel drive unit test {time.strftime('%Y-%m-%d %H:%M:%S')}: fixed base, wheels in the air, zero gravity, "
             "Isaac Sim PhysX dt 1 ms, 1 env"]
    with launch_simulation(cfg.sim, a):
        env = MoonwalkEnv(cfg)
        env.reset()
        env.clutch = None                                  # this test drives the wheel gains directly
        r = env.robot
        j = r.joint_names.index("L_wheel_axle")
        b = r.body_names.index("L_wheel")
        jt = torch.tensor([j], device=env.device)
        inert = r.data.default_inertia.torch[0, b].reshape(3, 3) if r.data.default_inertia.torch.dim() == 3 else \
            r.data.default_inertia.torch[0, b].reshape(3, 3)
        I = float(inert[0, 0])                             # about the link x axis (= axle)
        lines.append(f"wheel link inertia about the axle (sim): {I:.3e} kg·m²")
        act = env.q_stand.unsqueeze(0)

        def set_w(pos, vel, k, d):
            jp = r.data.joint_pos.torch.clone()
            jv = torch.zeros_like(jp)
            jp[0, j], jv[0, j] = pos, vel
            r.write_joint_position_to_sim_index(position=jp)
            r.write_joint_velocity_to_sim_index(velocity=jv)
            r.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[k]], device=env.device), joint_ids=jt)
            r.write_joint_damping_to_sim_index(damping=torch.tensor([[d]], device=env.device), joint_ids=jt)

        def run_steps(n, every=1):
            out_ = []
            for i in range(n):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                if i % every == 0:
                    out_.append((float(r.data.joint_pos.torch[0, j]), float(r.data.joint_vel.torch[0, j])))
            return out_

        for D in (1e-6, 3e-6, 1e-5, 3e-5, 1.28e-4):
            set_w(0.0, 10.0, 0.0, D)
            tr = run_steps(60)
            w = [v for _, v in tr]
            # implicit (backward-Euler) damping: w[n+1] = w[n] / (1 + D_eff dt / I); fit on samples with w > 0.05
            ratios = [w[i + 1] / w[i] for i in range(2, len(w) - 1) if w[i] > 0.05 and w[i + 1] > 0]
            if ratios:
                rr = sum(ratios) / len(ratios)
                D_eff = (1 / rr - 1) * I / env.physics_dt
                lines.append(f"damping written {D:.2e} N·m·s/rad: per-step ratio {rr:.4f} over {len(ratios)} steps -> "
                             f"D_eff {D_eff:.3e} (ratio {D_eff / D:.3f}); w after 10/60 ms {w[9]:.3f}/{w[-1]:.3f} rad/s")
            else:
                lines.append(f"damping written {D:.2e}: w fell below 0.05 rad/s within 3 ms (w {w[:4]})")
        for K in (1e-4, 1e-3):
            set_w(0.2, 0.0, K, 0.0)
            tr = run_steps(400)
            pos = [p_ for p_, _ in tr]
            cross = [i for i in range(1, len(pos)) if pos[i - 1] > 0 >= pos[i] or pos[i - 1] < 0 <= pos[i]]
            if len(cross) >= 3:
                T = 2 * (cross[2] - cross[1]) * env.physics_dt
                K_eff = I * (2 * math.pi / T) ** 2
                lines.append(f"stiffness written {K:.1e} N·m/rad: period {T * 1000:.1f} ms -> K_eff {K_eff:.3e} "
                             f"(ratio {K_eff / K:.3f})")
            else:
                lines.append(f"stiffness written {K:.1e} N·m/rad: no oscillation in 0.4 s (pos {pos[0]:.3f} -> "
                             f"{pos[-1]:.3f} rad)")
        (out / "result.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
