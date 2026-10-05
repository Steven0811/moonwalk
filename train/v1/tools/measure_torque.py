"""Torque, speed and saturation per joint from a run_gait trace (logged at every physics step, 1 kHz).

    ~/Desktop/IsaacLab/_isaac_sim/python.sh ~/Desktop/moonwalk/train/v1/tools/measure_torque.py <runs/gait/... folder>

Reports per joint: RMS and p99 |tau| (budget: RMS ≤ 0.216 N·m rated load, p99 ≤ 0.62 N·m = 70 % stall), time at
≥ 95 % of the available torque (< 2 %), p99 |speed| (< 307°/s), and for each ankle the mean signed torque while its
foot is the toe-raised support (root CLAUDE.md design rule 1: that ankle holds torque continuously, so its mean
must stay well below the rated load). Gait part only (start-up excluded).
"""
import json
import sys
from pathlib import Path

import numpy as np

JOINTS = ["L_hip_pitch", "L_knee_pitch", "L_ankle_pitch", "R_hip_pitch", "R_knee_pitch", "R_ankle_pitch"]
D = 180 / np.pi


def main(folder):
    folder = Path(folder)
    z = np.load(folder / "trace.npz", allow_pickle=True)
    meta = json.loads((folder / "summary.json").read_text())
    t0 = float(z["t_startup"])
    skip = int(round(t0 / 0.001))
    tau, vel, avail = z["tau"][skip:], z["vel"][skip:], z["avail"][skip:]
    n = len(tau)
    # support side per physics step (control-step label repeated 10x)
    supp = np.repeat(z["support"], 10)[skip:skip + n]
    phase = np.repeat(z["phase"], 10)[skip:skip + n]
    toe_phases = {"slide_start", "slide_mid", "slide_end"}       # A is toe-raised during the slide
    lines = [f"# Torque — {folder.name}", "",
             f"Conditions: {meta['conditions']}; cycle {meta['cycle_s']} s; {n} physics steps (gait only).", "",
             "| Joint | RMS N·m | p99 N·m | max N·m | ≥95 % avail | p99 speed °/s | mean signed τ while toe-raised support |",
             "|---|---|---|---|---|---|---|"]
    ok = True
    for k, j in enumerate(JOINTS):
        a = np.abs(tau[:, k])
        rms, p99, mx = np.sqrt((tau[:, k] ** 2).mean()), np.percentile(a, 99), a.max()
        sat = (a >= 0.95 * avail[:, k]).mean() * 100
        vp = np.percentile(np.abs(vel[:, k]), 99) * D
        hold = ""
        if "ankle" in j:
            m = (supp == j[0]) & np.isin(phase, list(toe_phases))
            if m.any():
                hold = f"{tau[m, k].mean():+.3f} N·m ({100 * abs(tau[m, k].mean()) / 0.216:.0f} % of rated)"
        ok &= rms <= 0.216 and p99 <= 0.62 and sat < 2 and vp < 307
        lines.append(f"| {j} | {rms:.3f} | {p99:.3f} | {mx:.3f} | {sat:.2f} % | {vp:.0f} | {hold} |")
    lines += ["", f"Budget (RMS ≤ 0.216, p99 ≤ 0.62 N·m, saturated < 2 %, p99 speed < 307°/s): {'PASS' if ok else 'FAIL'}"]
    (folder / "torque.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
