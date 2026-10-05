"""Fold a run_gait trace into one gait cycle: static margin, left-leg joint angles and |servo torque| per phase bin.

    ~/Desktop/IsaacLab/_isaac_sim/python.sh ~/Desktop/moonwalk/train/v1/tools/cycle_profile.py <runs/gait/... folder> [--bins 80]

The simulated counterpart of the phase-1 cycle figure (cad/out/analysis/cycle_<variant>.png). Pure numpy, no simulator.
Cycle time 0 = slide_start of the half in which the LEFT leg is the toe-raised support (A); the right leg is A in the
second half. Only steady-state cycles are used (the first two half cycles after start-up are skipped, as validate.py).
Unlike the phase-1 plot, every curve here is the physical left leg for the whole cycle.

Per bin (cycle_time / bins wide), across all steady-state cycles:
  margin  fore-aft / lateral (mm): min, median, max over the control steps (100 Hz) in the bin; the polygon and the CoM
          are computed exactly as validate.py (geometric contact of the collision-primitive bottoms, any load).
          Steps whose CoM line misses the polygon (validate.py reports -1000) are counted, not plotted.
  angle   L hip / knee / ankle (deg): min, median, max over the control steps in the bin.
  torque  |tau| L hip / knee / ankle (N·m): median, p99, max over the physics steps (1 kHz) in the bin.
Writes <run>/cycle_profile.csv (long format: t_s, panel, series, lo, mid, hi) and prints a summary.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

VDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VDIR))
from lib.contact import FootContacts, hull, margins  # noqa: E402
from lib.snapshot import robot as load_robot  # noqa: E402

D = 180 / math.pi
TOL, FMIN = 0.0005, -1.0          # same polygon rule as validate.py
LEFT = {"hip": 0, "knee": 1, "ankle": 2}   # column in q / tau (ACTIVE_JOINTS order: L hip, L knee, L ankle, R ...)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bins", type=int, default=80)
    a = ap.parse_args()
    folder = Path(a.run)
    z = np.load(folder / "trace.npz", allow_pickle=True)
    meta = json.loads((folder / "summary.json").read_text())
    robot = load_robot()
    variant = meta["variant"]
    T = float(z["cycle"])
    T2 = T / 2
    t0 = float(z["t_startup"])
    t = z["t"]
    # cycle time: start-up ends at slide_mid = 0.25 of a half (gait_fsm.GaitFSM.phase); half 0 has L as support
    tg = t - t0 + 0.25 * T2
    steady = (t >= t0) & (tg >= 2 * T2)
    ph = np.mod(tg, T)
    edges = np.linspace(0.0, T, a.bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    bin_c = np.clip(np.digitize(ph, edges) - 1, 0, a.bins - 1)

    # --- static margin per control step (validate.py rule)
    fc = FootContacts(robot, variant)
    feet, fq, com = z["feet"], z["foot_quat"], z["com"]
    wheels = z["wheels"] if variant == "wheel" else None
    f_feet = z["f_feet"]
    f_wheels = z["f_wheels"] if variant == "wheel" else None
    fa = np.full(len(t), np.nan)
    la = np.full(len(t), np.nan)
    miss = 0
    for i in np.where(steady)[0]:
        pts = []
        for k in range(2):
            P = fc.points(feet[i, k], fq[i, k], wheels[i, k] if wheels is not None else None)
            for zone, Q in P.items():
                force = f_wheels[i, k] if zone == "wheel" else f_feet[i, k]
                touch = Q[Q[:, 2] < TOL] if force > FMIN else Q[:0]
                pts += [q[:2] for q in touch]
        if len(pts) >= 3:
            f_, l_ = margins(hull(np.array(pts)), com[i, :2])
            if f_ <= -0.999 or l_ <= -0.999:
                miss += 1
                continue
            fa[i], la[i] = f_ * 1000, l_ * 1000

    rows = []

    def agg_ctrl(panel, series, v):
        for b in range(a.bins):
            m = steady & (bin_c == b) & ~np.isnan(v)
            if m.any():
                rows.append((mids[b], panel, series, float(v[m].min()), float(np.median(v[m])), float(v[m].max())))

    agg_ctrl("margin", "fore-aft", fa)
    agg_ctrl("margin", "lateral", la)
    q = z["q"]
    for name, k in LEFT.items():
        agg_ctrl("angle", name, q[:, k] * D)

    # --- torque per physics step
    tau = z["tau"]
    skip = int(round(t0 / 0.001))
    tau = tau[skip:]
    tg_p = np.arange(len(tau)) * 0.001 + 0.25 * T2
    steady_p = tg_p >= 2 * T2
    bin_p = np.clip(np.digitize(np.mod(tg_p, T), edges) - 1, 0, a.bins - 1)
    for name, k in LEFT.items():
        v = np.abs(tau[:, k])
        for b in range(a.bins):
            m = steady_p & (bin_p == b)
            if m.any():
                rows.append((mids[b], "torque", name, float(np.median(v[m])), float(np.percentile(v[m], 99)),
                             float(v[m].max())))

    out = folder / "cycle_profile.csv"
    with out.open("w") as f:
        f.write("t_s,panel,series,lo,mid,hi\n")
        for r in rows:
            f.write(f"{r[0]:.4f},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")
    n_cyc = (tg[steady].max() - 2 * T2) / T
    print(f"[cycle_profile] {folder.name}: {variant}, cycle {T} s, {n_cyc:.1f} steady-state cycles, {a.bins} bins")
    print(f"[cycle_profile] margin: min fore-aft {np.nanmin(fa):.1f} mm, min lateral {np.nanmin(la):.1f} mm; "
          f"CoM line missed the polygon in {miss} control steps (not plotted)")
    for name, k in LEFT.items():
        v = np.abs(tau[steady_p, k])
        print(f"[cycle_profile] L {name}: angle {q[steady, k].min() * D:.1f} … {q[steady, k].max() * D:.1f}°, "
              f"|tau| p99 {np.percentile(v, 99):.3f} max {v.max():.3f} N·m")
    print(f"[cycle_profile] -> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
