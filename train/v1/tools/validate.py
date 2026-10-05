"""Acceptance table from a run_gait trace (train/CLAUDE.md "Acceptance criteria"). Pure numpy, no simulator.

    ~/Desktop/IsaacLab/_isaac_sim/python.sh ~/Desktop/moonwalk/train/v1/tools/validate.py <runs/gait/... folder>

Sign conventions: backward distance = -(Δy) > 0 when moving toward -Y. Support-foot slip is reported forward (+Y)
and backward (-Y) separately. Steady state = half cycles after the first two.
Static margin uses the collision-primitive bottoms geometrically touching the ground (within 0.5 mm, any load) and
the CoM projection, measured along the Y / X lines through the CoM like phase 1. The share of time one foot carries
< 0.3 N is reported separately: the light foot is the sliding (or about-to-slide) one, and the closed leg chain
carries the resulting roll moment through lateral foot friction, so load-gated polygons would be misleading.
"""
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
JOINTS = ["L_hip_pitch", "L_knee_pitch", "L_ankle_pitch", "R_hip_pitch", "R_knee_pitch", "R_ankle_pitch"]
TOL, FMIN, F_UNLOAD = 0.0005, -1.0, 0.3     # polygon = geometric contact (any load); unloading reported separately


def sizing_margins(variant):
    """Phase-1 minimum margins (mm) for comparison, from the copy of phase1.json if present."""
    p = VDIR.parents[1] / "cad" / "out" / "analysis" / "phase1.json"
    if p.exists():
        d = json.loads(p.read_text()).get(variant, {})
        return d.get("fa"), d.get("lat")
    return None, None


def main(folder):
    folder = Path(folder)
    z = np.load(folder / "trace.npz", allow_pickle=True)
    meta = json.loads((folder / "summary.json").read_text())
    robot = load_robot()
    variant = meta["variant"]
    t, T2 = z["t"], float(z["cycle"]) / 2
    s_mm = float(z["step"])
    t0 = float(z["t_startup"])
    gi = t >= t0
    fc = FootContacts(robot, variant)
    feet, fq, com = z["feet"], z["foot_quat"], z["com"]          # all world frame
    wheels = z["wheels"] if variant == "wheel" else None
    f_feet = z["f_feet"]
    f_wheels = z["f_wheels"] if variant == "wheel" else None
    supp, phase = z["support"], z["phase"]
    thA, thB = z["theta_A"], z["theta_B"]
    face = robot["geometry"]["foot"]["toe_pad"]["face_angle"] / D
    rows = []
    # --- survival
    rows.append(("Standing time (no harness, no viewer)", f"{meta['survived_s']:.1f} s of {meta['gait_seconds_requested']:.0f} s"
                 f"{', fell' if meta['fell'] else ', no fall'}", "60 s without falling", not meta["fell"] and meta["survived_s"] >= 59.9))
    # --- per-half-cycle slide distance and support slip
    tg = t - t0 + 0.25 * T2                         # cycle time (start-up ends at slide_mid = 0.25 of a half)
    halves = np.floor(tg / T2).astype(int)
    slides = {"L": [], "R": []}
    slip_f, slip_b, roll = [], [], {"L": [], "R": []}
    for h in range(2, halves.max() if gi.any() else 0):
        idx = np.where((halves == h) & gi)[0]
        if len(idx) < 10:
            continue
        tin = tg[idx] - h * T2
        a_ = idx[0]
        b_ = idx[np.searchsorted(tin, 0.5 * T2) - 1]               # end of the slide (slide_end keyframe)
        sl = "R" if h % 2 == 0 else "L"
        sp = "L" if sl == "R" else "R"
        k_sl, k_sp = (0 if sl == "L" else 1), (0 if sp == "L" else 1)
        slides[sl].append(-(feet[b_, k_sl, 1] - feet[a_, k_sl, 1]) * 1000)
        # support-foot slip during the slide only (spec metric)
        d = (feet[a_:b_ + 1, k_sp, 1] - feet[a_, k_sp, 1]) * 1000
        slip_f.append(max(0.0, d.max()))
        slip_b.append(max(0.0, -d.min()))
        # weight transfer + role swap (second part of the half): how far each foot rolls (step 6 risk)
        c_ = idx[-1]
        for k, side in ((k_sl, sl), (k_sp, sp)):
            dd = (feet[b_:c_ + 1, k, 1] - feet[b_, k, 1]) * 1000
            roll[side].append((float(dd.min()), float(dd.max())))
    all_sl = slides["L"] + slides["R"]
    mean_sl = float(np.mean(all_sl)) if all_sl else float("nan")
    rows.append(("Distance per slide (steady state)", f"{mean_sl:.1f} mm mean over {len(all_sl)} slides "
                 f"(min {min(all_sl):.1f}, max {max(all_sl):.1f}); target {s_mm:.0f} mm" if all_sl else "n/a",
                 "> 90 % of target", bool(all_sl) and mean_sl > 0.9 * s_mm))
    if slides["L"] and slides["R"]:
        mL, mR = np.mean(slides["L"]), np.mean(slides["R"])
        asym = abs(mL - mR) / max(mL, mR) * 100
        rows.append(("Left-right symmetry", f"slide L {mL:.1f} / R {mR:.1f} mm → {asym:.1f} %; cycle time identical "
                     f"by construction (open-loop timing) → 0 %", "< 10 % both", asym < 10))
    rb = [-m for v in roll.values() for m, _ in v]
    rf = [M for v in roll.values() for _, M in v]
    rows.append(("Weight transfer: foot roll-back (step 6 risk)", f"during transfer + role swap each foot moves at most "
                 f"{max(rb):.2f} mm backward (mean worst {np.mean(rb):.2f}) and {max(rf):.2f} mm forward"
                 if rb else "n/a", "reported (no spec threshold); compare with slip limit 4 mm", True))
    rows.append(("Support foot slip (per slide)", f"forward max {max(slip_f):.2f} mm, backward max {max(slip_b):.2f} mm"
                 f" (mean {np.mean(slip_f):.2f} / {np.mean(slip_b):.2f})" if slip_f else "n/a",
                 f"< 10 % of step ({0.1 * s_mm:.0f} mm)", bool(slip_f) and max(max(slip_f), max(slip_b)) < 0.1 * s_mm))
    # --- static margin and zone switching
    fa_all, lat_all, wrong, rel, unloaded = [], [], 0, 0, 0
    for i in np.where(gi)[0]:
        pts = []
        for k, side in enumerate(("L", "R")):
            P = fc.points(feet[i, k], fq[i, k], wheels[i, k] if wheels is not None else None)
            for zone, Q in P.items():
                force = f_wheels[i, k] if zone == "wheel" else f_feet[i, k]
                touch = Q[Q[:, 2] < TOL] if force > FMIN else Q[:0]
                pts += [q[:2] for q in touch]
            # zone switching: toe state -> wheels/sole lifted; flat state -> pad lifted
            th = thA[i] if supp[i] == side else thB[i]
            if th > face - 1.0 / D:
                rel += 1
                low = P["wheel"][:, 2].min() if "wheel" in P else P["sole"][:, 2].min()
                wrong += low < TOL
            elif th < 1.0 / D:
                rel += 1
                wrong += P["toe_pad"][:, 2].min() < TOL
        load = [f_feet[i, k] + (f_wheels[i, k] if f_wheels is not None else 0.0) for k in range(2)]
        unloaded += min(load) < F_UNLOAD
        if len(pts) >= 3:
            fa, la = margins(hull(np.array(pts)), com[i, :2])
            fa_all.append(fa * 1000)
            lat_all.append(la * 1000)
    fa_ph1, la_ph1 = sizing_margins(variant)
    if fa_all:
        fa_a, la_a = np.array(fa_all), np.array(lat_all)
        rows.append(("Static stability margin (slow version)", f"min fore-aft {fa_a.min():.1f} mm, min lateral "
                     f"{la_a.min():.1f} mm (p1 {np.percentile(fa_a, 1):.1f} / {np.percentile(la_a, 1):.1f}, p5 "
                     f"{np.percentile(fa_a, 5):.1f} / {np.percentile(la_a, 5):.1f}; -1000 = CoM line misses the "
                     f"polygon); one foot carrying < {F_UNLOAD} N for {100 * unloaded / max(gi.sum(), 1):.1f} % of the "
                     f"time; "
                     f"phase 1 predicted {fa_ph1:.1f} / {la_ph1:.1f} mm" if fa_ph1 else "",
                     "≥ 10 mm both", min(fa_all) >= 10 and min(lat_all) >= 10))
    rows.append(("Foot zone switching", f"wrong contact {100 * wrong / max(rel, 1):.1f} % of {rel} foot-steps in a "
                 f"toe or flat state", "< 5 %", wrong / max(rel, 1) < 0.05))
    # --- tracking, torque, speed
    q, qt = z["q"][gi], z["q_tgt"][gi]
    err = np.abs(q - qt) * D
    rows.append(("Joint tracking error", f"max {err.max():.2f}°, p99 {np.percentile(err, 99):.2f}° (worst joint "
                 f"{JOINTS[int(err.max(0).argmax())]})", "< 5°", err.max() < 5))
    nphys = int(round(0.01 / 0.001))
    tau, vel, avail = z["tau"], z["vel"], z["avail"]
    skip = int(t0 / 0.001)
    tau, vel, avail = tau[skip:], vel[skip:], avail[skip:]
    rms = np.sqrt((tau ** 2).mean(0))
    p99 = np.percentile(np.abs(tau), 99, axis=0)
    sat = (np.abs(tau) >= 0.95 * avail).mean(0) * 100
    ok_t = (rms <= 0.216).all() and (p99 <= 0.62).all() and (sat < 2).all()
    rows.append(("Torque (per joint, per physics step)", "; ".join(f"{j} rms {r_:.3f} p99 {p_:.3f} sat {s_:.2f}%"
                 for j, r_, p_, s_ in zip(JOINTS, rms, p99, sat)), "rms ≤ 0.216, p99 ≤ 0.62 N·m, sat < 2 %", ok_t))
    vp99 = np.percentile(np.abs(vel), 99, axis=0) * D
    rows.append(("Joint speed", f"p99 max {vp99.max():.0f}°/s ({JOINTS[int(vp99.argmax())]})", "p99 < 307°/s",
                 vp99.max() < 307))
    lines = [f"# Acceptance — {folder.name}", "",
             f"Conditions: {meta['conditions']}; ankle mode {meta.get('ankle_mode')}; cycle {meta['cycle_s']} s, step "
             f"{meta['step_mm']} mm; {len(t)} control steps; backward distance {meta['backward_distance_mm']:.0f} mm "
             f"(positive = moved toward -Y). Not measured here: robustness (randomized multi-env runs) and the visual "
             f"heel-toe comparison (video).", "",
             "| Metric | Measured | Criterion | Pass |", "|---|---|---|---|"]
    for name, val, crit, ok in rows:
        lines.append(f"| {name} | {val} | {crit} | {'✔' if ok else '✘'} |")
    (folder / "acceptance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
