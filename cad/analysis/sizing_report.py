"""Phase-1 sizing feasibility: sweeps + the feasibility report cad/out/sizing_report.md.

    conda run -n cad python cad/analysis/sizing_report.py            # full report
    conda run -n cad python cad/analysis/sizing_report.py --quick    # skip the big factorial

The report documents the design currently in config/ (the "selected design") and the sweeps around it.
Values the analysis derives (hip height, pelvis keyframes, standing pose) are compared with gait.yaml and
flagged if they differ, so config/ and the report cannot silently drift apart.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gait import GaitModel                                   # noqa: E402
from kinematics import D2R, contacts, foot_geom, foot_pose   # noqa: E402
from model import ROOT, load, variant as make_variant       # noqa: E402
from statics import _ray_extent                              # noqa: E402

OUT = ROOT / "cad" / "out"
FIG = OUT / "analysis"
RATED, P99_SIM, STALL = 0.216, 0.62, 0.88
KNEE_BACK = (15.0, 30.0)          # knee of the sliding foot at the back end of the slide
KNEE_TARGET = 27.0
MARGIN_MIN = 10.0                 # mm
STAND_KNEE = 25.0
PH1: dict = {}                    # machine-readable summary for cad/checks.py


def hip_height_for(gm: GaitModel, pel, knee_deg: float) -> float:
    """Hip height giving `knee_deg` at the sliding leg when it is flat at the back end of the slide."""
    kf = [k for k in gm.kf if k["name"] == "slide_end"][0]
    i = gm.kf.index(kf)
    ankle, _ = foot_pose(gm.fg, kf["B_y"] * gm.s, 0.0)
    dy = ankle[0] - pel[i]
    L1, L2 = gm.leg.L1, gm.leg.L2
    d = np.sqrt(L1**2 + L2**2 + 2 * L1 * L2 * np.cos(knee_deg * D2R))
    return ankle[1] + np.sqrt(max(d * d - dy * dy, 0.0))


def evaluate(overrides: dict, var: str, knee_target=KNEE_TARGET, torque=True, n=60, fixed_H=None,
             elec_factor=1.0, battery_dy=0.0):
    robot, gait = load()
    robot, gait = make_variant(robot, gait, **overrides)
    out = {"overrides": overrides, "variant": var}
    try:
        gm = GaitModel(robot, gait, var, elec_factor, battery_dy)
        pel = np.zeros(len(gm.kf) - 1)
        for _ in range(3):
            gm.H = fixed_H if fixed_H is not None else hip_height_for(gm, pel, knee_target)
            pel, worst = gm.optimise_pelvis()
        samples = gm.sample(pel, n)
    except AssertionError as e:
        out.update(ok=False, why=f"geometry: {e}")
        return out
    feas = all(s.feasible for s in samples)
    bad = [s for s in samples if not s.feasible]
    m = [s.margin for s in samples if s.margin]
    out.update(
        H=gm.H, pelvis=pel.tolist(), mass=gm.mm.total, feasible=feas,
        why=(bad[0].violation + f" @ {bad[0].phase}") if bad else "",
        fa=min(x["fore_aft"] for x in m) if m else -1e3,
        lat=min(x["lateral"] for x in m) if m else -1e3,
        euclid=min(x["euclid"] for x in m) if m else -1e3,
        com_z=float(np.mean([s.com[2] for s in samples if s.legs])),
    )
    se = [s for s in samples if s.phase == "slide_end" and s.half == 0][0]
    out["knee_back"] = se.legs[1].qk / D2R if se.legs else np.nan
    if samples[0].legs:
        q = np.array([[lg.qh, lg.qk, lg.qa] for s in samples if s.legs for lg in s.legs]) / D2R
        out["q_min"], out["q_max"] = q.min(0).tolist(), q.max(0).tolist()
    if torque and feas:
        gm.torques(samples)
        tb = np.array([[abs(v) for v in s.tau.values()] for s in samples if s.tau])
        tw = np.array([[v for v in s.tau_worst.values()] for s in samples if s.tau_worst])
        # columns: leg0 hip, knee, ankle, leg1 hip, knee, ankle -> per joint over both legs
        out["tau_best_p99"] = [float(np.percentile(np.r_[tb[:, j], tb[:, j + 3]], 99)) for j in range(3)]
        out["tau_worst_p99"] = [float(np.percentile(np.r_[tw[:, j], tw[:, j + 3]], 99)) for j in range(3)]
        out["tau_worst_max"] = [float(np.r_[tw[:, j], tw[:, j + 3]].max()) for j in range(3)]
    out["ok"] = bool(
        feas and min(out["fa"], out["lat"]) >= MARGIN_MIN and KNEE_BACK[0] <= out["knee_back"] <= KNEE_BACK[1]
        and (not torque or max(out.get("tau_worst_p99", [9])) <= RATED))
    return out


def _eval_star(a):
    return evaluate(*a[0], **a[1])


def run_pool(jobs):
    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        return p.map(_eval_star, jobs)


# --------------------------------------------------------------------------- figures
def figures(robot, gait, var, pel, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    gm = GaitModel(robot, gait, var)
    gm.H = H
    ss = gm.torques(gm.sample(np.array(pel), 80))
    t = np.array([s.t + s.half for s in ss]) * gait["cycle_time"] / 2
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, [s.margin["fore_aft"] for s in ss], label="fore-aft")
    ax[0].plot(t, [s.margin["lateral"] for s in ss], label="lateral")
    ax[0].plot(t, [s.margin["euclid"] for s in ss], ":", label="euclidean (info)")
    ax[0].axhline(MARGIN_MIN, color="r", lw=0.8)
    ax[0].set_ylabel("static margin [mm]")
    ax[0].legend(fontsize=8)
    names = ["hip", "knee", "ankle"]
    for j in range(3):
        ax[1].plot(t, [s.legs[0].__dict__[["qh", "qk", "qa"][j]] / D2R for s in ss], label=f"L {names[j]}")
    ax[1].set_ylabel("left joint angle [deg]")
    ax[1].legend(fontsize=8)
    for j in range(3):
        ax[2].plot(t, [list(s.tau_worst.values())[j] for s in ss], label=f"L {names[j]} worst")
        ax[2].plot(t, [abs(list(s.tau.values())[j]) for s in ss], "--", label=f"L {names[j]} best")
    ax[2].axhline(RATED, color="r", lw=0.8)
    ax[2].set_ylabel("|static torque| [N·m]")
    ax[2].set_xlabel("time [s] (4 s cycle)")
    ax[2].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Static analysis over one cycle — {var} foot")
    fig.tight_layout()
    fig.savefig(FIG / f"cycle_{var}.png", dpi=110)
    plt.close(fig)

    # support polygons at keyframes
    keys = [k for k in gm.kf[:-1]]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4.2))
    for a, k in zip(axs, keys):
        s = gm.pose(k["t"], np.array(pel), 0)
        h = np.vstack([s.hull, s.hull[:1]])
        a.fill(h[:, 0], h[:, 1], alpha=0.25)
        for lg in s.legs:
            a.plot([p[0] for p in lg.contacts], [p[1] for p in lg.contacts], "k.", ms=4)
        a.plot(s.com[0], s.com[1], "r+", ms=12)
        a.set_title(f"{k['name']}\nfa {s.margin['fore_aft']:.1f} / lat {s.margin['lateral']:.1f} mm", fontsize=8)
        a.set_aspect("equal")
        a.set_xlim(-60, 60)
        a.set_ylim(-110, 70)
        a.grid(alpha=0.3)
    fig.suptitle(f"Support polygon (x right, y forward) and CoM — {var}")
    fig.tight_layout()
    fig.savefig(FIG / f"polygons_{var}.png", dpi=110)
    plt.close(fig)

    # side-view stick figures at keyframes
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4.2))
    for a, k in zip(axs, keys):
        s = gm.pose(k["t"], np.array(pel), 0)
        for lg, col in zip(s.legs, ("tab:blue", "tab:orange")):
            knee = lg.hip + np.array([np.sin(lg.qh), -np.cos(lg.qh)]) * gm.leg.L1
            a.plot([lg.hip[0], knee[0], lg.ankle[0]], [lg.hip[1], knee[1], lg.ankle[1]], "-o", color=col, ms=3)
            R = np.array([[np.cos(lg.pitch), -np.sin(lg.pitch)], [np.sin(lg.pitch), np.cos(lg.pitch)]])
            fg = gm.fg
            pts = [(-fg.heel, -fg.h + (fg.heel_clear if fg.variant == "wheel" else 0)), tuple(fg.pad_rear),
                   tuple(fg.pad_tip), (fg.toe, 0.0), (0.0, 0.0), (-fg.heel, 0.0)]
            P = np.array([lg.ankle + R @ np.array(p) for p in pts + pts[:1]])
            a.plot(P[:, 0], P[:, 1], color=col, lw=1)
            if fg.variant == "wheel":
                c = lg.ankle + R @ np.array([fg.wheel_y, -fg.h + fg.wheel_r])
                a.add_patch(plt.Circle(c, fg.wheel_r, fill=False, color=col))
        a.axhline(0, color="k", lw=0.8)
        a.plot(s.com[1], s.com[2], "r+", ms=10)
        a.set_aspect("equal")
        a.set_xlim(-90, 90)
        a.set_ylim(-5, 215)
        a.set_title(k["name"], fontsize=9)
    fig.suptitle(f"Side view (y forward →) — {var}; blue = support leg A, orange = sliding leg B")
    fig.tight_layout()
    fig.savefig(FIG / f"keyframes_{var}.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    robot, gait = load()
    g = robot["geometry"]
    L = g["thigh"] + g["shank"] + g["ankle_height"]
    lines = []
    w = lines.append
    w("# Phase 1 — Sizing feasibility report\n")
    w(f"Generated by `cad/analysis/sizing_report.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
      "Do not edit by hand.\n")
    w("Method: quasi-static analysis of the 4-s slow gait in `config/gait.yaml` (both halves, 60 samples per "
      "half). Each sample is an equilibrium pose: feet roll without slipping between the flat / switch / toe "
      "states, the pelvis height is constant, the pelvis fore-aft keyframes are optimised to maximise the "
      "worst static margin. Margins are measured along the Y line (fore-aft) and X line (lateral) through the "
      "CoM projection, as the spec states; the Euclidean distance to the nearest polygon edge is listed for "
      "information. Torques: static gravity + ground reaction, with the sliding foot carrying a forward "
      f"friction force μ = {robot['friction']['slide_design_value']} (worst case of the 0.05–0.15 target) "
      "reacted by the support foot. Because two feet on the ground make the load split statically "
      "indeterminate, two numbers are given: **best** = the split minimising the largest torque ratio "
      "(what a good controller can reach), **worst** = the largest |τ| over every admissible split (an upper "
      "bound, independent of the controller). The pass criterion uses the worst case.\n")

    # ---------------------------------------------------------------- selected design
    sel = {}
    for var in ("wheel", "ptfe"):
        sel[var] = evaluate({}, var)
    H_used = {v: sel[v]["H"] for v in sel}
    w("## 1. Selected design (values now in config/)\n")
    w("| Parameter | Value |\n|---|---|")
    ft = g["foot"]
    rows = [("Leg length (hip axis → sole)", f"{L:.0f} mm"), ("Thigh / shank", f"{g['thigh']:.0f} / {g['shank']:.0f} mm"),
            ("Ankle height", f"{g['ankle_height']:.0f} mm"), ("Stance width (hip spacing)", f"{g['hip_spacing']:.0f} mm"),
            ("Foot toe / heel length, width", f"{ft['toe_length']:.0f} / {ft['heel_length']:.0f} mm, {ft['width']:.0f} mm"),
            ("Toe pad length, wedge face angle", f"{ft['toe_pad']['length']:.0f} mm, {ft['toe_pad']['face_angle']:.0f}°"),
            ("Switch-over angle (design)", f"{ft['toe_pad']['switch_angle']:.0f}°"),
            ("Wheel Ø, axle y, track", f"{ft['wheel']['diameter']:.0f} mm, {ft['wheel']['axle_y']:.0f} mm, {ft['width'] - ft['wheel']['width']:.0f} mm"),
            ("PTFE sole front edge → pad gap", f"{ft['ptfe']['pad_gap']:.0f} mm"),
            ("Step length (per cycle)", f"{gait['step_length']:.0f} mm"),
            ("Cycle time", f"{gait['cycle_time']:.1f} s")]
    for a, b in rows:
        w(f"| {a} | {b} |")
    w("")
    w("| Metric (both halves of the cycle) | wheel | ptfe | Criterion |\n|---|---|---|---|")
    def row(name, f, crit):
        w(f"| {name} | {f(sel['wheel'])} | {f(sel['ptfe'])} | {crit} |")
    row("Hip height H (derived: back-of-slide knee = 27°)", lambda r: f"{r['H']:.1f} mm ({100 * r['H'] / L:.1f} % of leg)", "90–95 % suggested")
    row("Knee of sliding leg at back of slide", lambda r: f"{r['knee_back']:.1f}°", "15–30°")
    row("Reachable, all joints in range, knee ≥ 5° throughout", lambda r: "yes" if r["feasible"] else f"NO: {r['why']}", "yes")
    row("Min fore-aft margin", lambda r: f"{r['fa']:.1f} mm", "≥ 10 mm")
    row("Min lateral margin", lambda r: f"{r['lat']:.1f} mm", "≥ 10 mm")
    row("Min Euclidean margin (info)", lambda r: f"{r['euclid']:.1f} mm", "—")
    for j, nm in enumerate(("hip", "knee", "ankle")):
        row(f"Static τ p99 {nm}: worst / best", lambda r, j=j: f"{r['tau_worst_p99'][j]:.3f} / {r['tau_best_p99'][j]:.3f} N·m ({100 * r['tau_worst_p99'][j] / RATED:.0f} %)", "worst ≤ 0.216")
    row("Joint angle range used: hip", lambda r: f"{r['q_min'][0]:.1f} … {r['q_max'][0]:.1f}°", "−30 … 90")
    row("knee", lambda r: f"{r['q_min'][1]:.1f} … {r['q_max'][1]:.1f}°", "0 … 120")
    row("ankle", lambda r: f"{r['q_min'][2]:.1f} … {r['q_max'][2]:.1f}°", "−35 … 30")
    row("Total mass (estimate)", lambda r: f"{r['mass']:.0f} g", "≤ 700 g")
    row("Mean CoM height", lambda r: f"{r['com_z']:.1f} mm", "phase-2 check ±10 %")
    row("**Pass**", lambda r: "**yes**" if r["ok"] else "**NO**", "")
    w("")
    for var in ("wheel", "ptfe"):
        r = sel[var]
        names = [k["name"] for k in gait["half_cycle"][var][:-1]]
        cfg = [k["pelvis_y"] for k in gait["half_cycle"][var][:-1]]
        w(f"Pelvis keyframes ({var}), y relative to the support foot's flat position: " +
          ", ".join(f"{n} {p:+.1f}" for n, p in zip(names, r["pelvis"])) + " mm.")
        if max(abs(a - b) for a, b in zip(cfg, r["pelvis"])) > 0.5 or abs(gait["hip_height"][var] - r["H"]) > 0.5:
            w(f"**gait.yaml differs from these derived values** (hip_height {gait['hip_height'][var]}, pelvis_y {cfg}). "
              "Copy the values above into gait.yaml.")
        w("")
    w(f"![cycle wheel](analysis/cycle_wheel.png)\n![cycle ptfe](analysis/cycle_ptfe.png)\n")
    w(f"![keyframes wheel](analysis/keyframes_wheel.png)\n![polygons wheel](analysis/polygons_wheel.png)\n")
    w(f"![keyframes ptfe](analysis/keyframes_ptfe.png)\n![polygons ptfe](analysis/polygons_ptfe.png)\n")

    # ---------------------------------------------------------------- foot geometry
    w("## 2. Foot geometry and switch-over angle\n")
    w("The toe pad is a TPU **wedge**: its face rises toward the toe at the face angle, so in the toe-raised "
      "support state the whole pad face is flat on the ground (a patch, not a line). Its rear edge is recessed "
      "above the ground so that, as the foot plantarflexes about its first pivot (wheel axle, or PTFE sole "
      "front edge), the rear edge touches exactly at the switch-over angle. Recess = f(pivot, switch angle); "
      "it is derived, not typed.\n")
    w("| | wheel | ptfe |\n|---|---|---|")
    fgs = {v: foot_geom(robot, v) for v in ("wheel", "ptfe")}
    w("| First pivot (y, z) in foot frame | " + " | ".join(f"({fgs[v].pivot1[0]:.1f}, {fgs[v].pivot1[1]:.1f})" for v in fgs) + " |")
    w("| Pad rear-edge recess above ground (flat) | " + " | ".join(f"{fgs[v].pad_rear_recess:.2f} mm" for v in fgs) + " |")
    w("| Pad tip height above ground (flat) | " + " | ".join(f"{fgs[v].pad_tip[1] + fgs[v].h:.2f} mm" for v in fgs) + " |")
    rows = []
    for th in (0, 5, 8, 10, 12, 15, 18, 22):
        cells = []
        for v, fg in fgs.items():
            an, p = foot_pose(fg, 0.0, th * D2R)
            cps, _ = contacts(fg, 0.0, an, p)
            kinds = sorted(set(k for _x, _y, k in cps))
            extra = ""
            if v == "wheel":
                c = an + np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]]) @ np.array([fg.wheel_y, -fg.h + fg.wheel_r])
                extra = f", wheel bottom {c[1] - fg.wheel_r:.1f} mm"
            cells.append("+".join(kinds) + extra)
        rows.append(f"| θ = {th}° plantarflexion: touching | " + " | ".join(cells) + " |")
    lines.extend(rows)
    w("")
    sw = ft["toe_pad"]["switch_angle"]
    w(f"Switch-over at {sw:.0f}° (criterion 8–12°): flat → only wheels / sole touch; at {sw:.0f}° the pad rear "
      f"edge joins; past it the wheels lift; at {ft['toe_pad']['face_angle']:.0f}° the whole pad face is down.\n")
    w("Wheel parts (starting choice, fits the 10 mm wheel): 3 mm steel axle, one-way needle clutch HF0306 "
      "(3 × 6.5 × 6 mm) pressed into the foot body on the axle, MR63ZZ (3 × 6 × 2.5) on the other side; the two "
      "printed rims are fixed to the axle so both lock together. Estimated set mass ≈ 4 g per foot (< 6 g). "
      "Interference of the wheels with the ankle linkage is checked in phase 2 (`cad/checks.py`); in phase 1 the "
      "ankle lever (r = 12 mm about the ankle axis) sweeps z ≥ −7 mm in the foot frame, the wheel top is at "
      f"z = {-g['ankle_height'] + ft['wheel']['diameter']:.0f} mm: clear by ≥ "
      f"{-7 - (-g['ankle_height'] + ft['wheel']['diameter']):.0f} mm.\n")

    # ---------------------------------------------------------------- sweeps
    w("## 3. Sweeps\n")
    w("Every row re-derives the hip height (knee 27° at the back of the slide unless the row varies it) and "
      "re-optimises the pelvis keyframes. `fa`/`lat` = minimum fore-aft / lateral margin [mm]; τ = worst-case "
      "static p99 [N·m] hip/knee/ankle. ✔ = all criteria met.\n")

    def table(title, key, vals, var="wheel", label=None, extra=None):
        jobs = []
        for v in vals:
            ov = dict(extra or {})
            kw = {}
            if key == "knee":
                kw["knee_target"] = v
            elif key == "legscale":
                for p in ("geometry.thigh", "geometry.shank", "geometry.ankle_height"):
                    ov["robot." + p] = robot["geometry"][p.split(".")[1]] * v
            elif key == "thigh_frac":
                tot = g["thigh"] + g["shank"]
                ov["robot.geometry.thigh"], ov["robot.geometry.shank"] = tot * v, tot * (1 - v)
            else:
                ov[key] = v
            jobs.append(((ov, var), kw))
        res = run_pool(jobs)
        w(f"### {title} ({var})\n")
        w(f"| {label or key} | H [mm] | knee@back | fa | lat | τ hip / knee / ankle | feasible | ✔ |\n|---|---|---|---|---|---|---|---|")
        for v, r in zip(vals, res):
            if "H" not in r:
                w(f"| {v} | — | — | — | — | — | {r.get('why')} | ✘ |")
                continue
            tau = " / ".join(f"{x:.3f}" for x in r["tau_worst_p99"]) if "tau_worst_p99" in r else "—"
            w(f"| {v if not isinstance(v, float) else round(v, 3)} | {r['H']:.0f} | {r['knee_back']:.1f} | {r['fa']:.1f} | "
              f"{r['lat']:.1f} | {tau} | {'yes' if r['feasible'] else r['why']} | {'✔' if r['ok'] else '✘'} |")
        w("")
        return res

    table("Leg length (thigh, shank, ankle height scaled together)", "legscale", [0.85, 0.9, 1.0, 1.1, 1.2],
          label="scale (1.0 = 180 mm)")
    table("Thigh share of thigh + shank", "thigh_frac", [0.48, 0.5, 0.526, 0.55, 0.58], label="thigh fraction")
    table("Ankle height", "robot.geometry.ankle_height", [22.0, 25.0, 28.0, 32.0, 36.0], label="ankle height [mm]")
    table("Nominal hip height (via the back-of-slide knee angle)", "knee", [15.0, 20.0, 27.0, 30.0, 40.0], label="knee@back target [°]")
    for var in ("wheel", "ptfe"):
        table("Step length", "gait.step_length", [20.0, 30.0, 40.0, 50.0, 60.0], var=var, label="step [mm]")
        table("Foot width", "robot.geometry.foot.width", [32.0, 36.0, 40.0, 44.0, 48.0], var=var, label="width [mm]")
        table("Stance width", "robot.geometry.hip_spacing", [48.0, 52.0, 56.0, 62.0, 68.0], var=var, label="hip spacing [mm]")
        table("Toe length", "robot.geometry.foot.toe_length", [36.0, 39.0, 42.0, 46.0, 50.0], var=var, label="toe [mm]")
    table("Heel length", "robot.geometry.foot.heel_length", [23.0, 26.0, 30.0], label="heel [mm]")
    table("Wheel axle position", "robot.geometry.foot.wheel.axle_y", [-10.0, -14.0, -18.0, -20.0], label="axle y [mm]")
    table("Wheel diameter", "robot.geometry.foot.wheel.diameter", [8.0, 10.0, 12.0], label="Ø [mm]")
    table("Switch-over angle", "robot.geometry.foot.toe_pad.switch_angle", [8.0, 10.0, 12.0], label="switch [°]")
    table("PTFE pad gap", "robot.geometry.foot.ptfe.pad_gap", [3.0, 6.0, 10.0], var="ptfe", label="gap [mm]")

    if not args.quick:
        w("### Factorial: step length × foot width × stance width × toe length (wheel, stability only)\n")
        grid = list(itertools.product([30.0, 40.0, 50.0], [36.0, 40.0, 44.0], [52.0, 56.0, 62.0], [38.0, 42.0, 46.0]))
        jobs = [(({"gait.step_length": a, "robot.geometry.foot.width": b, "robot.geometry.hip_spacing": c,
                   "robot.geometry.foot.toe_length": d}, "wheel"), {"torque": False}) for a, b, c, d in grid]
        res = run_pool(jobs)
        okc = [(p, r) for p, r in zip(grid, res) if "H" in r and r["feasible"]]
        okc.sort(key=lambda pr: -min(pr[1]["fa"], pr[1]["lat"]))
        n_pass = sum(1 for _p, r in okc if min(r["fa"], r["lat"]) >= MARGIN_MIN)
        w(f"{len(grid)} combinations, {n_pass} reach ≥ 10 mm in both directions. Best 12:\n")
        w("| step | foot width | stance | toe | fa | lat | euclid |\n|---|---|---|---|---|---|---|")
        for p, r in okc[:12]:
            w(f"| {p[0]:.0f} | {p[1]:.0f} | {p[2]:.0f} | {p[3]:.0f} | {r['fa']:.1f} | {r['lat']:.1f} | {r['euclid']:.1f} |")
        w("")

    # ---------------------------------------------------------------- electronics / battery
    w("## 4. Electronics uncertainty and the battery placement region\n")
    w("The gait (pelvis keyframes and hip height) is held fixed at the selected design's values; the electronics "
      "mass is scaled by 0.5 / 1.0 / 1.5 and the battery centre is moved along Y in the pelvis frame. Battery "
      "height does not enter the static margin (the CoM projection ignores z) and is limited only by the pelvis "
      "geometry; it matters for dynamics, so it is randomised in phase 3.\n")
    dys = np.arange(-120.0, 120.01, 5.0)
    region = {}
    for var in ("wheel", "ptfe"):
        # keep the pelvis keyframes and hip height of the nominal design fixed (no re-optimisation)
        res = []
        r0, g0 = load()
        for f in (0.5, 1.0, 1.5):
            row_ = []
            for dy in dys:
                gm = GaitModel(r0, g0, var, f, dy)
                gm.H = H_used[var]
                ss = gm.sample(np.array(sel[var]["pelvis"]), 60)
                mm = min(min(s.margin["fore_aft"], s.margin["lateral"]) if s.margin else -1e3 for s in ss)
                row_.append(mm)
            res.append(row_)
        res = np.array(res)
        region[var] = res
        w(f"**{var}** — worst margin [mm] vs battery Δy (columns) and electronics mass factor (rows):\n")
        w("| factor \\ Δy | " + " | ".join(f"{d:+.0f}" for d in dys[::4]) + " |")
        w("|---" * (len(dys[::4]) + 1) + "|")
        for f, rr in zip((0.5, 1.0, 1.5), res):
            w(f"| {f} | " + " | ".join(("**" if x >= MARGIN_MIN else "") + f"{x:.1f}" + ("**" if x >= MARGIN_MIN else "") for x in rr[::4]) + " |")
        ok_all = np.all(res >= MARGIN_MIN, axis=0)
        if ok_all.any():
            lo, hi = dys[ok_all].min(), dys[ok_all].max()
            w(f"\nAllowed battery centre (all three mass factors, margin ≥ 10 mm): **Δy ∈ [{lo:+.1f}, {hi:+.1f}] mm** "
              f"about the nominal position y = {robot['electronics']['battery']['position'][1]:+.1f} mm.")
            rail = robot["electronics"]["battery_rail"]["travel"]
            w(f" Rail travel in robot.yaml: [{rail[0]:+.0f}, {rail[1]:+.0f}] mm → "
              + ("**rail inside the region**" if lo <= rail[0] and hi >= rail[1] else "**rail exceeds the region — shorten it**") + ".\n")
        else:
            w("\n**No battery position keeps ≥ 10 mm for all mass factors.**\n")
        region[var + "_range"] = (float(dys[ok_all].min()), float(dys[ok_all].max())) if ok_all.any() else None
        PH1.setdefault(var, {})["battery_region"] = region[var + "_range"]

    # ---------------------------------------------------------------- standing
    w("## 5. Standing pose (reset pose for train/)\n")
    w("Feet side by side at y = 0, knees pre-bent to 25° (spec: 15–30°), pelvis y chosen so the CoM is centred "
      "fore-aft in the support polygon. Joint angles are what train/ resets to.\n")
    from statics import LegState, com as _com, convex_hull, link_points, margins as _m
    for var in ("wheel", "ptfe"):
        r0, g0 = load()
        gm = GaitModel(r0, g0, var)
        st = g0["standing"]
        th = gm.theta[st["foot_state"][var]]
        an, p = foot_pose(gm.fg, 0.0, th)
        L1, L2 = gm.leg.L1, gm.leg.L2
        d = np.sqrt(L1**2 + L2**2 + 2 * L1 * L2 * np.cos(STAND_KNEE * D2R))
        py = 0.0
        for _ in range(6):
            Hs = an[1] + np.sqrt(d * d - (an[0] - py) ** 2)
            qh, qk = gm.leg.ik(an[0] - py, an[1] - Hs)
            qa = p - (qh - qk)
            ls = [LegState(x, np.array([py, Hs]), qh, qk, qa, an, p, contacts(gm.fg, x, an, p)[0])
                  for x in (-gm.w / 2, gm.w / 2)]
            c = _com(link_points(gm.mm, ls, L1, np.array([py, Hs])))
            hull = convex_hull(np.array([[q[0], q[1]] for lg in ls for q in lg.contacts]))
            ext = _ray_extent(hull, c[:2], 1)
            py += 0.5 * (ext[0] + ext[1]) - c[1]
        mg = _m(hull, c[:2])
        PH1.setdefault(var, {}).update(
            mass=gm.mm.total, standing_H=float(Hs), standing_pelvis_y=float(py), standing_com_z=float(c[2]),
            standing_q={"hip_pitch": float(qh / D2R), "knee_pitch": float(qk / D2R), "ankle_pitch": float(qa / D2R)},
            standing_margin={k: float(v) for k, v in mg.items()})
        cfg_H, cfg_py = st["hip_height"][var], st["pelvis_y"][var]
        w(f"- **{var}** ({st['foot_state'][var]} feet): H = {Hs:.1f} mm, pelvis y = {py:+.1f} mm → hip {qh / D2R:.1f}°, "
          f"knee {qk / D2R:.1f}°, ankle {qa / D2R:.1f}°; margins fore-aft {mg['fore_aft']:.1f}, lateral {mg['lateral']:.1f} mm."
          + ("" if abs(cfg_H - Hs) < 0.5 and abs(cfg_py - py) < 0.5 else
             f" **gait.yaml standing differs (H {cfg_H}, pelvis_y {cfg_py}) — copy these values.**"))
    w("")

    # ---------------------------------------------------------------- verdict
    w("## 6. Pass criteria\n")
    def yes(b):
        return "✔" if b else "✘"
    for var in ("wheel", "ptfe"):
        r = sel[var]
        w(f"**{var}**")
        w(f"- {yes(max(r['tau_worst_p99']) <= RATED)} Static torque p99 ≤ 0.216 N·m (worst-case split): "
          f"max {max(r['tau_worst_p99']):.3f} N·m")
        w(f"- {yes(min(r['fa'], r['lat']) >= MARGIN_MIN)} Slow-gait static margin ≥ 10 mm: fore-aft {r['fa']:.1f}, lateral {r['lat']:.1f} mm")
        w(f"- {yes(r['feasible'])} Sliding foot reaches the floor throughout (IK feasible, knee ≥ 5°, joints in range)")
        w(f"- {yes(KNEE_BACK[0] <= r['knee_back'] <= KNEE_BACK[1])} Knee at back of slide 15–30°: {r['knee_back']:.1f}°")
        w(f"- {yes(8 <= sw <= 12)} Switch-over angle 8–12°: {sw:.0f}° (by construction; verified geometrically in §2)")
        rg = region.get(var + "_range")
        w(f"- {yes(rg is not None)} Battery placement region non-empty: " + (f"Δy ∈ [{rg[0]:+.1f}, {rg[1]:+.1f}] mm" if rg else "empty"))
        w("")
    w(f"Runtime {time.time() - t0:.0f} s.\n")
    figures(robot, gait, "wheel", sel["wheel"]["pelvis"], H_used["wheel"])
    figures(robot, gait, "ptfe", sel["ptfe"]["pelvis"], H_used["ptfe"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sizing_report.md").write_text("\n".join(lines))
    for var in sel:
        PH1.setdefault(var, {}).update(H=sel[var]["H"], fa=sel[var]["fa"], lat=sel[var]["lat"],
                                       com_z_gait=sel[var]["com_z"], tau_worst_p99=sel[var].get("tau_worst_p99"))
    (FIG / "phase1.json").write_text(json.dumps(PH1, indent=1))
    print("\n".join(lines[:60]))
    print(f"... written {OUT / 'sizing_report.md'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
