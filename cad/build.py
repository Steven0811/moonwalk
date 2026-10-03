"""Produce every cad/ output in one run:

    conda run -n cad python cad/build.py [--fast]

  cad/out/stl/        printed parts in their print orientation (left and right versions)
  cad/out/step/       assembly (zero pose, vendor servo models) per foot variant + every printed part
  cad/out/urdf/       URDF per foot variant (+ moonwalk_mini.urdf = robot.yaml foot_variant) and link meshes
  cad/out/views/      front / side / top / iso views per variant, posed moonwalk keyframes
  cad/out/model_report.md

The checks run first; if any fails, no URDF is written (the other outputs are still produced for review).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from build123d import Compound, Rot, export_step, export_stl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checks                                                  # noqa: E402
from assembly import fk, joints, to_location                   # noqa: E402
from links import link_bodies, links                           # noqa: E402
from mass import box_inertia, body_props, link_props           # noqa: E402
from params import OUT, P, ROOT, load_gait, servo_bands       # noqa: E402
from parts import servo                                        # noqa: E402
from parts.tolerance import tolerance_piece                    # noqa: E402
from render import draw, save_views, scene                    # noqa: E402
from urdf import collisions, write_urdf                        # noqa: E402

VARIANTS = ("wheel", "ptfe")


def printed_parts():
    """{file stem: Body} for every printed part (left/right where they differ)."""
    out = {}
    for v in VARIANTS:
        for name, side, kind in links(v):
            for b in link_bodies(kind, side, v):
                if not b.printed:
                    continue
                stem = b.name
                if kind == "foot" and b.name == "foot":
                    stem = f"foot_{v}"
                if side:
                    stem = f"{stem}_{side}"
                if b.name in ("wheel_in", "wheel_out"):
                    stem = "wheel"                                   # four identical wheels
                if b.name == "toe_pad":
                    stem = f"toe_pad_{side}"
                out.setdefault(stem, (b, v, side))
    for b in tolerance_piece():
        out[b.name] = (b, None, None)
    return out


def export_prints(parts):
    d = OUT / "stl"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, (b, v, side) in sorted(parts.items()):
        s = Rot(*b.print_rot) * b.shape
        bb = s.bounding_box()
        from build123d import Pos
        s = Pos(-bb.center().X, -bb.center().Y, -bb.min.Z) * s
        export_stl(s, str(d / f"{stem}.stl"), tolerance=0.02, angular_tolerance=0.1)
        m = body_props(b).mass
        rows.append((stem, b.material, m, s.bounding_box().size, b.note))
    return rows


def export_steps(parts):
    d = OUT / "step"
    d.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        T = fk({}, v)
        objs = []
        for name, side, kind in links(v):
            loc = to_location(T[name])
            for b in link_bodies(kind, side, v):
                if not b.collide and b.visual is None:
                    continue
                s = (b.visual if b.visual is not None else b.shape).moved(loc)
                s.label = f"{name}/{b.name}"
                objs.append(s)
        c = Compound(children=objs, label=f"moonwalk_mini_{v}")
        export_step(c, str(d / f"assembly_{v}.step"))
    for stem, (b, v, side) in parts.items():
        export_step(b.shape, str(d / f"part_{stem}.step"))


def export_urdf(props_by_variant):
    d = OUT / "urdf"
    md = d / "meshes"
    if d.exists():
        shutil.rmtree(d)
    md.mkdir(parents=True)
    for v in VARIANTS:
        meshes = {}
        for name, side, kind in links(v):
            shapes = [(b.visual if b.visual is not None else b.shape) for b in link_bodies(kind, side, v)
                      if b.collide or b.visual is not None]
            stem = f"{name}_{v}" if kind in ("foot",) else name
            p = md / f"{stem}.stl"
            if not p.exists():
                export_stl(Compound(shapes), str(p), tolerance=0.3, angular_tolerance=0.5)
            meshes[name] = f"meshes/{stem}.stl"
        props = props_by_variant[v]
        tot = sum(pp.mass for pp in props.values())
        write_urdf(d / f"moonwalk_mini_{v}.urdf", v, links(v), props, meshes, tot)
    shutil.copy(d / f"moonwalk_mini_{P.foot_variant}.urdf", d / "moonwalk_mini.urdf")


def gait_keyframe_views(variant):
    """Side views of the robot in the phase-1 moonwalk keyframes (both halves)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "cad" / "analysis"))
    from gait import GaitModel
    from model import load
    r, g = load()
    gm = GaitModel(r, g, variant)
    pel = gm.default_pelvis()
    kfs = gm.kf
    fig, axs = plt.subplots(2, len(kfs) - 1, figsize=(3.3 * (len(kfs) - 1), 8.2))
    for half in (0, 1):
        for i, k in enumerate(kfs[:-1]):
            s = gm.pose(k["t"], pel, half)
            q = {}
            for lg in s.legs:
                sd = "L" if lg.x < 0 else "R"
                q[f"{sd}_hip_pitch"] = np.degrees(lg.qh)
                q[f"{sd}_knee_pitch"] = np.degrees(lg.qk)
                q[f"{sd}_ankle_pitch"] = np.degrees(lg.qa)
            base = np.eye(4)
            base[:3, 3] = (0.0, s.pelvis[0], s.pelvis[1])
            ax = axs[half, i]
            draw(ax, scene(variant, q, base, tol=0.3), "side")
            ax.plot([s.com[1]], [s.com[2]], "r+", ms=12, mew=2)
            ax.set_xlim(-75, 95)
            ax.set_ylim(-3, 255)
            sup = "left" if half == 0 else "right"
            ax.set_title(f"{k['name']} (support: {sup})\nfa {s.margin['fore_aft']:.1f} / lat {s.margin['lateral']:.1f} mm",
                         fontsize=8)
    fig.suptitle(f"Moonwalk keyframes, side view from the robot's right (forward →), {variant} foot; "
                 "red + = CoM", fontsize=10)
    fig.tight_layout()
    p = OUT / "views" / f"gait_keyframes_{variant}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def report(results, props_by_variant, prints, t0):
    L = []
    w = L.append
    v0 = P.foot_variant
    w("# Phase 2 — Model report\n")
    w(f"Generated by `cad/build.py` on {time.strftime('%Y-%m-%d %H:%M')}. Do not edit by hand.\n")
    w(f"Default foot variant (robot.yaml `foot_variant`): **{v0}**. Hip roll: "
      f"**{'on' if P.hip_roll.enabled else 'off'}**.\n")
    # checks
    w("## Check results (`cad/checks.py`)\n")
    for v, res in results.items():
        w(f"**{v} foot**\n")
        w("| Check | Result | Details |\n|---|---|---|")
        for r in res:
            tag = "PENDING" if r.pending else ("PASS" if r.ok else "**FAIL**")
            w(f"| {r.name} | {tag} | {r.detail} |")
        w("")
    # masses
    w("## Mass, centre of mass, inertia per link\n")
    w("Link frame = origin on the joint axis to the parent, zero-pose orientation (Z up, +Y forward). "
      "Inertia about the CoM. The box check compares the trace of the inertia with that of a solid box of the "
      "link's bounding-box size and the same mass (an order-of-magnitude difference would mean a unit error).\n")
    for v in VARIANTS:
        props = props_by_variant[v]
        tot = sum(p.mass for p in props.values())
        w(f"### {v} foot — total **{tot:.1f} g**\n")
        w("| Link | Mass [g] | CoM [mm] | Ixx, Iyy, Izz [g·mm²] | Ixy, Ixz, Iyz | trace / box trace |\n|---|---|---|---|---|---|")
        for name, side, kind in links(v):
            p = props[name]
            bbs = Compound([b.shape for b in link_bodies(kind, side, v) if b.collide]).bounding_box().size
            ratio = np.trace(p.inertia) / np.trace(box_inertia(p.mass, (bbs.X, bbs.Y, bbs.Z)))
            I = p.inertia
            w(f"| {name} | {p.mass:.2f} | {p.com[0]:+.1f}, {p.com[1]:+.1f}, {p.com[2]:+.1f} | "
              f"{I[0, 0]:.0f}, {I[1, 1]:.0f}, {I[2, 2]:.0f} | {I[0, 1]:.0f}, {I[0, 2]:.0f}, {I[1, 2]:.0f} | {ratio:.2f} |")
        T = fk({}, v)
        c = sum(p.mass * (T[n][:3, :3] @ p.com + T[n][:3, 3]) for n, p in props.items()) / tot
        w(f"\nWhole-robot CoM at the zero pose: ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:.1f}) mm.\n")
    w("### Bodies per link (left leg; the right is the mirror image)\n")
    w("| Link | Body | Material | Mass [g] | Note |\n|---|---|---|---|---|")
    for name, side, kind in links("wheel") + [("L_foot (ptfe)", "L", "foot")]:
        if side == "R":
            continue
        var = "ptfe" if "ptfe" in name else "wheel"
        for b in link_bodies(kind, side, var):
            w(f"| {name} | {b.name} | {b.material} | {body_props(b).mass:.2f} | {b.note} |")
    w("")
    # budget
    props = props_by_variant[v0]
    cats = {"servos": 0.0, "electronics": 0.0, "printed": 0.0, "hardware/wiring": 0.0, "wheel set": 0.0}
    for name, side, kind in links(v0):
        for b in link_bodies(kind, side, v0):
            m = body_props(b).mass
            if b.name.endswith("servo") or b.name.startswith("hip_servo"):
                cats["servos"] += m
            elif b.name.startswith("elec_"):
                cats["electronics"] += m
            elif kind == "wheel" or b.name == "clutch_bearings":
                cats["wheel set"] += m
            elif b.printed:
                cats["printed"] += m
            else:
                cats["hardware/wiring"] += m
    w("### Mass budget (cad/CLAUDE.md) vs model\n")
    w("| Item | Budget | Model |\n|---|---|---|")
    bud = {"servos": "126 g", "electronics": "~55 g (battery 30 + boards 25)", "printed": "~300 g",
           "hardware/wiring": "~80 g", "wheel set": "6–12 g"}
    for k, m in cats.items():
        w(f"| {k} | {bud[k]} | {m:.1f} g |")
    w(f"| **total** | ~570 g, limit 700 g | **{sum(cats.values()):.1f} g** |\n")
    w("Printed masses use the effective densities in robot.yaml (estimates until the slicer / scale gives "
      "real numbers). The printed structure is far lighter than the 300 g budget line: the parts are thin "
      "(2–3.5 mm walls) and the effective density assumes 4 walls + 30 % gyroid.\n")
    # servo
    m = servo.measured_geometry()
    w("## Servo model (vendor STEP)\n")
    w(f"Source `{P.servo.vendor_step}` (see SOURCE.md). Servo frame S: origin on the output-shaft axis at the "
      "output-horn face, +Z out of the output horn, +Y toward the case's long end. Measured in code:\n")
    w("| Quantity (S frame, mm) | Measured |\n|---|---|")
    for k, val in m.items():
        w(f"| {k} | {val} |")
    bb = servo.simplified_vs_step()
    w(f"\nSimplified shape (case box + horn discs + horn-screw heads) vs full STEP bounding box: Δ = {bb} mm "
      "(criterion < 0.5 mm). Validation note: the spec's 34 × 20 × 23 mm bounding-box check applies to the case "
      "body (measured 34.00 × 20.00 × 22.97 mm). The full STEP also contains both horns and their screws, "
      "20 × 34 × 30.2 mm; horn face to horn face is 29.0 mm, as on the datasheet. Units are mm.\n")
    # joints
    w("## Joints (URDF)\n")
    w("| Joint | Parent → child | Origin [mm] | Axis | Range [°] | Effort / velocity |\n|---|---|---|---|---|---|")
    for j in joints(v0):
        rng = f"{j.lower:+.0f} … {j.upper:+.0f}" if j.lower is not None else "continuous (passive)"
        ev = f"{P.servo.stall_torque} N·m / {P.servo.no_load_speed} rad/s" if j.kind == "revolute" else "—"
        w(f"| {j.name} | {j.parent} → {j.child} | {j.origin[0]:+.1f}, {j.origin[1]:+.1f}, {j.origin[2]:+.1f} | "
          f"{tuple(int(a) for a in j.axis)} | {rng} | {ev} |")
    w("\nServo rotation sense (for firmware): every servo's output horn faces outward. For the left leg the "
      "servo +Z_S is −X, so a positive servo rotation (CCW seen from the horn) is a rotation about −X: hip "
      "and ankle angles = −servo angle, knee angle = +servo angle. For the right leg (+Z_S = +X) the signs "
      "flip. Measure this on the bench before trusting it.\n")
    w(f"Ankle drive: parallelogram linkage (crank {P.structure.linkage.crank_length:.0f} mm on the ankle servo, "
      "rod, equal lever on the foot), so ankle angle = servo angle. The URDF models the ankle as a direct "
      "revolute joint; crank and rod mass is lumped into the shank. **Estimated linkage backlash "
      f"{P.structure.linkage.backlash_deg}° (two MR52ZZ pin joints) on top of the servo's 0.5° → train/ servo "
      "model: ankle backlash ≈ 0.7°.**\n")
    # collisions
    w("## Collision primitives (names for train/ physics-material binding)\n")
    w("| Link | Name | Type | Zone | Centre [mm] | Size [mm] | rpy [°] |\n|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        for name, side, kind in links(v):
            if side == "R" or (v == "ptfe" and kind not in ("foot",)):
                continue
            for p in collisions(kind, side, v):
                w(f"| {name} ({v}) | `{p.name}` | {p.kind} | {p.zone} | "
                  f"{', '.join(f'{x:+.1f}' for x in p.center)} | {', '.join(f'{x:.1f}' for x in p.size)} | "
                  f"{', '.join(f'{np.degrees(x):.0f}' for x in p.rpy)} |")
    w("\nNaming scheme (robot.yaml `collision.names`): `{side}_foot_toe_pad`, `{side}_foot_sole`, "
      "`{side}_foot_heel`, `{side}_wheel_in` / `_out` (spheres), `{side}_{link}_body`, `base_link_*`.\n")
    # printing
    w("## Printed parts (`cad/out/stl/`, already in print orientation)\n")
    w("| File | Material | Mass est. [g] | Size in print orientation [mm] | Note |\n|---|---|---|---|---|")
    for stem, mat, mm, sz, note in prints:
        w(f"| {stem}.stl | {mat} | {mm:.1f} | {sz.X:.0f} × {sz.Y:.0f} × {sz.Z:.0f} | {note} |")
    w("\nFirst print **tolerance_holes / tolerance_horn / tolerance_servo_collars** and report which sizes fit; "
      "the values go to robot.yaml `print:`.\n")
    # views
    w("## Views\n")
    for v in VARIANTS:
        for vw in ("iso", "iso_back", "front", "side", "top"):
            w(f"![{vw} {v}](views/{vw}_{v}.png)")
        w(f"\n![gait {v}](views/gait_keyframes_{v}.png)\n")
    w(f"\nBuild time {time.time() - t0:.0f} s.\n")
    (OUT / "model_report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="coarser interference sweep (development only)")
    ap.add_argument("--no-views", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    props = {v: {n: link_props(link_bodies(k, s, v)) for n, s, k in links(v)} for v in VARIANTS}
    results = {}
    for v in VARIANTS:
        print(f"== checks ({v}) ==", flush=True)
        results[v] = checks.run(v, a.fast, props[v])
    all_ok = all(r.ok for res in results.values() for r in res)
    parts = printed_parts()
    print("== STL ==", flush=True)
    prints = export_prints(parts)
    print("== STEP ==", flush=True)
    export_steps(parts)
    if all_ok and not a.fast:
        print("== URDF ==", flush=True)
        export_urdf(props)
    else:
        u = OUT / "urdf"
        if u.exists():
            shutil.rmtree(u)
        print("!! URDF NOT written: " + ("checks failed" if not all_ok else "--fast build"), flush=True)
    if not a.no_views:
        print("== views ==", flush=True)
        for v in VARIANTS:
            save_views(v, OUT / "views")
            gait_keyframe_views(v)
    report(results, props, prints, t0)
    print(f"done in {time.time() - t0:.0f} s; all checks pass: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
