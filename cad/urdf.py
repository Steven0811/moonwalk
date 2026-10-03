"""URDF generation. THE ONLY PLACE where model units (mm, g, g·mm²) are converted to SI (m, kg, kg·m²).

* inertials: computed from the solids (mass.py), in the link frame, CoM origin, full tensor
* collisions: primitives only (box / sphere), defined here from robot.yaml geometry, named per
  robot.yaml collision.names — train/ binds physics materials by these names
* visuals: one coarse STL per link (display only)
* limits: robot.yaml joint ranges, stall torque, no-load speed
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np

from assembly import joints
from params import P, foot_geom, servo_bands

MM = 1e-3          # mm -> m
G = 1e-3           # g -> kg
GMM2 = 1e-9        # g·mm² -> kg·m²


def si_length(mm):
    return np.asarray(mm, float) * MM


def si_mass(g):
    return g * G


def si_inertia(gmm2):
    return np.asarray(gmm2, float) * GMM2


@dataclass
class Prim:
    name: str
    kind: str            # box | sphere
    center: tuple        # mm, link frame
    size: tuple          # box: (sx, sy, sz) mm; sphere: (r,)
    rpy: tuple = (0.0, 0.0, 0.0)   # rad
    zone: str = "body"   # toe_pad | sole | heel | wheel | body  (physics material binding)


def _box(name, x0, x1, y0, y1, z0, z1, zone="body"):
    return Prim(name, "box", ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), zone=zone)


def collisions(kind: str, side: str | None, variant: str) -> list[Prim]:
    """Collision primitives of a link (left leg frame; mirrored for the right)."""
    N = P.collision.names
    s = P.collision.shrink
    S = P.structure
    g = P.geometry
    G_ = P.servo.geometry
    a1 = servo_bands()["arm"][1]
    sd = side or ""
    out = []
    if kind == "base_link":
        xs = g.hip_spacing / 2 + a1 - s
        H = S.hip_module
        out.append(_box("base_link_body", -xs, xs, -G_.case[0] / 2, G_.case[0] / 2, H.bottom_z + s, S.pelvis.plate_z[0]))
        t = S.tray
        out.append(_box("base_link_tray", -t.size_xy[0] / 2 + s, t.size_xy[0] / 2 - s, S.pelvis.plate_y[0] + s,
                        S.pelvis.plate_y[1] - s, S.pelvis.plate_z[0], t.battery_rail.rail_z[1]))
        b = P.electronics.battery
        (bx, by, bz), (sx, sy, sz) = b["position"], b["size"]
        out.append(_box("base_link_battery", bx - sx / 2, bx + sx / 2, by - sy / 2, by + sy / 2, bz - sz / 2, bz + sz / 2))
        return out
    nm = N.body.format(side=sd, link=kind)
    if kind == "thigh":
        T = S.thigh
        out.append(_box(nm, -a1 + s, a1 - s, -T.lower_half_width + s, T.lower_half_width - s,
                        -g.thigh + G_.case_center[1] - G_.case[1] / 2 + s, -T.top_gap - s))
    elif kind == "shank":
        sh = S.shank
        yw = G_.case[0] / 2 + P["print"].servo_pocket_clearance + S.pocket_wall
        bot = sh.ankle_servo_shaft_z - G_.case_center[1] - G_.case[1] / 2 - S.pocket_wall
        out.append(_box(nm, -S.side_plate[1] + s, S.side_plate[1] - s, -yw + s, yw - s, bot + s, sh.body_top - s))
    elif kind == "foot":
        fg = foot_geom(variant)
        hw = g.foot.width / 2
        pad = g.foot.toe_pad
        # toe pad: box aligned with the wedge face, its bottom face on the pad face
        a = fg.face
        L = pad.length / math.cos(a)
        t = pad.thickness
        r, tip = fg.pad_rear, fg.pad_tip
        mid = ((r[0] + tip[0]) / 2 - math.sin(a) * t / 2, (r[1] + tip[1]) / 2 + math.cos(a) * t / 2)
        out.append(Prim(N.toe_pad.format(side=sd), "box", (0.0, mid[0], mid[1]), (2 * hw, L, t), (a, 0.0, 0.0),
                        zone="toe_pad"))
        if variant == "wheel":
            w = g.foot.wheel
            hb = S.foot.heel_block
            z0 = -fg.h + w.heel_clearance
            out.append(_box(N.heel.format(side=sd), -hb.half_width, hb.half_width, hb.y[0], hb.y[1], z0, z0 + 3.0,
                            zone="heel"))
        else:
            hs = g.foot.ptfe.heel_split
            out.append(_box(N.heel.format(side=sd), -hw, hw, -fg.heel, -fg.heel + hs, -fg.h, -fg.h + 3.0, zone="heel"))
            out.append(_box(N.sole.format(side=sd), -hw, hw, -fg.heel + hs, fg.sole_front, -fg.h, -fg.h + 3.0,
                            zone="sole"))
        out.append(_box(nm, -hw + s, hw - s, -fg.heel + 2 * s, r[0] - s, S.foot.stop_top - 9.0, S.foot.stop_top))
    elif kind == "wheel":
        w = g.foot.wheel
        hw = g.foot.width / 2
        for sgn, lr in ((-1, "out"), (1, "in")):
            out.append(Prim(N.wheel.format(side=sd, lr=lr), "sphere", (sgn * (hw - w.width / 2), 0.0, 0.0),
                            (w.diameter / 2,), zone="wheel"))
    if side == "R":
        out = [Prim(p.name, p.kind, (-p.center[0], p.center[1], p.center[2]), p.size, p.rpy, p.zone) for p in out]
    return out


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def write_urdf(path, variant: str, link_list, props: dict, mesh_names: dict, mass_total: float) -> str:
    """props: {link: MassProps (g, mm, g·mm²)}; mesh_names: {link: relative STL path (mm units)}."""
    lines = ['<?xml version="1.0"?>',
             f"<!-- Generated by cad/build.py — do not edit. Foot variant: {variant}. "
             f"Total mass {mass_total:.1f} g. Units: m, kg, kg·m², rad. -->",
             f'<robot name="moonwalk_mini_{variant}">']
    for name, side, kind in link_list:
        mp = props[name]
        com = si_length(mp.com)
        I = si_inertia(mp.inertia)
        lines.append(f'  <link name="{name}">')
        lines.append("    <inertial>")
        lines.append(f'      <origin xyz="{_f(com)}" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{si_mass(mp.mass):.6g}"/>')
        lines.append(f'      <inertia ixx="{I[0, 0]:.6g}" ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" '
                     f'iyy="{I[1, 1]:.6g}" iyz="{I[1, 2]:.6g}" izz="{I[2, 2]:.6g}"/>')
        lines.append("    </inertial>")
        if name in mesh_names:
            lines.append("    <visual>")
            lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            lines.append(f'      <geometry><mesh filename={quoteattr(mesh_names[name])} scale="{MM} {MM} {MM}"/></geometry>')
            lines.append("    </visual>")
        for p in collisions(kind, side, variant):
            lines.append(f'    <collision name="{p.name}">')
            lines.append(f'      <origin xyz="{_f(si_length(p.center))}" rpy="{_f(p.rpy)}"/>')
            if p.kind == "box":
                lines.append(f'      <geometry><box size="{_f(si_length(p.size))}"/></geometry>')
            else:
                lines.append(f'      <geometry><sphere radius="{si_length(p.size[0]):.6g}"/></geometry>')
            lines.append("    </collision>")
        lines.append("  </link>")
    sv = P.servo
    for j in joints(variant):
        lines.append(f'  <joint name="{j.name}" type="{j.kind}">')
        lines.append(f'    <parent link="{j.parent}"/>')
        lines.append(f'    <child link="{j.child}"/>')
        lines.append(f'    <origin xyz="{_f(si_length(j.origin))}" rpy="0 0 0"/>')
        lines.append(f'    <axis xyz="{_f(j.axis)}"/>')
        if j.kind == "revolute":
            lines.append(f'    <limit lower="{math.radians(j.lower):.6g}" upper="{math.radians(j.upper):.6g}" '
                         f'effort="{sv.stall_torque}" velocity="{sv.no_load_speed}"/>')
        else:
            fr = P.friction
            lines.append('    <limit effort="0" velocity="200"/>')
            lines.append(f'    <dynamics damping="{fr.wheel_axle_damping}" friction="{fr.wheel_axle_coulomb}"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    txt = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    return txt
