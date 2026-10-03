"""base_link parts: hip modules (one per side), pelvis plate, electronics tray, placeholders, hip servos.

base_link frame: origin midway between the two hip-pitch axes, Z up, +Y forward, +X right.
"""
from __future__ import annotations

from build123d import Pos

from params import P, leg_plane_x, servo_bands
from parts import servo
from parts.common import (Body, box, carve_joint, cyl_x, lumped, rect, servo_frame)

S = P.structure
PR = P["print"]
G = P.servo.geometry


def hip_servo_loc(side="L"):
    xc = leg_plane_x(side)
    b = servo_bands()
    return servo_frame((xc - b["horn_face"], 0.0, 0.0), "up")


def hip_module(side="L") -> Body:
    """Separate printed cradle holding one hip-pitch servo; screwed to the pelvis plate (replaceable when
    hip roll is added). Its walls in the thigh's arm bands are cut by the thigh-arm sweep over the hip range,
    which leaves the hip hard stops at exactly the joint limits."""
    xc = leg_plane_x(side)
    b = servo_bands()
    H = S.hip_module
    a0, a1 = b["arm"]
    cl = PR.servo_pocket_clearance
    case_top = G.case_center[1] + G.case[1] / 2              # long end above the hip axis (servo +Y_S = up)
    top = case_top + cl + H.top_wall
    yw = G.case[0] / 2 + cl + S.pocket_wall
    shape = box(xc - a1, xc + a1, -yw, yw, H.bottom_z, top)
    for s in (-1, 1):
        x0, x1 = sorted((xc + s * a0, xc + s * a1))
        shape = shape + box(x0, x1, H.back_extent, H.front_extent, H.bottom_z, top)
    shape = shape - servo.simplified(clearance=cl, horns=False).moved(hip_servo_loc(side))
    # open the pocket downward so the servo slides in from below
    shape = shape - box(xc - G.case[2] / 2 - cl, xc + G.case[2] / 2 + cl, -G.case[0] / 2 - cl, G.case[0] / 2 + cl,
                        H.bottom_z - 1, 0.0)
    # hip hard stops: thigh arm (hub r = hw, strip down to the thigh) swept over the hip range
    lo, hi = P.joints.hip_pitch.range
    bands = [(xc - a1 - 0.5, xc - a0), (xc + a0, xc + a1 + 0.5)]
    shape = carve_joint(shape, (0.0, 0.0), S.arm_half_width, 60.0, lo, hi, bands)
    # horn clearance through the horn bands, and slots for the horns while the servo slides in from below
    hr = G.horn_diameter / 2 + S.hub_clearance
    shape = shape - cyl_x((0, 0), hr, xc - a0, xc + a0)
    for s in (-1, 1):
        x0, x1 = sorted((xc + s * (G.case[2] / 2 - 0.1), xc + s * (a0 + 0.01)))
        shape = shape - box(x0, x1, -hr, hr, H.bottom_z - 1, 0.0)
    # case screws through the side walls (both sides) into the Ø1.6 holes at the far end
    for hx, hy in G.mount_holes:
        if hy < 10:
            continue                                          # holes near the shaft are inside the horn region
        for s in (-1, 1):
            x0, x1 = sorted((xc + s * G.case[2] / 2, xc + s * (a1 + 1)))
            shape = shape - cyl_x((-hx, hy), PR.screw_m2_clear / 2, x0, x1)
    # connector openings in the front/back walls (cable exits on the ±X_S faces -> ∓Y)
    for cx, cy, cz in G.cable_exit:
        xw = xc - b["horn_face"] - cz
        shape = shape - box(xw - 5, xw + 5, -yw - 1, yw + 1, cy - 4.5, cy + 4.5)
    # heat-set inserts for the pelvis screws (M2.5) in the top
    for dx in (-1, 1):
        for dy in (-1, 1):
            shape = shape - Pos(xc + dx * (G.case[2] / 2 + 2.5), dy * 7.0, top - PR.insert_m2_depth / 2) * \
                cyl_z(PR.insert_m25_hole / 2, PR.insert_m2_depth)
    return Body(f"hip_module_{side}", shape, "petg_structural", printed=True, print_rot=(0, 90, 0),
                note="print on its side (X vertical): sagittal loads stay in-layer")


def cyl_z(r, h):
    from build123d import Cylinder
    return Cylinder(r, h)


def pelvis_plate() -> Body:
    pp = S.pelvis
    xs = P.geometry.hip_spacing / 2 + servo_bands()["arm"][1]
    z0, z1 = pp.plate_z
    y0, y1 = pp.plate_y
    shape = box(-xs, xs, y0, y1, z0, z1)
    # lightening window in the middle (cables pass up to the tray)
    shape = shape - box(-pp.spine_half_width, pp.spine_half_width, -10, 10, z0 - 1, z1 + 1)
    # tray bolts (M2.5 through, nut underneath)
    mx, my = S.tray.mount_xy
    for sx_ in (-1, 1):
        for sy_ in (-1, 1):
            shape = shape - Pos(sx_ * mx, sy_ * my, (z0 + z1) / 2) * cyl_z(PR.insert_m25_hole / 2 - 0.4, z1 - z0 + 2)
    # screw holes to the hip modules
    for side in ("L", "R"):
        xc = leg_plane_x(side)
        for dx in (-1, 1):
            for dy in (-1, 1):
                shape = shape - Pos(xc + dx * (G.case[2] / 2 + 2.5), dy * 7.0, (z0 + z1) / 2) * \
                    cyl_z(1.4, z1 - z0 + 2)
    return Body("pelvis_plate", shape, "petg_structural", printed=True, print_rot=(0, 0, 0),
                note="print flat (Z up): the hip loads bend it about Y, stresses along X stay in-layer")


def tray() -> Body:
    """Electronics tray (reprinted when the electronics are chosen): floor with a hole grid, battery rail on
    four posts with strap slots every 5 mm along the battery's fore-aft travel."""
    t = S.tray
    z0, z1 = t.floor_z
    sx, sy = t.size_xy
    shape = box(-sx / 2, sx / 2, -sy / 2, sy / 2, z0, z1)
    n_x, n_y = int(sx // t.hole_pitch), int(sy // t.hole_pitch)
    for i in range(n_x):
        for j in range(n_y):
            x = (i - (n_x - 1) / 2) * t.hole_pitch
            y = (j - (n_y - 1) / 2) * t.hole_pitch
            shape = shape - Pos(x, y, (z0 + z1) / 2) * cyl_z(t.hole_d / 2, z1 - z0 + 2)
    r = t.battery_rail
    rz0, rz1 = r.rail_z
    rx = r.rail_x
    for xa, xb in ((rx[0], rx[1]), (rx[2], rx[3])):
        rail = box(xa, xb, -sy / 2, sy / 2, rz0, rz1)
        for py in (-sy / 2 + r.post_half, sy / 2 - r.post_half):
            rail = rail + box(xa, xb, py - r.post_half, py + r.post_half, z1, rz0)
        travel = P.electronics.battery_rail.travel
        y = travel[0]
        while y <= travel[1] + 1e-6:
            rail = rail - box(xa + 1.5, xb - 1.5, y - r.strap_slot[0] / 2, y + r.strap_slot[0] / 2, rz0 - 1, rz1 + 1)
            y += 5.0
        shape = shape + rail
    return Body("tray", shape, "petg_light", printed=True, print_rot=(0, 0, 0))


def electronics() -> list[Body]:
    out = []
    for name, e in P.electronics.items():
        if not isinstance(e, dict) or "size" not in e:
            continue
        if e.get("enabled", True) is False:
            continue
        sx, sy, sz = e["size"]
        x, y, z = e["position"]
        out.append(Body(f"elec_{name}", box(x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2, z - sz / 2, z + sz / 2),
                        "purchased", mass=e["mass"], mirror="none", note="placeholder"))
    return out


def hip_servo(side="L") -> Body:
    loc = hip_servo_loc(side)
    return Body(f"hip_servo_{side}", servo.simplified().moved(loc), "purchased", mass=P.servo.mass,
                visual=servo.vendor_shape().moved(loc), mirror="rotate")


def base_bodies() -> list[Body]:
    """All base_link bodies (both sides built explicitly; the right side mirrors the left in links.py)."""
    f = S.fasteners
    bodies = [hip_module("L"), hip_servo("L"), pelvis_plate(), tray()] + electronics()
    xc = leg_plane_x("L")
    z_top = S.pelvis.plate_z[1]
    bodies += [
        lumped("screws_hip_module_L", 4 * (f.m25_screw + f.m2_insert) + 4 * f.m2_screw, (xc, 0, 20)),
        lumped("screws_tray", 4 * (f.m25_screw + f.m25_nut), (0, 0, z_top)),
        lumped("wiring_base", S.wiring.base_link, (0, 0, 35), note="bus wiring, battery leads, switch leads"),
    ]
    return bodies
