"""Thigh, shank and ankle linkage (left leg, each in its own link frame).

thigh frame: origin on the hip axis at the leg centre plane. Knee axis at (0, 0, -thigh).
shank frame: origin on the knee axis. Ankle axis at (0, 0, -shank). Ankle servo high in the shank, driving
the foot through a parallelogram (crank on the servo horn, rod, lever on the foot): ankle angle = servo angle.
"""
from __future__ import annotations

import functools

from build123d import Pos

from params import P, servo_bands
from parts import servo
from parts.common import (Body, box, carve_joint, circle, cyl_x, horn_holes, lumped, poly, prism, rect,
                          servo_frame, strip)

S = P.structure
PR = P["print"]
G = P.servo.geometry
GEO = P.geometry


def _bands():
    b = servo_bands()
    a0, a1 = b["arm"]
    return b, a0, a1


def knee_servo_loc():
    b, _, _ = _bands()
    return servo_frame((-b["horn_face"], 0.0, -GEO.thigh), "up")


def ankle_servo_loc():
    b, _, _ = _bands()
    return servo_frame((-b["horn_face"], 0.0, S.shank.ankle_servo_shaft_z), "down")


def _horn_recess(hub, x_face, outward: float):
    """Locating recess for the stock horn in a horn-mounted plate whose inner face is 1 mm inside the horn,
    plus a centre hole clearing the stock horn screw head (and giving screwdriver access to it)."""
    r = G.horn_diameter / 2 + PR.horn_recess_clearance
    d = PR.horn_recess_depth
    x0, x1 = sorted((x_face, x_face - outward * d))
    centre = cyl_x(hub, S.horn_screw_clear_radius, min(x0, x_face + outward * 10) - 0.01,
                   max(x1, x_face + outward * 10) + 0.01)
    return cyl_x(hub, r, x0 - 0.01, x1 + 0.01) + centre


def _cable_slots(loc_origin_z, long_up: bool, yw: float):
    """Openings in the front/back pocket walls at the servo's connector positions."""
    out = None
    b, _, _ = _bands()
    for cx, cy, cz in G.cable_exit:
        xw = -b["horn_face"] - cz
        zc = loc_origin_z + (cy if long_up else -cy)
        s = box(xw - 5, xw + 5, -yw - 1, yw + 1, zc - 4.5, zc + 4.5)
        out = s if out is None else out + s
    return out


def _servo_screws(origin_z, long_up: bool, sides=(-1, 1)):
    """Case screws through the side walls at the far-end case holes."""
    out = None
    for hx, hy in G.mount_holes:
        if hy < 10:
            continue
        y = -hx if long_up else hx
        z = origin_z + (hy if long_up else -hy)
        for s in sides:
            x0, x1 = sorted((s * G.case[2] / 2, s * (servo_bands()["arm"][1] + 1)))
            c = cyl_x((y, z), PR.screw_m2_clear / 2, x0, x1)
            out = c if out is None else out + c
    return out


# ---------------------------------------------------------------------------- thigh
def thigh() -> Body:
    b, a0, a1 = _bands()
    T = S.thigh
    L1 = GEO.thigh
    hw = S.arm_half_width
    cl = PR.servo_pocket_clearance
    zk = -L1
    yu, yl = T.upper_half_width, T.lower_half_width
    zt0, zt1 = T.taper_z
    case_top = zk + G.case_center[1] + G.case[1] / 2           # knee servo long end up
    case_bot = zk + G.case_center[1] - G.case[1] / 2
    # side-view outline of the whole thigh (all bands), then per-band trimming
    outline = (rect(-yu, yu, zt0, -T.top_gap) + poly((-yu, zt0), (yu, zt0), (yl, zt1), (-yl, zt1))
               + rect(-yl, yl, case_bot - cl, zt1))
    body = prism(outline, -a1, a1)
    # arm bands: hub + strip up to the hip, kneecap flange below the knee
    arm = strip((0, 0), hw, T.top_gap + 1) + rect(-yl, yl, zk - T.kneecap.below, case_bot)
    kc = rect(T.kneecap.y[0], T.kneecap.y[1], zk - T.kneecap.below, zt1)
    for s in (-1, 1):
        x0, x1 = sorted((s * a0, s * a1))
        body = body + prism(arm + kc, x0, x1)
    # tube cavity (cable channel) and knee-servo pocket
    w = T.wall
    body = body - box(-a0 + 0.2, a0 - 0.2, -yu + w, yu - w, zt0 + 1, -T.top_gap + 1)
    body = body - box(-a0 + 0.2, a0 - 0.2, -yl + w, yl - w, case_top + cl + S.pocket_wall, zt0 + 1.01)
    body = body - servo.simplified(clearance=cl, horns=False).moved(knee_servo_loc())
    body = body - box(-G.case[2] / 2 - cl, G.case[2] / 2 + cl, -G.case[0] / 2 - cl, G.case[0] / 2 + cl,
                      case_bot - 5, zk)                             # pocket open below: servo slides in
    hr = G.horn_diameter / 2 + S.hub_clearance
    for sg in (-1, 1):                                              # slots for the horns on the way in
        x0, x1 = sorted((sg * (G.case[2] / 2 - 0.1), sg * (a0 + 0.01)))
        body = body - box(x0, x1, -hr, hr, case_bot - 5, zk)
    # horn clearances at the knee (horn bands) and horn recesses at the hip (arm bands)
    body = body - cyl_x((0, zk), G.horn_diameter / 2 + S.hub_clearance, -a0, a0)
    for s in (-1, 1):
        body = body - _horn_recess((0, 0), s * a0 + s * PR.horn_recess_depth, s)
    body = body - horn_holes((0, 0), -a1 - 1, -a0 + 1) - horn_holes((0, 0), a0 - 1, a1 + 1)
    # knee hard stops: shank arms swept over the knee range (knee axis is -X: a = -q)
    lo, hi = P.joints.knee_pitch.range
    sh = S.shank
    body = carve_joint(body, (0, zk), hw, -sh.arm_bottom, -hi, -lo, [(-a1 - 0.5, -a0), (a0, a1 + 0.5)])
    body = body - _servo_screws(zk, True) - _cable_slots(zk, True, yl)
    return Body("thigh", body, "petg_structural", printed=True, print_rot=(0, 90, 0),
                note="print on its side (X vertical)")


def knee_servo() -> Body:
    loc = knee_servo_loc()
    return Body("knee_servo", servo.simplified().moved(loc), "purchased", mass=P.servo.mass,
                visual=servo.vendor_shape().moved(loc), mirror="rotate")


# ---------------------------------------------------------------------------- shank
@functools.lru_cache(maxsize=1)
def _shank_split():
    b, a0, a1 = _bands()
    sh = S.shank
    L2 = GEO.shank
    hw = S.arm_half_width
    cl = PR.servo_pocket_clearance
    zs = sh.ankle_servo_shaft_z
    za = -L2
    case_top = zs - G.case_center[1] + G.case[1] / 2         # long end down: short end above the shaft
    case_bot = zs - G.case_center[1] - G.case[1] / 2
    yw = G.case[0] / 2 + cl + S.pocket_wall
    xw = S.side_plate[1]
    # knee arms (arm bands) down to the cap
    arm = strip((0, 0), hw, -sh.arm_bottom)
    body = None
    for s in (-1, 1):
        x0, x1 = sorted((s * a0, s * a1))
        p = prism(arm, x0, x1)
        body = p if body is None else body + p
    # cap joining the arms, servo pocket body, tongue with keel
    body = body + box(-a1, a1, -yw, yw, sh.arm_bottom, sh.body_top)
    bot = case_bot - cl - S.pocket_wall
    body = body + box(-xw, xw, -yw, yw, bot, sh.body_top)
    t = sh.tongue_half_width
    tongue = circle((0, za), sh.tongue_radius) + rect(-sh.tongue_radius, sh.tongue_radius, za, bot + 0.5) + \
        rect(-sh.keel.half_width, sh.keel.half_width, za - sh.keel.length, za)
    body = body + prism(tongue, -t, t)
    # ankle servo pocket (open toward the inner side for insertion), horn clearances
    body = body - servo.simplified(clearance=cl, horns=False).moved(ankle_servo_loc())
    body = body - cyl_x((0, zs), G.horn_diameter / 2 + S.hub_clearance, -a1 - 1, a1 + 1)
    # horn recesses + screws at the knee
    for s in (-1, 1):
        body = body - _horn_recess((0, 0), s * a0 + s * PR.horn_recess_depth, s)
    body = body - horn_holes((0, 0), -a1 - 1, -a0 + 1) - horn_holes((0, 0), a0 - 1, a1 + 1)
    # ankle bearings (both faces of the tongue) and pin hole
    br = S.ankle.bearing
    body = body - cyl_x((0, za), P.structure.ankle.pin.d / 2 + PR.shaft_clear, -t - 1, t + 1)
    body = body - cyl_x((0, za), br.D / 2 + PR.bearing_press / 2, -t - 0.01, -t + br.B)
    body = body - cyl_x((0, za), br.D / 2 + PR.bearing_press / 2, t - br.B, t + 0.01)
    body = body - _servo_screws(zs, False, sides=(1,)) - _cable_slots(zs, False, yw)
    # the inner side wall is a separate cover: the servo goes in from the inner side, outer horn first
    x_face = G.case[2] / 2 + cl
    cut = box(x_face, xw + 0.01, -yw - 0.01, yw + 0.01, bot - 0.01, sh.arm_bottom)
    cover = body & cut
    body = body - cut
    # cover screws: 4 bosses beside the pocket (front/back x top/bottom) with M2 inserts along X; the cover
    # grows matching tabs. The pocket walls (1.8 mm) are too thin to take an insert themselves.
    cb = sh.cover_boss
    for zc in (bot + cb.z_offset, sh.arm_bottom - cb.z_offset):
        for sy in (-1, 1):
            y0, y1 = sorted((sy * (yw - 0.5), sy * cb.y_out))
            body = body + box(x_face - cb.depth, x_face, y0, y1, zc - cb.half_h, zc + cb.half_h)
            cover = cover + box(x_face, xw, y0, y1, zc - cb.half_h, zc + cb.half_h)
            body = body - cyl_x((sy * cb.insert_y, zc), PR.insert_m2_hole / 2, x_face - PR.insert_m2_depth, x_face + 0.01)
            cover = cover - cyl_x((sy * cb.insert_y, zc), PR.screw_m2_clear / 2, x_face - 0.01, xw + 0.01)
    return body, cover


def shank() -> Body:
    body, _ = _shank_split()
    return Body("shank", body, "petg_structural", printed=True, print_rot=(0, 90, 0),
                note="print on its side (X vertical)")


def ankle_cover() -> Body:
    _, cover = _shank_split()
    return Body("ankle_cover", cover, "petg_structural", printed=True, print_rot=(0, 90, 0),
                note="inner side wall of the ankle-servo pocket; 4 x M2 + 2 case screws")


def ankle_servo() -> Body:
    loc = ankle_servo_loc()
    return Body("ankle_servo", servo.simplified().moved(loc), "purchased", mass=P.servo.mass,
                visual=servo.vendor_shape().moved(loc), mirror="rotate")


# ---------------------------------------------------------------------------- linkage (posed by the ankle angle)
def crank(q_deg: float = 0.0) -> Body:
    """Crank on the ankle servo's output horn (outer arm band). Rotates with the ankle angle."""
    _, a0, a1 = _bands()
    k = S.linkage
    zs = S.shank.ankle_servo_shaft_z
    prof = circle((0, 0), S.arm_half_width) + circle((k.crank_length, 0), k.lobe_radius) + \
        poly((0, -k.lobe_radius), (k.crank_length, -k.lobe_radius), (k.crank_length, k.lobe_radius), (0, k.lobe_radius))
    sh = prism(prof, -a1, -a0)
    sh = sh - _horn_recess((0, 0), -a0 - PR.horn_recess_depth, -1) - horn_holes((0, 0), -a1 - 1, -a0 + 1)
    sh = sh - cyl_x((k.crank_length, 0), PR.insert_m2_hole / 2, -a1 - 1, -a0 + 1)
    from build123d import Rot
    sh = Pos(0, 0, zs) * Rot(q_deg, 0, 0) * sh
    return Body("crank", sh, "petg_structural", printed=True, print_rot=(0, 90, 0))


def rod(q_deg: float = 0.0) -> Body:
    """Parallelogram rod between the crank pin and the foot-lever pin (outside the arm band)."""
    import math
    _, a0, a1 = _bands()
    k = S.linkage
    zs = S.shank.ankle_servo_shaft_z
    za = -GEO.shank
    x1 = -a1 - k.rod_gap
    x0 = x1 - k.rod_thickness
    re = k.bearing.D / 2 + 1.5
    prof = circle((0, zs), re) + circle((0, za), re) + rect(-3.0, 3.0, za, zs)
    sh = prism(prof, x0, x1)
    for zc in (zs, za):
        sh = sh - cyl_x((0, zc), k.bearing.D / 2 + PR.bearing_press / 2, x0 - 1, x1 + 1)
    dy, dz = k.crank_length * math.cos(math.radians(q_deg)), k.crank_length * math.sin(math.radians(q_deg))
    sh = Pos(0, dy, dz) * sh
    return Body("rod", sh, "petg_structural", printed=True, print_rot=(0, 90, 0))


def leg_hardware_thigh() -> list[Body]:
    f = S.fasteners
    zk = -GEO.thigh
    return [
        lumped("screws_hip_horns", 2 * S.horn_screws * f.m2_screw, (0, 0, 0)),
        lumped("screws_knee_servo_case", 4 * f.m2_screw, (0, 0, zk + 22.5)),
        lumped("wiring_thigh", S.wiring.thigh, (0, 0, zk / 2)),
    ]


def leg_hardware_shank() -> list[Body]:
    f = S.fasteners
    k = S.linkage
    zs = S.shank.ankle_servo_shaft_z
    za = -GEO.shank
    return [
        lumped("screws_knee_horns", 2 * S.horn_screws * f.m2_screw, (0, 0, 0)),
        lumped("screws_ankle_servo_case", 2 * f.m2_screw + S.horn_screws * f.m2_screw, (0, 0, zs - 22.5)),
        lumped("crank_insert", f.m2_insert, (-15, k.crank_length, zs)),
        lumped("ankle_cover_screws_inserts", 4 * (f.m2_screw + f.m2_insert), (12.5, 0, zs - 7.5)),
        lumped("wiring_shank", S.wiring.shank, (0, 0, zs)),
    ]
