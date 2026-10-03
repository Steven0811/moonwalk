"""Foot (two variants), toe pad, wheel set — left leg, foot frame (origin on the ankle axis; ground at
z = -ankle_height with the foot flat).

Contact geometry (pad recess, pivots, switch-over angle) comes from cad/analysis/kinematics.foot_geom, the
same derivation phase 1 used.
"""
from __future__ import annotations

import math

from build123d import Pos, Rot

from params import P, foot_geom, servo_bands
from parts.common import Body, box, carve_joint, circle, cyl_x, lumped, poly, prism, rect, strip

S = P.structure
PR = P["print"]
GEO = P.geometry
F = S.foot


def pad_polygon(fg):
    """Toe pad cross-section (y, z): the wedge face from the rear edge to the tip, thickness normal to it."""
    t = GEO.foot.toe_pad.thickness
    a = fg.face
    n = (-math.sin(a), math.cos(a))
    r, tip = tuple(fg.pad_rear), tuple(fg.pad_tip)
    return [r, tip, (tip[0] + t * n[0], tip[1] + t * n[1]), (r[0] + t * n[0], r[1] + t * n[1])]


def underside(fg, y: float) -> float:
    """Lowest allowed z of the foot body (wheel variant) at y between the axle and the pad: above the flat
    ground and above the ground line of the switch posture, plus clearance."""
    ground = -fg.h
    sw_line = fg.pad_rear[1] + (y - fg.pad_rear[0]) * math.tan(fg.switch)
    return max(ground, sw_line) + F.underside_clearance


def foot(variant: str | None = None) -> Body:
    fg = foot_geom(variant)
    v = fg.variant
    b = servo_bands()
    a0, a1 = b["arm"]
    hw = GEO.foot.width / 2
    h = fg.h
    pt = F.plate_thickness
    fi, ft = F.fork_inner, F.fork_thickness
    pad = pad_polygon(fg)
    seat_t = F.pad_seat_thickness
    # ---- sole plate (side view), extruded over the full width
    if v == "wheel":
        hb = F.heel_block
        y_mid0 = hb.y[1]
        ys = [y_mid0 + (fg.pad_rear[0] - y_mid0) * i / 8 for i in range(9)]
        bottom = [(y, underside(fg, y)) for y in ys]
        plate = poly(*bottom, *[(y, z + pt) for y, z in reversed(bottom)])
        w = GEO.foot.wheel
        heel_top = -h + w.diameter + 0.8 + F.friction_pad_tab.thickness     # carries the friction-pad tabs
        zb = lambda y: max(-h + w.heel_clearance, underside(fg, y))
        ysb = [hb.y[0]] + [w.axle_y + (hb.y[1] + 0.5 - w.axle_y) * i / 6 for i in range(7)]
        heel = poly(*[(y, zb(y)) for y in ysb], (hb.y[1] + 0.5, heel_top), (hb.y[0], heel_top))
        prof = plate + heel
        body = prism(prof, -hw, hw)
        # heel block is narrower than the foot: wheels run outside it
        for s in (-1, 1):
            x0, x1 = sorted((s * hb.half_width, s * (hw + 1)))
            body = body - box(x0, x1, hb.y[0] - 1, hb.y[1] + 0.49, -h - 1, -h + GEO.foot.wheel.diameter + 1.0)
    else:
        sole_front = fg.sole_front
        bottom = [(-fg.heel, -h), (sole_front, -h)]
        # recessed underside between the sole front edge and the pad (rises to the pad rear edge + clearance)
        prof = poly((-fg.heel, -h), (sole_front, -h), (fg.pad_rear[0], fg.pad_rear[1] + F.underside_clearance),
                    (fg.pad_rear[0], fg.pad_rear[1] + F.underside_clearance + pt), (sole_front, -h + pt),
                    (-fg.heel, -h + pt))
        body = prism(prof, -hw, hw)
    # ---- pad seat: plate above the toe pad, joined to the sole plate
    (ry, rz), (ty, tz), (ty2, tz2), (ry2, rz2) = pad
    n = (-math.sin(fg.face), math.cos(fg.face))
    seat = poly((ry2, rz2), (ty2, tz2), (ty2 + seat_t * n[0], tz2 + seat_t * n[1]),
                (ry2 + seat_t * n[0] - 3.0, rz2 + seat_t * n[1]), (ry2 - 3.0, rz2 - 0.5))
    body = body + prism(seat, -hw, hw)
    z_top_front = max(rz2 + seat_t * n[1], tz2 + seat_t * n[1])
    # ---- fork arms around the shank tongue, outer one thickened into the lever spacer + lever
    hub_r = F.hub_radius
    y_top = lambda y: (underside(fg, y) + pt) if v == "wheel" else (-h + pt)
    z_plate = min(y_top(-hub_r), y_top(hub_r))
    arm_prof = circle((0, 0), hub_r) + rect(-hub_r, hub_r, z_plate - 0.5, 0)
    body = body + prism(arm_prof, fi, fi + ft)
    body = body + prism(arm_prof, -a0, -fi)                       # outer fork arm + spacer up to the arm band
    k = S.linkage
    lever = circle((0, 0), hub_r) + circle((k.crank_length, 0), k.lobe_radius) + \
        rect(0, k.crank_length, -k.lobe_radius, k.lobe_radius)
    body = body + prism(lever + arm_prof, -a1, -a0)
    body = body - cyl_x((k.crank_length, 0), PR.insert_m2_hole / 2, -a1 - 1, -a0 + 1)
    # ankle pin hole
    body = body - cyl_x((0, 0), S.ankle.pin.d / 2 + PR.shaft_clear, -a1 - 1, fi + ft + 1)
    # ---- ankle hard stops: blocks between the fork arms, cut by the shank keel sweep
    lo, hi = P.joints.ankle_pitch.range
    xb = fi - 0.4
    blk = box(-xb, xb, -16.0, 16.0, z_plate - 0.5, F.stop_top)
    blk = carve_joint(blk, (0, 0), S.shank.keel.half_width, S.shank.keel.length, -hi, -lo, [(-xb - 1, xb + 1)])
    blk = blk - cyl_x((0, 0), S.shank.tongue_radius + S.hub_clearance, -xb - 1, xb + 1)
    body = body + blk
    # ---- wheel variant: axle bore, clutch and bearing seats, friction-pad tabs over the wheels
    if v == "wheel":
        w = GEO.foot.wheel
        ws = S.wheel_set
        c = (w.axle_y, -h + w.diameter / 2)
        hb = F.heel_block
        body = body - cyl_x(c, ws.clutch.d / 2 + PR.shaft_clear, -hw - 1, hw + 1)
        body = body - cyl_x(c, ws.clutch.D / 2 + PR.bearing_press / 2, -ws.clutch.B / 2, ws.clutch.B / 2)
        for s in (-1, 1):
            x0, x1 = sorted((s * hb.half_width, s * (hb.half_width - ws.bearing.B)))
            body = body - cyl_x(c, ws.bearing.D / 2 + PR.bearing_press / 2, x0 - 0.01, x1)
        tab = F.friction_pad_tab
        z0 = -h + w.diameter + 0.8
        for s in (-1, 1):
            x0, x1 = sorted((s * (hb.half_width - 2), s * hw))
            t_ = box(x0, x1, w.axle_y - 3, w.axle_y + 3, z0, z0 + tab.thickness)
            t_ = t_ - Pos(s * (hw - w.width / 2), w.axle_y, z0 + 1) * _cyl_z(tab.hole / 2, 4)
            body = body + t_
    # the pad (with its locating nubs) is glued into the seat: subtract it so the seat fits exactly
    body = body - toe_pad(variant).shape
    return Body(f"foot", body, "petg_structural", printed=True, print_rot=(0, 90, 0),
                note=f"{v} variant; print on its side (X vertical)")


def _cyl_z(r, hgt):
    from build123d import Cylinder
    return Cylinder(r, hgt)


def toe_pad(variant: str | None = None) -> Body:
    """TPU wedge pad (same part for both variants; only its seat height differs)."""
    fg = foot_geom(variant)
    hw = GEO.foot.width / 2
    sh = prism(poly(*pad_polygon(fg)), -hw, hw)
    n = (-math.sin(fg.face), math.cos(fg.face))
    (ry, rz), (ty, tz) = fg.pad_rear, fg.pad_tip
    nub = S.foot.pad_nub
    t = GEO.foot.toe_pad.thickness
    for s in (-1, 1):
        cy, cz = (ry + ty) / 2 + n[0] * (t + nub.h / 2), (rz + tz) / 2 + n[1] * (t + nub.h / 2)
        sh = sh + Pos(s * hw / 2, cy, cz) * Rot(-math.degrees(fg.face), 0, 0) * _cyl_z(nub.d / 2, nub.h)
    return Body("toe_pad", sh, "tpu_85a", printed=True, print_rot=(-math.degrees(fg.face) + 180, 0, 0),
                note="print face down")


def wheel_set() -> list[Body]:
    """L_wheel link bodies (wheel frame: origin on the axle): two TPU wheels fixed to a 3 mm steel axle."""
    w = GEO.foot.wheel
    ws = S.wheel_set
    hw = GEO.foot.width / 2
    out = []
    for s, nm in ((-1, "out"), (1, "in")):
        x0, x1 = sorted((s * (hw - w.width), s * hw))
        # nominal bore (the press fit itself is a print parameter, not modelled as overlap)
        sh = cyl_x((0, 0), w.diameter / 2, x0, x1) - cyl_x((0, 0), w.axle_diameter / 2, x0 - 1, x1 + 1)
        out.append(Body(f"wheel_{nm}", sh, "tpu_85a", printed=True, print_rot=(0, 90, 0)))
    out.append(Body("axle", cyl_x((0, 0), w.axle_diameter / 2, -hw, hw), "purchased", mass=ws.axle_mass,
                    mirror="rotate", note="3 mm hardened steel pin"))
    return out


def foot_hardware(variant: str | None = None) -> list[Body]:
    fg = foot_geom(variant)
    f = S.fasteners
    from parts import hardware
    out = [hardware.ankle_pin(), lumped("lever_insert", f.m2_insert, (-15, S.linkage.crank_length, 0))]
    if fg.variant == "wheel":
        out += hardware.wheel_bearings()
    else:
        out.append(lumped("ptfe_tape", 1.0, (0, (-fg.heel + fg.sole_front) / 2, -fg.h + 0.5), note="PTFE tape on the sole"))
    return out
