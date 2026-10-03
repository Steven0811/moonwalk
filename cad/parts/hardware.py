"""Purchased mechanical hardware drawn as simple solids (bearings, clutch, pins) — left leg, link frames.

Masses are the given part masses (robot.yaml); they replace the lumped point masses used before.
Bearings and the clutch are press-fitted: their overlap with the printed seat is a designed overlap.
"""
from __future__ import annotations

import math

from build123d import Pos

from params import P, servo_bands
from parts.common import Body, bolt_x, ring_x

S = P.structure
GEO = P.geometry


def _press(D: float, B: float) -> float:
    """Expected overlap (mm³) of a part of Ø D, length B pressed into a seat of Ø D + print.bearing_press."""
    return math.pi * D * abs(P["print"].bearing_press) / 2 * B


def _steel(name, shape, mass, note="", press=None):
    return Body(name, shape, "purchased", mass=mass, mirror="rotate",
                press_fit=_press(*press) if press else 0.0, note=note)


def ankle_bearings() -> list[Body]:
    """Two MR63ZZ pressed into the faces of the shank tongue (shank frame)."""
    b = S.ankle.bearing
    t = S.shank.tongue_half_width
    za = -GEO.shank
    return [_steel(f"ankle_bearing_{nm}", ring_x((0, za), b.D, b.d, x0, x1), b.mass, b.name, press=(b.D, b.B))
            for nm, (x0, x1) in (("out", (-t, -t + b.B)), ("in", (t - b.B, t)))]


def ankle_pin() -> Body:
    """M3 x 30 screw through lever, fork arms and the two bearings, nylock nut on the inner side (foot frame)."""
    a1 = servo_bands()["arm"][1]
    F = S.foot
    p, n = S.ankle.pin, S.ankle.nut
    x_in = F.fork_inner + F.fork_thickness
    shape = bolt_x((0, 0), p.d, (-a1, x_in), tuple(p.head), -1, (n.D, n.h))
    return _steel("ankle_pin", shape, p.mass + n.mass, f"{p.name} + {n.name}")


def linkage_hardware(q_deg: float = 0.0) -> list[Body]:
    """MR52ZZ in both rod ends and the two M2 shoulder screws (crank pin, lever pin), posed like the rod
    (shank frame)."""
    k = S.linkage
    a0, a1 = servo_bands()["arm"]
    zs = S.shank.ankle_servo_shaft_z
    za = -GEO.shank
    x1 = -a1 - k.rod_gap
    x0 = x1 - k.rod_thickness
    off = Pos(0, k.crank_length * math.cos(math.radians(q_deg)), k.crank_length * math.sin(math.radians(q_deg)))
    out = []
    for nm, zc in (("crank", zs), ("lever", za)):
        out.append(_steel(f"rod_bearing_{nm}", off * ring_x((0, zc), k.bearing.D, k.bearing.d, x0, x1),
                          k.bearing.mass, k.bearing.name, press=(k.bearing.D, k.bearing.B)))
        out.append(_steel(f"pin_{nm}", off * bolt_x((0, zc), k.pin.d, (x0, -a0), tuple(k.pin.head), -1),
                          k.pin.mass, k.pin.name))
    return out


def wheel_bearings() -> list[Body]:
    """HF0306 one-way clutch (centre) and two MR63ZZ (ends of the heel block) on the axle (foot frame)."""
    ws = S.wheel_set
    w = GEO.foot.wheel
    c = (w.axle_y, -GEO.ankle_height + w.diameter / 2)
    hb = S.foot.heel_block.half_width
    cl, br = ws.clutch, ws.bearing
    out = [_steel("wheel_clutch", ring_x(c, cl.D, cl.d, -cl.B / 2, cl.B / 2), cl.mass,
                  f"{cl.name} one-way needle clutch", press=(cl.D, cl.B))]
    for nm, (x0, x1) in (("out", (-hb, -hb + br.B)), ("in", (hb - br.B, hb))):
        out.append(_steel(f"wheel_bearing_{nm}", ring_x(c, br.D, br.d, x0, x1), br.mass, br.name, press=(br.D, br.B)))
    return out
