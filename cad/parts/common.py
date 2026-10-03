"""Shared helpers for part code: the Body record, placement, side-view profiles, joint sweeps.

Frames: every part is built in its link frame for the LEFT leg (link origin on the joint axis to the
parent, x measured from the leg's centre plane, outward = -X). The right leg is produced by mirroring
(links.py). Side-view profiles are drawn in the YZ plane as (y, z) and extruded along X.

Hard stops: a parent's material in the lateral band of a child's horn arm is cut by the region the arm
sweeps over the joint range. The cut's boundary at each end of the range is exactly the arm's edge at the
limit angle, so the stop engages at the limit by construction (checked in checks.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from build123d import (Axis, Box, Circle, Cylinder, Face, Location, Plane, Polygon, Pos, Rectangle, Rot, Shape,
                       Sketch, Solid, Vector, extrude)

from params import P

MATERIALS = set(P.materials.keys())


@dataclass
class Body:
    """One solid with a material tag. Printed parts: mass = volume × effective density.
    Purchased parts: `mass` is given (g) and density is derived from the simplified volume."""

    name: str
    shape: Shape
    material: str                    # a key of robot.yaml materials, or "purchased"
    mass: float | None = None
    printed: bool = False
    print_rot: tuple = (0.0, 0.0, 0.0)   # rotation (deg about X, Y, Z) to the suggested print orientation
    collide: bool = True             # part of the interference sweep
    visual: Shape | None = None      # display geometry if different (vendor servo)
    mirror: str = "mirror"           # right side: "mirror" (printed) or "rotate" (real purchased part)
    press_fit: float = 0.0           # designed press-fit overlap (mm³) with its seat in the same link; 0 = none
    note: str = ""

    def __post_init__(self):
        if self.material == "purchased":
            if self.mass is None:
                raise ValueError(f"{self.name}: purchased part without mass")
        elif self.material not in MATERIALS:
            raise ValueError(f"{self.name}: material tag {self.material!r} missing from robot.yaml materials")


# ------------------------------------------------------------------ placement
def frame(origin, x_dir, z_dir) -> Location:
    return Location(Plane(origin=Vector(*origin), x_dir=Vector(*x_dir), z_dir=Vector(*z_dir)))


def servo_frame(origin, long_end: str) -> Location:
    """Servo frame S placed in a link frame (left leg): output horn faces outward (-X); the case's long end
    points up (+Z) or down (-Z). origin = output horn face centre."""
    ys = (0, 0, 1) if long_end == "up" else (0, 0, -1)
    zs = (-1, 0, 0)
    xs = (ys[1] * zs[2] - ys[2] * zs[1], ys[2] * zs[0] - ys[0] * zs[2], ys[0] * zs[1] - ys[1] * zs[0])
    return frame(origin, xs, zs)


# ------------------------------------------------------------------ side-view profiles
def yz(sk: Sketch | Face) -> Face:
    """Map a sketch drawn in XY (x→y, y→z) onto the YZ plane."""
    return Plane.YZ * sk


def prism(face_xy, x0: float, x1: float) -> Solid:
    """Profile drawn in XY as (y, z) -> solid spanning x0 … x1."""
    f = Plane.YZ.offset(x0) * face_xy
    return extrude(f, amount=x1 - x0, dir=(1, 0, 0))          # explicit: polygon winding must not matter


def poly(*pts) -> Sketch:
    return Sketch() + Polygon(*pts, align=None)


def circle(c, r) -> Sketch:
    return Pos(c[0], c[1]) * Circle(r)


def rect(y0, y1, z0, z1) -> Sketch:
    return Pos((y0 + y1) / 2, (z0 + z1) / 2) * Rectangle(abs(y1 - y0), abs(z1 - z0))


def strip(hub, hw: float, length: float, angle: float = 0.0) -> Sketch:
    """Hub disc of radius hw + a strip of half-width hw from the hub pointing 'down' (-z), rotated by
    `angle` (deg, about +X: toward +y first is negative... see note) around the hub.

    Note: rotation about +X maps +y toward +z, so in the (y, z) sketch it is a counter-clockwise rotation.
    A child rotated by joint angle a about +X has its 'down' strip pointing at (sin a, -cos a)."""
    s = circle((0, 0), hw) + Pos(0, -length / 2) * Rectangle(2 * hw, length)
    return Pos(hub[0], hub[1]) * Rot(0, 0, angle) * s


def swept_strip(hub, hw: float, length: float, a_lo: float, a_hi: float, step: float = 2.0) -> Sketch:
    """Union of the strip over rotation angles a_lo … a_hi (deg about +X), both ends included exactly."""
    n = max(2, int(abs(a_hi - a_lo) / step) + 1)
    angs = [a_lo + (a_hi - a_lo) * i / (n - 1) for i in range(n)]
    sk = strip(hub, hw, length, angs[0])
    for a in angs[1:]:
        sk = sk + strip(hub, hw, length, a)
    return sk


def carve_joint(parent: Shape, hub, hw: float, length: float, a_lo: float, a_hi: float, bands) -> Shape:
    """Remove from `parent` the region swept by a child's horn strip in each lateral band."""
    S = P.structure
    sw = swept_strip(hub, hw, length + S.sweep_tip_clearance, a_lo, a_hi) + circle(hub, hw + S.hub_clearance)
    for x0, x1 in bands:
        parent = parent - prism(sw, x0, x1)
    return parent


def cyl_x(c_yz, r: float, x0: float, x1: float) -> Solid:
    """Cylinder along X through (y, z) = c_yz from x0 to x1."""
    return Pos((x0 + x1) / 2, c_yz[0], c_yz[1]) * Rot(0, 90, 0) * Cylinder(r, abs(x1 - x0))


def box(x0, x1, y0, y1, z0, z1) -> Solid:
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))


def horn_holes(hub, x0: float, x1: float, n: int | None = None, d: float | None = None) -> Shape:
    """Screw holes through a horn-mounted plate on the horn's hole circle."""
    import math
    g = P.servo.geometry
    n = n or P.structure.horn_screws
    d = d or P["print"]["screw_m2_clear"]
    r = g.horn_hole_pcd / 2
    out = None
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        c = cyl_x((hub[0] + r * math.cos(a), hub[1] + r * math.sin(a)), d / 2, x0, x1)
        out = c if out is None else out + c
    return out


def lumped(name: str, mass: float, pos, size: float = 2.0, note: str = "") -> Body:
    """Small fastener/wiring mass as a cube at `pos` (mass only; not collided)."""
    return Body(name, Pos(*pos) * Box(size, size, size), "purchased", mass=mass, collide=False, mirror="rotate",
                note=note)


def ring_x(c_yz, D: float, d: float, x0: float, x1: float) -> Solid:
    """Annulus along X (bearing / clutch): outer Ø D, bore Ø d, from x0 to x1."""
    return cyl_x(c_yz, D / 2, x0, x1) - cyl_x(c_yz, d / 2, x0 - 1, x1 + 1)


def bolt_x(c_yz, d: float, x_shank: tuple, head: tuple | None, head_side: float,
           nut: tuple | None = None) -> Solid:
    """Screw / pin along X: shank Ø d over x_shank, a head (Ø, h) beyond the end on `head_side` (±1),
    optionally a nut (Ø, h) beyond the other end."""
    x0, x1 = sorted(x_shank)
    s = cyl_x(c_yz, d / 2, x0, x1)
    if head:
        hx = x0 if head_side < 0 else x1
        s = s + cyl_x(c_yz, head[0] / 2, hx, hx + head_side * head[1]) if head_side > 0 else \
            s + cyl_x(c_yz, head[0] / 2, hx - head[1], hx)
    if nut:
        nx = x1 if head_side < 0 else x0
        s = s + (cyl_x(c_yz, nut[0] / 2, nx, nx + nut[1]) if head_side < 0 else cyl_x(c_yz, nut[0] / 2, nx - nut[1], nx))
    return s
