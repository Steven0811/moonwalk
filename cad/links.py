"""Parts -> links. Every link is a list of Body in its own frame.

The right leg is never modelled separately: every right-leg body is the mirror image of the left one about
the link's YZ plane. Real purchased parts (servos) are not mirror-imaged: their display model is placed by
the proper rotation equivalent to the mirror (the servo case is symmetric about its own x_S = 0 plane, so
the simplified shape used for pockets, mass and collisions is identical either way).
"""
from __future__ import annotations

import functools
from dataclasses import replace

from build123d import Plane

from params import P
from parts import base, foot, hardware, leg
from parts.common import Body
from parts import servo

LEG_LINKS = ("thigh", "shank", "foot", "wheel")


def link_name(link: str, side: str | None) -> str:
    return link if side is None else f"{side}_{link}"


def mirror_body(b: Body) -> Body:
    shape = b.shape.mirror(Plane.YZ)
    visual = None
    if b.visual is not None:
        # proper placement of the real part: mirror in the part's own frame, then in the link frame
        visual = b.visual.mirror(Plane.YZ)
        if getattr(b, "_s_loc", None) is not None:
            visual = servo.vendor_shape().mirror(Plane.YZ).moved(b._s_loc).mirror(Plane.YZ)
    rot = (b.print_rot[0], -b.print_rot[1], -b.print_rot[2])
    nb = replace(b, shape=shape, visual=visual, print_rot=rot)
    return nb


def _tag_servo(b: Body, loc) -> Body:
    b._s_loc = loc
    return b


@functools.lru_cache(maxsize=None)
def left_leg_link(link: str, variant: str) -> tuple:
    if link == "thigh":
        bodies = [leg.thigh(), _tag_servo(leg.knee_servo(), leg.knee_servo_loc())] + leg.leg_hardware_thigh()
    elif link == "shank":
        bodies = [leg.shank(), leg.ankle_cover(), _tag_servo(leg.ankle_servo(), leg.ankle_servo_loc())] + \
            hardware.ankle_bearings() + leg.leg_hardware_shank()
    elif link == "foot":
        bodies = [foot.foot(variant), foot.toe_pad(variant)] + foot.foot_hardware(variant)
    elif link == "wheel":
        bodies = foot.wheel_set() if variant == "wheel" else []
    else:
        raise KeyError(link)
    return tuple(bodies)


def linkage(q_ankle: float = 0.0, side: str = "L") -> list[Body]:
    """Crank + rod posed for an ankle angle, in the shank frame (they move with the ankle, not with a link)."""
    bodies = [leg.crank(q_ankle), leg.rod(q_ankle)] + hardware.linkage_hardware(q_ankle)
    return bodies if side == "L" else [mirror_body(b) for b in bodies]


@functools.lru_cache(maxsize=1)
def linkage_names() -> frozenset:
    """Names of the bodies that move with the ankle linkage (posed separately from the shank)."""
    return frozenset(b.name for b in linkage(0.0))


@functools.lru_cache(maxsize=None)
def link_bodies(link: str, side: str | None, variant: str) -> tuple:
    """Bodies of a link (zero pose for the linkage, which is lumped into the shank for mass)."""
    if link == "base_link":
        bl = base.base_bodies()
        out = []
        for b in bl:
            if b.name.endswith("_L"):
                if b.name.startswith("hip_servo"):
                    _tag_servo(b, base.hip_servo_loc("L"))
                out.append(b)
                r = mirror_body(b)
                out.append(replace(r, name=b.name[:-2] + "_R"))
            else:
                out.append(b)
        return tuple(out)
    bodies = list(left_leg_link(link, variant))
    if link == "shank":
        bodies += linkage(0.0)
    if side == "R":
        bodies = [mirror_body(b) for b in bodies]
    return tuple(bodies)


def links(variant: str | None = None) -> list[tuple[str, str | None, str]]:
    """(urdf link name, side, link kind) for the current configuration."""
    variant = variant or P.foot_variant
    out = [("base_link", None, "base_link")]
    for side in ("L", "R"):
        for lk in LEG_LINKS:
            if lk == "wheel" and variant != "wheel":
                continue
            out.append((link_name(lk, side), side, lk))
    return out
