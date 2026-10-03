"""Feetech HD-1910-C001 from the vendor STEP.

* validates the import (units, solid validity) — on failure it raises; nothing is "fixed" here
* defines the servo frame S: origin = output-shaft axis ∩ output-horn mounting face, +Z = output shaft
  direction (out of the output horn), +Y = from the shaft toward the far (long) end of the case
* measures the key dimensions from the STEP (shaft position, horn face, mounting holes, cable exits)
* builds the simplified shape used for pockets, mass/inertia and the interference sweep

The vendor file is never modified; the transform STEP -> S lives in `STEP_TO_S` below.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math

from build123d import (Axis, Box, BoundBox, Compound, Cylinder, GeomType, Location, Pos, Rot, Solid,
                       import_step)

from OCP.BRepAdaptor import BRepAdaptor_Surface

from params import OUT, P, ROOT

STEP = ROOT / P.servo.vendor_step
TOL_BBOX = 0.3


class ServoImportError(RuntimeError):
    pass


CACHE = OUT / "cache"


def _hash() -> str:
    return hashlib.sha256(STEP.read_bytes()).hexdigest()[:16]


def _hole_axis_xy(face) -> tuple:
    """Axis location (x, y) of a cylindrical face (both half-faces of a hole give the same axis)."""
    ax = BRepAdaptor_Surface(face.wrapped).Cylinder().Axis().Location()
    return (round(ax.X(), 3), round(ax.Y(), 3))


@functools.lru_cache(maxsize=1)
def raw() -> Compound:
    return import_step(str(STEP))


def _case_solids(c: Compound):
    """Solids of the case body: every solid whose section spans the full 20 x 34 footprint."""
    out = []
    for s in c.solids():
        bb = s.bounding_box()
        if bb.size.X > 19.0 and bb.size.Y > 33.0:
            out.append(s)
    return out


@functools.lru_cache(maxsize=1)
def measure() -> dict:
    """Measure the STEP in its own coordinates (cached by file hash). Raises ServoImportError on failure."""
    f = CACHE / f"servo_measure_{_hash()}.json"
    if f.exists():
        return json.loads(f.read_text())
    m = _measure()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(m, indent=1))
    return m


def _measure() -> dict:
    c = raw()
    if not c.is_valid:
        raise ServoImportError("vendor STEP is not valid geometry")
    sols = c.solids()
    bad = [i for i, s in enumerate(sols) if not (s.is_valid and s.volume > 0)]
    if bad:
        raise ServoImportError(f"vendor STEP solids not valid/closed: {bad}")
    case = _case_solids(c)
    cb = BoundBox.from_topo_ds(Compound(case).wrapped)
    # the case box: full-width section between the horn faces
    full = c.bounding_box()
    # shaft: the horn discs are the Ø16 cylinders; output horn = the one carrying the 25T spline (Ø~5)
    horns = []
    spline_z = []
    for s in sols:
        bb = s.bounding_box()
        if abs(bb.size.X - 16.0) < 0.3 and abs(bb.size.Y - 16.0) < 0.3 and bb.size.Z < 4.5:
            horns.append(s)
        for f in s.faces():
            if f.geom_type == GeomType.CYLINDER and abs(f.radius - 2.5) < 0.05:
                spline_z.append(f.center().Z)
    if len(horns) != 2:
        raise ServoImportError(f"expected 2 horn discs, found {len(horns)}")
    horns.sort(key=lambda s: s.bounding_box().max.Z)
    idle, out = horns
    ob, ib = out.bounding_box(), idle.bounding_box()
    shaft_xy = ((ob.min.X + ob.max.X) / 2, (ob.min.Y + ob.max.Y) / 2)
    # case section: full footprint (20 x 34) between z where the section is full
    def full_section(z):
        sec = c.intersect(Box(100, 100, 0.01).moved(Location((0, 0, z))))
        if not sec:
            return False
        bb = Compound(list(sec)).bounding_box()
        return bb.size.X > 19.5 and bb.size.Y > 33.5

    zs = [ob.max.Z - 0.5 * k for k in range(int((ob.max.Z - ib.min.Z) / 0.5) + 1)]
    inside = [z for z in zs if full_section(z)]

    def edge(a, b):                       # full_section(a) != full_section(b): bisect to 0.01 mm
        fa = full_section(a)
        while abs(b - a) > 0.01:
            m_ = (a + b) / 2
            if full_section(m_) == fa:
                a = m_
            else:
                b = m_
        return (a + b) / 2

    case_top = max(s.bounding_box().max.Z for s in case)      # flat top face of the case (edges are chamfered)
    case_bot = edge(min(inside), min(inside) - 0.5)
    # case mounting holes: Ø1.6 blind holes on the horn faces (cylinders r 0.8 in the case solids)
    holes = set()
    for s in case:
        for f in s.faces():
            if f.geom_type == GeomType.CYLINDER and abs(f.radius - 0.8) < 0.02:
                holes.add(_hole_axis_xy(f))
    # horn holes: Ø1.6 on the output horn
    hh = set()
    for f in out.faces():
        if f.geom_type == GeomType.CYLINDER and abs(f.radius - 0.8) < 0.02:
            x, y = _hole_axis_xy(f)
            hh.add((round(x - shaft_xy[0], 2), round(y - shaft_xy[1], 2)))
    pcd = sorted(set(round(2 * math.hypot(x, y), 1) for x, y in hh))
    # cable connectors: the two small solids recessed into the ±X faces
    conns = [s.bounding_box() for s in sols if 3.0 < s.bounding_box().size.X < 4.0 and 7.0 < s.bounding_box().size.Y < 8.5]
    return dict(
        full_bbox=(tuple(full.min), tuple(full.max)),
        case_bbox=((cb.min.X, cb.min.Y, case_bot), (cb.max.X, cb.max.Y, case_top)),
        shaft_xy=shaft_xy, out_horn_face=ob.max.Z, out_horn_bottom=ob.min.Z, idle_horn_face=ib.min.Z,
        horn_diameter=ob.size.X, horn_hole_pcd=pcd, horn_hole_count=len(hh),
        mount_holes_step=sorted(holes),
        mount_hole_depth=max(s.bounding_box().max.Z for s in case) - min(
            f.bounding_box().min.Z for s in case for f in s.faces()
            if f.geom_type == GeomType.CYLINDER and abs(f.radius - 0.8) < 0.02 and f.bounding_box().max.Z > 0),
        connectors=[((b.min.X + b.max.X) / 2, (b.min.Y + b.max.Y) / 2, (b.min.Z + b.max.Z) / 2,
                     b.min.X, b.max.X) for b in conns],
        spline_found=bool(spline_z),
    )


def step_to_s() -> Location:
    """STEP coordinates -> servo frame S (translate shaft/horn face to the origin, turn the long end to +Y)."""
    m = measure()
    sx, sy = m["shaft_xy"]
    far_y = m["case_bbox"][0][1] if abs(m["case_bbox"][0][1] - sy) > abs(m["case_bbox"][1][1] - sy) else m["case_bbox"][1][1]
    flip = far_y < sy                                         # long end at -Y in the STEP -> rotate 180° about Z
    loc = Pos(0, 0, -m["out_horn_face"]) * Pos(-sx, -sy, 0)
    return (Rot(0, 0, 180) * loc) if flip else loc


def to_s(p):
    """A STEP point -> S frame (x, y, z)."""
    v = step_to_s() * Pos(*p)
    t = v.position
    return (t.X, t.Y, t.Z)


def validate() -> dict:
    """Spec checks right after import. Returns a dict of results; raises on failure."""
    m = measure()
    (x0, y0, z0), (x1, y1, z1) = m["case_bbox"]
    case = (x1 - x0, y1 - y0, z1 - z0)
    span = m["out_horn_face"] - m["idle_horn_face"]
    exp = P.servo.geometry
    res = {
        "case_size": case,
        "case_size_expected": tuple(exp.case),
        "horn_to_horn": span,
        "full_bbox": tuple(b - a for a, b in zip(*m["full_bbox"])),
    }
    errs = [abs(a - b) for a, b in zip(case, exp.case)]
    if max(errs) > TOL_BBOX:
        raise ServoImportError(f"case bbox {case} differs from {tuple(exp.case)} by {max(errs):.2f} mm (units?)")
    if abs(span - 29.0) > TOL_BBOX:
        raise ServoImportError(f"horn-to-horn {span:.2f} mm differs from the datasheet 29 mm")
    res["ok"] = True
    return res


def measured_geometry() -> dict:
    """Key dimensions in the S frame, in the same layout as robot.yaml servo.geometry."""
    m = measure()
    (x0, y0, z0), (x1, y1, z1) = m["case_bbox"]
    c0, c1 = to_s((x0, y0, z0)), to_s((x1, y1, z1))
    lo = [min(a, b) for a, b in zip(c0, c1)]
    hi = [max(a, b) for a, b in zip(c0, c1)]
    holes = sorted((round(to_s((x, y, 0))[0], 2) + 0.0, round(to_s((x, y, 0))[1], 2) + 0.0) for x, y in m["mount_holes_step"])
    conns = []
    for cx, cy, cz, xa, xb in m["connectors"]:
        p = to_s((cx, cy, cz))
        side = 1.0 if p[0] > 0 else -1.0
        conns.append([side * (hi[0] - lo[0]) / 2, round(p[1], 2), round(p[2], 2)])
    return dict(
        case=[round(hi[i] - lo[i], 2) for i in range(3)],
        case_center=[round((hi[i] + lo[i]) / 2, 2) for i in range(3)],
        horn_diameter=round(m["horn_diameter"], 2),
        horn_thickness=round(m["out_horn_face"] - m["out_horn_bottom"], 2),
        horn_face_to_case=round(-hi[2], 2),
        idle_horn_face_z=round(m["idle_horn_face"] - m["out_horn_face"], 2),
        horn_hole_pcd=m["horn_hole_pcd"][0] if len(m["horn_hole_pcd"]) == 1 else m["horn_hole_pcd"],
        mount_hole_depth=round(m["mount_hole_depth"], 2),
        horn_hole_count=m["horn_hole_count"],
        mount_holes=[list(h) for h in holes],
        cable_exit=sorted(conns),
    )


# ------------------------------------------------------------------ shapes in the S frame
@functools.lru_cache(maxsize=1)
def vendor_shape() -> Compound:
    """Full vendor model in the S frame (visuals, assembled STEP, URDF visual); cached as BREP."""
    from build123d import export_brep, import_brep
    f = CACHE / f"servo_S_{_hash()}.brep"
    if f.exists():
        return import_brep(str(f))
    shp = raw().moved(step_to_s())
    CACHE.mkdir(parents=True, exist_ok=True)
    export_brep(shp, str(f))
    return shp


def simplified(clearance: float = 0.0, horns: bool = True) -> Solid:
    """Case box + both horn discs (+ horn screw heads), from robot.yaml servo.geometry.

    `clearance` grows the case for pockets (horns are left out of pockets: they are separate cut-outs).
    """
    g = P.servo.geometry
    cx, cy, cz = g.case_center
    case = Pos(cx, cy, cz) * Box(g.case[0] + 2 * clearance, g.case[1] + 2 * clearance, g.case[2] + 2 * clearance)
    if not horns:
        return case
    r = g.horn_diameter / 2
    t = g.horn_face_to_case
    out = Pos(0, 0, -t / 2) * Cylinder(r, t)
    idle = Pos(0, 0, g.idle_horn_face_z + t / 2) * Cylinder(r, t)
    # horn screw heads (measured protrusion: +0.8 / -0.4 mm in the vendor model)
    head_t = P.servo.horn_screw_head
    hd = Pos(0, 0, head_t[0] / 2) * Cylinder(2.5, head_t[0])
    hd2 = Pos(0, 0, g.idle_horn_face_z - head_t[1] / 2) * Cylinder(2.5, head_t[1])
    return case.fuse(out, idle, hd, hd2).clean()


def simplified_vs_step() -> tuple:
    """Bounding-box difference (per axis, both ends, mm) between the simplified shape and the vendor model."""
    a = simplified().bounding_box()
    b = vendor_shape().bounding_box()
    return tuple(round(max(abs(tuple(a.min)[i] - tuple(b.min)[i]), abs(tuple(a.max)[i] - tuple(b.max)[i])), 3)
                 for i in range(3))
