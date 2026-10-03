"""Mass, centre of mass and inertia from the solids (OCP GProp), combined per link with the parallel-axis
theorem. Units here: g, mm, g·mm². Conversion to SI happens only in urdf.py.

Printed parts: mass = volume × effective density (robot.yaml materials). Purchased parts: the given mass,
spread uniformly over the simplified shape (density = mass / volume).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from params import P
from parts.common import Body


@dataclass
class MassProps:
    mass: float                 # g
    com: np.ndarray             # mm (link frame)
    inertia: np.ndarray         # g·mm², about the CoM, link-frame axes


def solid_props(shape) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume (mm³), centroid (mm) and inertia about the centroid for unit density (mm⁵)."""
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g)
    v = g.Mass()
    c = g.CentreOfMass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return v, np.array([c.X(), c.Y(), c.Z()]), I


def density(b: Body, volume: float) -> float:
    """g/mm³."""
    if b.material == "purchased":
        return b.mass / volume
    return P.materials[b.material]["density"] / 1000.0


def body_props(b: Body) -> MassProps:
    v, c, I = solid_props(b.shape)
    if v <= 0:
        raise ValueError(f"{b.name}: non-positive volume {v}")
    rho = density(b, v)
    return MassProps(rho * v, c, rho * I)


def combine(props: list[MassProps]) -> MassProps:
    m = sum(p.mass for p in props)
    c = sum(p.mass * p.com for p in props) / m
    I = np.zeros((3, 3))
    for p in props:
        d = p.com - c
        I += p.inertia + p.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return MassProps(m, c, I)


def link_props(bodies) -> MassProps:
    return combine([body_props(b) for b in bodies])


def box_inertia(m: float, size) -> np.ndarray:
    a, b, c = size
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
