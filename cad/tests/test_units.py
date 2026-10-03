"""Unit conversion and mass-property tests (run: conda run -n cad pytest cad/tests -q)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mass import MassProps, body_props, box_inertia, combine  # noqa: E402
from parts.common import Body  # noqa: E402
from urdf import si_inertia, si_length, si_mass  # noqa: E402


def pla_cube():
    # 10 mm cube; robot.yaml has no plain-PLA material, so tag it as a purchased part of the PLA mass
    return Body("cube", Box(10, 10, 10), "purchased", mass=1.24)


def test_pla_cube_mass_and_inertia_in_si():
    p = body_props(pla_cube())
    assert p.mass == pytest.approx(1.24, rel=1e-9)                       # g
    assert si_mass(p.mass) == pytest.approx(1.24e-3)                     # kg
    I_c = 1.24 * 10.0**2 / 6.0                                           # g·mm² about a central axis
    assert np.allclose(np.diag(p.inertia), I_c, rtol=1e-6)
    assert si_inertia(p.inertia)[0, 0] == pytest.approx(1.24e-3 * 0.01**2 / 6.0, rel=1e-6)   # kg·m²
    assert np.allclose(p.inertia - np.diag(np.diag(p.inertia)), 0.0, atol=1e-9)


def test_printed_density_from_material_tag():
    # petg_structural effective density (robot.yaml) x 1 cm³
    from params import P
    b = Body("c", Box(10, 10, 10), "petg_structural")
    assert body_props(b).mass == pytest.approx(P.materials.petg_structural.density, rel=1e-9)


def test_missing_material_tag_is_an_error():
    with pytest.raises(ValueError):
        Body("x", Box(1, 1, 1), "steel_default_from_exporter")
    with pytest.raises(ValueError):
        Body("x", Box(1, 1, 1), "purchased")                                # purchased without mass


def test_length_conversion():
    assert np.allclose(si_length([1000.0, -80.0, 28.0]), [1.0, -0.08, 0.028])


def test_parallel_axis_two_cubes():
    a = body_props(Body("a", Pos(-10, 0, 0) * Box(10, 10, 10), "purchased", mass=1.0))
    b = body_props(Body("b", Pos(10, 0, 0) * Box(10, 10, 10), "purchased", mass=1.0))
    c = combine([a, b])
    assert c.mass == pytest.approx(2.0)
    assert np.allclose(c.com, 0.0, atol=1e-9)
    # Iyy = 2·(m a²/6) + 2·m·d²  with d = 10 mm
    assert c.inertia[1, 1] == pytest.approx(2 * (100 / 6) + 2 * 100, rel=1e-6)
    assert c.inertia[0, 0] == pytest.approx(2 * (100 / 6), rel=1e-6)


def test_cylinder_inertia_vs_formula():
    m, r, h = 3.0, 5.0, 20.0
    p = body_props(Body("cy", Cylinder(r, h), "purchased", mass=m))
    assert p.inertia[2, 2] == pytest.approx(0.5 * m * r * r, rel=1e-4)
    assert p.inertia[0, 0] == pytest.approx(m * (3 * r * r + h * h) / 12, rel=1e-4)


def test_box_formula_helper():
    assert np.allclose(box_inertia(1.24, (10, 10, 10)), np.eye(3) * 1.24 * 100 / 6)
