"""Analytic checks on the electrostatics core."""

import numpy as np
import pytest

from emsim.core import PointCharge, Scene
from emsim.core.entities import radius_for_charge
from emsim.units import K_COULOMB


@pytest.fixture
def single():
    s = Scene()
    s.add_charge(PointCharge(q=1.0, position=[0.0, 0.0, 0.0], radius=0.03))
    return s


def test_point_charge_field_and_potential(single):
    out = single.evaluate([[2.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
    assert out["E"][0] == pytest.approx([K_COULOMB / 4, 0, 0])
    assert out["V"][0] == pytest.approx(K_COULOMB / 2)
    assert out["E"][1] == pytest.approx([0, -K_COULOMB / 16, 0])
    assert out["V"][1] == pytest.approx(K_COULOMB / 4)


def test_inside_the_ball_uses_the_solid_sphere_solution(single):
    a = single.charges[0].radius
    out = single.evaluate([[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [a, 0.0, 0.0]])
    # E vanishes at the centre, grows linearly, and matches Coulomb at r = a
    assert out["E"][0] == pytest.approx([0, 0, 0])
    assert out["E"][1][0] == pytest.approx(K_COULOMB * (a / 2) / a**3)
    assert out["E"][2][0] == pytest.approx(K_COULOMB / a**2)
    # V is finite at the centre and continuous at the surface
    assert out["V"][0] == pytest.approx(1.5 * K_COULOMB / a)
    assert out["V"][2] == pytest.approx(K_COULOMB / a)


def test_superposition():
    s = Scene()
    s.add_charge(PointCharge(q=3e-9, position=[0.4, -0.2, 0.1]))
    s.add_charge(PointCharge(q=-7e-9, position=[-0.3, 0.5, -0.2]))
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    total = s.evaluate(pts)
    parts = []
    for c in s.charges:
        one = Scene()
        one.add_charge(PointCharge(q=c.q, position=c.position, radius=c.radius))
        parts.append(one.evaluate(pts))
    assert total["E"] == pytest.approx(sum(p["E"] for p in parts))
    assert total["V"] == pytest.approx(sum(p["V"] for p in parts))


def test_dipole_midplane_potential_vanishes():
    s = Scene()
    s.add_charge(PointCharge(q=5e-9, position=[-0.5, 0.0, 0.0]))
    s.add_charge(PointCharge(q=-5e-9, position=[0.5, 0.0, 0.0]))
    mid = np.array([[0.0, y, z] for y in (-1, 0, 1) for z in (-1, 0, 2)])
    assert s.evaluate(mid)["V"] == pytest.approx(np.zeros(len(mid)), abs=1e-15)


def test_newtons_third_law_and_coulomb_magnitude():
    s = Scene()
    s.add_charge(PointCharge(q=2e-9, position=[-0.5, 0.0, 0.0]))
    s.add_charge(PointCharge(q=-3e-9, position=[0.5, 0.0, 0.0]))
    F = s.forces()
    assert F[0] == pytest.approx(-F[1])
    expected = K_COULOMB * 2e-9 * 3e-9 / 1.0**2
    assert np.linalg.norm(F[0]) == pytest.approx(expected)
    assert F[0][0] > 0            # opposite charges attract


def test_self_force_is_excluded():
    s = Scene()
    s.add_charge(PointCharge(q=1e-8, position=[0.1, 0.2, 0.3]))
    assert s.forces()[0] == pytest.approx([0, 0, 0])


def test_field_is_curl_free():
    """A conservative field: circulation around a closed loop must vanish."""
    s = Scene()
    s.add_charge(PointCharge(q=4e-9, position=[0.13, -0.07, 0.21]))
    s.add_charge(PointCharge(q=-2e-9, position=[-0.4, 0.3, -0.1]))
    n, r = 4000, 0.8
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)   # no duplicated endpoint
    loop = np.stack([r * np.cos(th), r * np.sin(th), 0.35 * np.ones_like(th)], 1)
    seg = np.roll(loop, -1, axis=0) - loop
    mid = loop + 0.5 * seg                              # midpoint rule, periodic
    dots = np.einsum("ij,ij->i", s.evaluate(mid)["E"], seg)
    assert abs(dots.sum()) < 1e-6 * np.abs(dots).sum()


def test_gradient_of_potential_is_minus_e():
    s = Scene()
    s.add_charge(PointCharge(q=6e-9, position=[0.2, 0.0, -0.1]))
    s.add_charge(PointCharge(q=-4e-9, position=[-0.25, 0.15, 0.3]))
    p = np.array([0.5, -0.4, 0.6])
    h = 1e-6
    grad = np.array([
        (s.evaluate([p + h * e])["V"][0] - s.evaluate([p - h * e])["V"][0]) / (2 * h)
        for e in np.eye(3)])
    assert -grad == pytest.approx(s.evaluate([p])["E"][0], rel=1e-6)


def test_gauss_law_over_an_enclosing_sphere():
    """Flux through a closed surface equals q_enc / eps0."""
    from emsim.units import EPS0

    s = Scene()
    s.add_charge(PointCharge(q=7e-9, position=[0.1, 0.05, -0.02]))
    s.add_charge(PointCharge(q=-3e-9, position=[-0.2, 0.1, 0.15]))
    # Fibonacci sphere of radius 2 (well outside both charges)
    n, R = 20000, 2.0
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    nrm = np.stack([np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    E = s.evaluate(nrm * R)["E"]
    flux = np.einsum("ij,ij->i", E, nrm).mean() * 4 * np.pi * R**2
    assert flux == pytest.approx(4e-9 / EPS0, rel=1e-3)


def test_radius_grows_as_cube_root_of_charge():
    r1 = radius_for_charge(10e-9)
    r8 = radius_for_charge(80e-9)
    assert r8 / r1 == pytest.approx(2.0, rel=1e-6)
    # and is clamped at the extremes
    assert radius_for_charge(1e-3) < 10 * r1
    assert radius_for_charge(1e-30) > 0


def test_empty_scene_is_safe():
    s = Scene()
    out = s.evaluate([[0.0, 0.0, 0.0]])
    assert out["E"].shape == (1, 3) and out["V"].shape == (1,)
    assert s.forces().shape == (0, 3)
