"""Geometry and discretisation of charged bodies."""

import math

import numpy as np
import pytest

from emsim.core.bodies import (Disk, Line, Loop, Material, Sheet, Sphere,
                               orthonormal_frame)


def _bodies():
    return [
        (Line(length=1.0), 1.0),
        (Loop(radius=0.4), 2 * math.pi * 0.4),
        (Sheet(width=0.8, height=0.6), 0.48),
        (Disk(radius=0.5), math.pi * 0.25),
        (Sphere(radius=0.3), 4 * math.pi * 0.09),
    ]


@pytest.mark.parametrize("body, total", _bodies())
def test_site_measures_sum_to_the_exact_geometric_total(body, total):
    s = body.surface_sites(300)
    assert s.measure.sum() == pytest.approx(total, rel=1e-9)
    assert np.all(s.measure > 0)
    assert s.measure.std() == pytest.approx(0.0, abs=1e-15)   # equal measure


def test_insulating_sphere_fills_its_volume():
    b = Sphere(radius=0.3, material=Material.INSULATOR)
    s = b.sites(500)
    assert s.measure.sum() == pytest.approx(4 / 3 * math.pi * 0.3**3, rel=1e-9)
    r = np.linalg.norm(s.pos - b.position, axis=1)
    assert r.max() < 0.3
    assert r.min() < 0.05                       # actually reaches the centre
    # uniform in volume => radii distributed as r^3
    assert np.median(r) == pytest.approx(0.3 * 0.5 ** (1 / 3), rel=0.05)


def test_conducting_sphere_puts_every_site_on_the_surface():
    b = Sphere(radius=0.3)
    r = np.linalg.norm(b.sites(400).pos - b.position, axis=1)
    assert r == pytest.approx(np.full(400, 0.3))


@pytest.mark.parametrize("body, _", _bodies())
def test_sites_are_translated_and_oriented_with_the_body(body, _):
    body.position = np.array([0.3, -0.2, 0.7])
    body.axis = np.array([1.0, 1.0, 0.0])
    body.__post_init__()
    s = body.sites(200)
    assert np.linalg.norm(s.pos.mean(axis=0) - body.position) < 0.05
    if body.kind in ("sheet", "disk"):
        # every site lies in the plane normal to the axis
        assert (s.pos - body.position) @ body.axis == pytest.approx(
            np.zeros(len(s)), abs=1e-12)


def test_orthonormal_frame_is_right_handed():
    for axis in ([0, 0, 1], [1, 0, 0], [0.3, -0.5, 0.8], [0, 0, -1]):
        u, v, w = orthonormal_frame(axis)
        assert np.cross(u, v) == pytest.approx(w)
        assert u @ v == pytest.approx(0, abs=1e-12)
        assert np.linalg.norm(u) == pytest.approx(1)


# -- ray casting -----------------------------------------------------------
def test_sphere_raycast_hits_the_near_face():
    b = Sphere(radius=0.3, position=[0.0, 0.0, 0.0])
    t = b.raycast(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    assert t == pytest.approx(1.7)
    assert b.raycast(np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, -1.0])) is None


def test_plate_raycast_respects_its_bounds():
    b = Sheet(width=0.8, height=0.4, axis=[0, 0, 1])
    d = np.array([0.0, 0.0, -1.0])
    assert b.raycast(np.array([0.3, 0.1, 1.0]), d) == pytest.approx(1.0)
    assert b.raycast(np.array([0.5, 0.0, 1.0]), d) is None     # past the width
    assert b.raycast(np.array([0.0, 0.3, 1.0]), d) is None     # past the height


def test_disk_raycast_respects_its_radius():
    b = Disk(radius=0.4, axis=[0, 0, 1])
    d = np.array([0.0, 0.0, -1.0])
    assert b.raycast(np.array([0.2, 0.2, 1.0]), d) == pytest.approx(1.0)
    assert b.raycast(np.array([0.35, 0.35, 1.0]), d) is None   # outside r=0.4


def test_line_and_loop_raycast_near_the_wire():
    line = Line(length=1.0, axis=[0, 0, 1], wire_radius=0.02)
    assert line.raycast(np.array([1.0, 0.0, 0.1]),
                        np.array([-1.0, 0.0, 0.0])) is not None
    assert line.raycast(np.array([1.0, 0.0, 0.9]),
                        np.array([-1.0, 0.0, 0.0])) is None    # beyond the end

    loop = Loop(radius=0.4, axis=[0, 0, 1], wire_radius=0.02)
    assert loop.raycast(np.array([0.4, 0.0, 1.0]),
                        np.array([0.0, 0.0, -1.0])) is not None
    assert loop.raycast(np.array([0.0, 0.0, 1.0]),
                        np.array([0.0, 0.0, -1.0])) is None    # through the hole


def test_raycast_ignores_geometry_behind_the_camera():
    b = Sphere(radius=0.3)
    assert b.raycast(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0])) is None


@pytest.mark.parametrize("body, _", _bodies())
def test_every_body_produces_a_renderable_mesh(body, _):
    v, f = body.mesh()
    assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 2
    assert f.ndim == 2 and f.shape[1] == 3 and len(f) > 0
    assert f.max() < len(v)
    assert np.isfinite(v).all()
