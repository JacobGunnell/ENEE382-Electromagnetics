"""Field-line seeding and tracing.

The geometric claims a field-line picture makes -- lines leave conductors
perpendicular to the surface, they begin and end on charge, and how many there
are is proportional to how much charge there is -- are all checkable, so they
are checked here rather than left to the eye.
"""

import numpy as np
import pytest

from emsim.core import Disk, PointCharge, Scene, Sheet, Sphere
from emsim.units import K_COULOMB
from emsim.viz.fieldlines import (collect_seeds, direction_markers, segments,
                                  trace)

STEP = 1.0 / 60.0


def _dipole(qa=10e-9, qb=-10e-9):
    s = Scene()
    a = s.add_charge(PointCharge(q=qa, position=[-0.35, 0.0, 0.0]))
    b = s.add_charge(PointCharge(q=qb, position=[0.35, 0.0, 0.0]))
    return s, a, b


def _run(scene, n=120, domain=1.0):
    seeds, signs = collect_seeds(scene, n, domain)
    return trace(scene, seeds, signs, domain=domain, step=domain / 60.0)


# -- seeding ---------------------------------------------------------------
def test_line_count_is_proportional_to_charge():
    """The convention that makes line density mean something."""
    s, _, _ = _dipole(9e-9, -3e-9)
    _, signs = collect_seeds(s, 120, 1.0)
    assert (signs > 0).sum() / (signs < 0).sum() == pytest.approx(3.0, rel=0.1)


def test_density_argument_controls_the_number_of_lines():
    s, _, _ = _dipole()
    counts = [len(collect_seeds(s, n, 1.0)[0]) for n in (40, 120, 300)]
    assert counts[0] < counts[1] < counts[2]
    assert counts[1] == pytest.approx(120, abs=4)


def test_seeds_start_outside_the_bodies_they_belong_to():
    s = Scene()
    for body in (Sphere(radius=0.3, position=[-0.5, 0, 0], n_sites=150),
                 Disk(radius=0.3, position=[0.5, 0, 0], n_sites=150)):
        body.charge = 5e-9
        s.add_body(body)
    seeds, _ = collect_seeds(s, 100, 1.0)
    assert len(seeds)
    for b in s.bodies:
        assert not b.contains(seeds).any()


def test_a_neutral_conductor_still_grows_lines():
    """Induced charge is charge: weight by |q| summed, not by the net."""
    s = Scene()
    plate = s.add_body(Sheet(width=1.0, height=1.0, n_sites=200))
    s.add_charge(PointCharge(q=6e-9, position=[0.0, 0.0, 0.4]))
    assert plate.charge == 0.0
    seeds, signs = collect_seeds(s, 100, 1.0)
    # both the point charge and the plate contribute seeds
    near_plate = np.abs(seeds[:, 2]) < 0.2
    assert near_plate.sum() > 5


def test_empty_scene_seeds_nothing():
    seeds, signs = collect_seeds(Scene(), 100, 1.0)
    assert len(seeds) == 0 and len(signs) == 0
    paths, mags, kept = trace(Scene(), seeds, signs, domain=1.0, step=STEP)
    assert paths == [] and mags == [] and len(kept) == 0


def test_uncharged_scene_seeds_nothing():
    s = Scene()
    s.add_charge(PointCharge(q=0.0, position=[0, 0, 0]))
    assert len(collect_seeds(s, 50, 1.0)[0]) == 0


# -- tracing geometry ------------------------------------------------------
def test_lines_around_a_lone_charge_are_radial():
    s = Scene()
    c = s.add_charge(PointCharge(q=8e-9, position=[0.1, -0.2, 0.05]))
    paths, mags, _ = _run(s, 60)
    assert len(paths) > 30
    for p in paths:
        d = p - c.position
        u = d / np.linalg.norm(d, axis=1)[:, None]
        # every point on the line lies along the same ray
        assert np.degrees(np.arccos(np.clip(u @ u[0], -1, 1))).max() < 1.0


def test_field_strength_recorded_along_a_line_matches_coulomb():
    s = Scene()
    c = s.add_charge(PointCharge(q=8e-9, position=[0.0, 0.0, 0.0]))
    paths, mags, _ = _run(s, 40)
    p, m = paths[0], mags[0]
    r = np.linalg.norm(p - c.position, axis=1)
    far = r > 3 * c.radius          # outside the regularisation shell
    assert far.sum() > 10
    assert np.abs(m[far] / (K_COULOMB * 8e-9 / r[far] ** 2) - 1).max() < 1e-9


def test_lines_leave_a_conductor_perpendicular_to_its_surface():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=400))
    b.charge = 5e-9
    paths, _, _ = _run(s, 80)
    assert len(paths) > 40
    ang = []
    for p in paths:
        n_hat = p[0] - b.position
        n_hat /= np.linalg.norm(n_hat)
        t = p[2] - p[0]
        ang.append(np.degrees(np.arccos(np.clip(n_hat @ t / np.linalg.norm(t),
                                                -1, 1))))
    assert np.mean(ang) < 1.0
    assert np.max(ang) < 5.0


def test_dipole_lines_terminate_on_the_negative_charge():
    s, _, neg = _dipole()
    paths, _, signs = _run(s, 160)
    ends = np.array([p[-1] for p, g in zip(paths, signs) if g > 0])
    landed = np.linalg.norm(ends - neg.position, axis=1) < 0.1
    escaped = np.abs(ends).max(axis=1) > 0.95
    assert landed.mean() > 0.5
    assert (landed | escaped).all()      # every line ends somewhere definite


def test_lines_terminate_on_a_conductor_rather_than_wandering_inside_it():
    s = Scene()
    a = s.add_body(Disk(radius=0.45, position=[0, 0, -0.18], n_sites=300))
    b = s.add_body(Disk(radius=0.45, position=[0, 0, 0.18], n_sites=300))
    a.charge, b.charge = 4e-9, -4e-9
    paths, _, _ = _run(s, 120)
    assert len(paths) > 60
    # no line should be left running at the step cap
    assert max(len(p) for p in paths) < 400
    for p in paths:
        assert not a.contains(p[:-1]).any()
        assert not b.contains(p[:-1]).any()


def test_no_line_leaves_the_region_of_interest():
    s, _, _ = _dipole()
    paths, _, _ = _run(s, 120)
    for p in paths:
        assert np.abs(p).max() <= 1.0 + 1e-9


def test_traced_points_are_finite():
    s, _, _ = _dipole()
    paths, mags, _ = _run(s, 80)
    for p, m in zip(paths, mags):
        assert np.isfinite(p).all() and np.isfinite(m).all()
        assert len(p) == len(m) >= 2


# -- rendering helpers -----------------------------------------------------
def test_segments_pairs_every_vertex_with_its_own_strength():
    s, _, _ = _dipole()
    paths, mags, _ = _run(s, 60)
    verts, m = segments(paths, mags)
    assert len(verts) == len(m)
    # one pair of vertices per interior segment
    assert len(verts) == sum(2 * (len(p) - 1) for p in paths)
    assert np.isfinite(verts).all() and np.isfinite(m).all()


def test_segments_of_nothing_is_empty_not_an_error():
    verts, m = segments([], [])
    assert verts.shape == (0, 3) and m.shape == (0,)


def test_direction_markers_point_along_the_field_not_along_the_trace():
    """Lines seeded on negative charge are traced backwards; the heads must
    still show the field direction."""
    s, _, _ = _dipole()
    paths, mags, signs = _run(s, 120)
    pos, tan, mag = direction_markers(paths, mags, signs)
    assert len(pos) > 50
    E = s.evaluate(pos)["E"]
    E /= np.linalg.norm(E, axis=1)[:, None]
    assert np.einsum("ij,ij->i", E, tan).min() > 0.99
    assert len(mag) == len(pos)


def test_direction_markers_are_spread_along_each_line():
    s, _, _ = _dipole()
    paths, mags, signs = _run(s, 60)
    pos, _, _ = direction_markers(paths, mags, signs, per_line=3)
    assert 2 * len(paths) < len(pos) < 6 * len(paths)
